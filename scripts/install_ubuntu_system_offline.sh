#!/usr/bin/env bash
# APT resolves packages using a local repository. Installation requires sudo.
set -Eeuo pipefail
[[ $# -eq 0 || ( $# -eq 1 && "$1" == --check ) ]] || { echo 'Usage: bash scripts/install_ubuntu_system_offline.sh [--check]'; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /etc/os-release
[[ "$ID" == ubuntu && ( "$VERSION_ID" == 22.04 || "$VERSION_ID" == 24.04 || "$VERSION_ID" == 26.04 ) && "$(dpkg --print-architecture)" == amd64 ]] || { echo 'Requires Ubuntu 22.04/24.04/26.04 amd64'; exit 1; }
PACKAGES="$(dirname -- "$ROOT")/offline/ubuntu-$VERSION_ID-amd64"
[[ -f "$PACKAGES/packages.json" ]] || { echo "Missing Ubuntu package bundle: $PACKAGES"; exit 1; }
[[ -s "$PACKAGES/roots.txt" ]] || { echo "Missing Ubuntu package list: $PACKAGES/roots.txt"; exit 1; }
mapfile -t packages < "$PACKAGES/roots.txt"
shopt -s nullglob
archives=("$PACKAGES"/*.deb)
(( ${#archives[@]} > 0 )) || { echo "No Ubuntu package archives in $PACKAGES"; exit 1; }

# Keep the package index, sources and archive cache private to this run.
CACHE="$(mktemp -d /tmp/weight-offline-apt.XXXXXXXX)"
trap 'rm -rf -- "$CACHE"' EXIT
chmod 755 "$CACHE"
mkdir -p "$CACHE/repo" "$CACHE/lists" "$CACHE/archives/partial"
cp -- "${archives[@]}" "$CACHE/repo/"
chmod 644 "$CACHE/repo/"*.deb
for deb in "$CACHE/repo/"*.deb; do
    dpkg-deb --field "$deb"
    printf 'Filename: ./%s\nSize: %s\nSHA256: %s\n\n' \
        "${deb##*/}" "$(stat -c '%s' "$deb")" "$(sha256sum < "$deb" | cut -d ' ' -f 1)"
done | gzip -c > "$CACHE/repo/Packages.gz"
printf 'deb [trusted=yes] file:%s/repo ./\n' "$CACHE" > "$CACHE/sources.list"

APT_OPTIONS=(--no-remove --no-upgrade --no-install-recommends -o Acquire::Retries=0
    -o Acquire::Languages=none
    -o Acquire::CompressionTypes::Order::=gz
    -o "Dir::Etc::sourcelist=$CACHE/sources.list" -o Dir::Etc::sourceparts=-
    -o "Dir::State::lists=$CACHE/lists" -o "Dir::Cache::archives=$CACHE/archives"
    -o Dir::Cache::pkgcache= -o Dir::Cache::srcpkgcache=)

echo "Resolving Ubuntu $VERSION_ID runtime packages from the local bundle."
apt-get "${APT_OPTIONS[@]}" update
if [[ "${1:-}" == --check ]]; then
    apt-get --simulate "${APT_OPTIONS[@]}" install "${packages[@]}"
    echo 'Offline package plan passed. No packages or user groups were changed.'
    exit 0
fi
# apt resolves the actual target state; never force a downgrade or broken dependency.
sudo apt-get "${APT_OPTIONS[@]}" install "${packages[@]}"
for command in flock v4l2-ctl; do
    command -v "$command" >/dev/null || { echo "System setup incomplete: missing $command"; exit 1; }
done
sudo usermod -aG video,dialout "${SUDO_USER:-$USER}"
echo 'System setup complete. Log out and back in before using the camera/scale.'
