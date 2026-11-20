#!/usr/bin/env bash
# apt is forbidden to fetch any missing packages. Requires target administrator.
set -Eeuo pipefail
[[ $# -eq 0 || ( $# -eq 1 && "$1" == --check ) ]] || { echo 'Usage: bash scripts/install_ubuntu_system_offline.sh [--check]'; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /etc/os-release
[[ "$ID" == ubuntu && ( "$VERSION_ID" == 22.04 || "$VERSION_ID" == 24.04 || "$VERSION_ID" == 26.04 ) && "$(dpkg --print-architecture)" == amd64 ]] || { echo 'Requires Ubuntu 22.04/24.04/26.04 amd64'; exit 1; }
PACKAGES="$(dirname -- "$ROOT")/offline/ubuntu-$VERSION_ID-amd64"
[[ -f "$PACKAGES/packages.json" ]] || { echo "Missing Ubuntu package bundle: $PACKAGES"; exit 1; }
echo 'Installing local Ubuntu runtime libraries. No network downloads are permitted.'
# A private directory keeps the machine's APT configuration and cache intact.
CACHE="$(mktemp -d /tmp/weight-offline-apt.XXXXXXXX)"
trap 'rm -rf -- "$CACHE"' EXIT
chmod 755 "$CACHE"
mkdir "$CACHE/partial"
cp -- "$PACKAGES"/*.deb "$CACHE/"
chmod 644 "$CACHE"/*.deb
APT_OPTIONS=(--no-download --no-remove -o Acquire::Retries=0 -o "Dir::Cache::archives=$CACHE")
packages=("$PACKAGES"/*.deb)
if [[ "$VERSION_ID" == 26.04 ]]; then
    # Treat the archives as candidates so APT can keep valid newer packages on
    # a machine that already has updates. All sources point to this local copy.
    [[ -s "$PACKAGES/roots.txt" ]] || { echo 'Missing Ubuntu 26.04 package list'; exit 1; }
    mapfile -t packages < "$PACKAGES/roots.txt"
    for deb in "$CACHE"/*.deb; do
        dpkg-deb --field "$deb"
        printf 'Filename: ./%s\nSize: %s\nSHA256: %s\n\n' \
            "${deb##*/}" "$(stat -c '%s' "$deb")" "$(sha256sum < "$deb" | cut -d ' ' -f 1)"
    done > "$CACHE/Packages"
    printf 'deb [trusted=yes] file:%s ./\n' "$CACHE" > "$CACHE/sources.list"
    mkdir "$CACHE/lists"
    APT_OPTIONS=(--no-remove --no-upgrade --no-install-recommends -o Acquire::Retries=0
        -o Acquire::Languages=none
        -o "Dir::Etc::sourcelist=$CACHE/sources.list" -o Dir::Etc::sourceparts=-
        -o "Dir::State::lists=$CACHE/lists" -o "Dir::Cache::archives=$CACHE"
        -o Dir::Cache::pkgcache= -o Dir::Cache::srcpkgcache=)
    apt-get "${APT_OPTIONS[@]}" update
fi
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
