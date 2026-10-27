#!/usr/bin/env bash
# apt is forbidden to fetch any missing packages. Requires target administrator.
set -Eeuo pipefail
[[ $# -eq 0 || ( $# -eq 1 && "$1" == --check ) ]] || { echo 'Usage: bash scripts/install_ubuntu_system_offline.sh [--check]'; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /etc/os-release
[[ "$ID" == ubuntu && ( "$VERSION_ID" == 22.04 || "$VERSION_ID" == 24.04 ) && "$(dpkg --print-architecture)" == amd64 ]] || { echo 'Requires Ubuntu 22.04/24.04 amd64'; exit 1; }
PACKAGES="$(dirname -- "$ROOT")/offline/ubuntu-$VERSION_ID-amd64"
[[ -f "$PACKAGES/packages.json" ]] || { echo "Missing Ubuntu package bundle: $PACKAGES"; exit 1; }
echo 'Installing local Ubuntu runtime libraries. No network downloads are permitted.'
# --no-download also disables acquiring file: URLs. Stage the archives first,
# otherwise APT rejects local .debs that are not already in its archive cache.
# A private cache avoids modifying or cleaning the machine's normal APT cache.
CACHE="$(mktemp -d /tmp/weight-offline-apt.XXXXXXXX)"
trap 'rm -rf -- "$CACHE"' EXIT
chmod 755 "$CACHE"
mkdir "$CACHE/partial"
cp -- "$PACKAGES"/*.deb "$CACHE/"
chmod 644 "$CACHE"/*.deb
APT_OPTIONS=(--no-download --no-remove -o Acquire::Retries=0 -o "Dir::Cache::archives=$CACHE")
if [[ "${1:-}" == --check ]]; then
    apt-get --simulate "${APT_OPTIONS[@]}" install "$PACKAGES"/*.deb
    echo 'Offline package plan passed. No packages or user groups were changed.'
    exit 0
fi
# apt resolves the actual target state; never force a downgrade or broken dependency.
sudo apt-get "${APT_OPTIONS[@]}" install "$PACKAGES"/*.deb
for command in flock v4l2-ctl; do
    command -v "$command" >/dev/null || { echo "System setup incomplete: missing $command"; exit 1; }
done
sudo usermod -aG video,dialout "${SUDO_USER:-$USER}"
echo 'System setup complete. Log out and back in before using the camera/scale.'
