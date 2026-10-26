#!/usr/bin/env bash
# apt is forbidden to fetch any missing packages. Requires target administrator.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /etc/os-release
[[ "$ID" == ubuntu && ( "$VERSION_ID" == 22.04 || "$VERSION_ID" == 24.04 ) && "$(dpkg --print-architecture)" == amd64 ]] || { echo 'Requires Ubuntu 22.04/24.04 amd64'; exit 1; }
PACKAGES="$(dirname -- "$ROOT")/offline/ubuntu-$VERSION_ID-amd64"
[[ -f "$PACKAGES/packages.json" ]] || { echo "Missing Ubuntu package bundle: $PACKAGES"; exit 1; }
echo 'Installing local Ubuntu runtime libraries. No network downloads are permitted.'
# apt resolves the actual target state; never force a downgrade or broken dependency.
sudo apt-get --no-download -o Acquire::Retries=0 install "$PACKAGES"/*.deb
sudo usermod -aG video,dialout "$USER"
echo 'System setup complete. Log out and back in before using the camera/scale.'
