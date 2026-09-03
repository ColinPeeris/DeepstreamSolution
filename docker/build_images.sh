#!/usr/bin/env bash
#
# build_images.sh
#
# Builds the Docker images for all supported DeepStream versions (9.1, 8.0, 6.1.1)
# from the shared docker/Dockerfile.
#
# Usage:
#   ./docker/build_images.sh          # build all three
#   ./docker/build_images.sh 9.1      # build only one version
#   ./docker/build_images.sh 9.1 8.0  # build selected versions
#
# Each image is tagged deepstreamsolutiondocker:<version>. The DS 8.0 build also
# compiles and installs the patched action-recognition custom sequence library
# (see docker/build_sequence_lib.sh).
#
# Run from the repo root so the Dockerfile's repo-relative COPY paths resolve.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log() { printf '[build_images] %s\n' "$*"; }
die() { printf '[build_images] ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Version -> (DS_FLAVOR, DS_FOLDER) mapping. These must match the build-arg
# comments at the bottom of docker/Dockerfile.
# ---------------------------------------------------------------------------
declare -A FLAVOR=(
    [9.1]=triton-multiarch
    [8.0]=gc-triton-devel
    [6.1.1]=devel
)
declare -A FOLDER=(
    [9.1]=9.1
    [8.0]=8.0
    [6.1.1]=6.1
)

# Versions to build: all by default, or the ones passed on the command line.
if [ "$#" -gt 0 ]; then
    VERSIONS=("$@")
else
    VERSIONS=(9.1 8.0 6.1.1)
fi

# Validate and deduplicate the requested versions.
declare -A SEEN=()
for ver in "${VERSIONS[@]}"; do
    [ -n "${FLAVOR[$ver]:-}" ] || die "unsupported version: $ver (expected: ${!FLAVOR[@]})"
    SEEN[$ver]=1
done

[ -f "$DOCKERFILE" ] || die "Dockerfile not found: $DOCKERFILE"
[ -f "$SCRIPT_DIR/build_sequence_lib.sh" ] || die "build_sequence_lib.sh not found in $SCRIPT_DIR"
[ -d "$REPO_ROOT/custom_sequence_preprocess" ] || die "custom_sequence_preprocess not found in $REPO_ROOT"
command -v docker >/dev/null 2>&1 || die "'docker' not found"

# ---------------------------------------------------------------------------
for ver in "${!SEEN[@]}"; do
    local_flavor="${FLAVOR[$ver]}"
    local_folder="${FOLDER[$ver]}"
    tag="deepstreamsolutiondocker:$ver"

    log "=== Building DeepStream ${ver} (flavor=${local_flavor}, folder=${local_folder}) -> ${tag} ==="
    log "Command: docker build -f ${DOCKERFILE} --build-arg DS_VERSION=${ver} --build-arg DS_FLAVOR=${local_flavor} --build-arg DS_FOLDER=${local_folder} -t ${tag} ${REPO_ROOT}"

    docker build \
        -f "$DOCKERFILE" \
        --build-arg "DS_VERSION=$ver" \
        --build-arg "DS_FLAVOR=$local_flavor" \
        --build-arg "DS_FOLDER=$local_folder" \
        -t "$tag" \
        "$REPO_ROOT"

    log "=== Built ${tag} ==="
done

log "All requested images built successfully."
