#!/usr/bin/env bash
#
# build_sequence_lib.sh
#
# Builds and installs libnvds_custom_sequence_preprocess.so into a DeepStream 8.0
# container image. This is the "patched" per-object action-recognition library for
# gst-nvdspreprocess (keyed by tracker object_id, with per-slot pool reclamation).
#
# The library must be compiled inside the container because it links against
# DeepStream's internal headers and libraries, and the include paths in the
# Makefile are resolved relative to the stock sample's build directory.
#
# This script is intended to be invoked from a Dockerfile (RUN ...), but it can
# also be run manually inside a running container for development.
#
# Usage (inside container, FROM nvcr.io/nvidia/deepstream:8.0-...):
#   ./build_sequence_lib.sh
#
# Overridable environment variables:
#   DS_VERSION   DeepStream version (default: 8.0)
#   CUDA_VER     CUDA toolkit version the Makefile targets. Auto-detected from
#                /usr/local/cuda if not set (e.g. DS 9.1 ships CUDA 13.2).
#                Falls back to 12.8 (DS 8.0).

set -euo pipefail

DS_VERSION="${DS_VERSION:-8.0}"

# Auto-detect the CUDA toolkit version from the default /usr/local/cuda symlink,
# then fall back to the overridable default (12.8, matching DeepStream 8.0).
# The detected directory name is "cuda-13.2", so strip the "cuda-" prefix to get
# the bare version CUDA_VER is expected to hold.
if [ -z "${CUDA_VER:-}" ] && [ -L /usr/local/cuda ]; then
    _cuda_link="$(readlink -f /usr/local/cuda)"
    CUDA_VER="${_cuda_link##*cuda-}"
fi
CUDA_VER="${CUDA_VER:-12.8}"

# Where the sample build directory lives inside the image.
SAMPLE_DIR="/opt/nvidia/deepstream/deepstream-${DS_VERSION}/sources/apps/sample_apps/deepstream-3d-action-recognition/custom_sequence_preprocess"

# Directory containing the custom library source (repo root relative to the
# script). The Dockerfile copies the source tree here before RUNning this script.
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../custom_sequence_preprocess"

# The .so is installed to BOTH DeepStream lib directories; at runtime the config
# points 'custom-lib-path' at /opt/nvidia/deepstream/deepstream/lib/... .
LIB_INSTALL_DIRS=(
    "/opt/nvidia/deepstream/deepstream-${DS_VERSION}/lib"
    "/opt/nvidia/deepstream/deepstream/lib"
)

log() { printf '[build_sequence_lib] %s\n' "$*"; }
die() { printf '[build_sequence_lib] ERROR: %s\n' "$*" >&2; exit 1; }

# --- sanity checks -----------------------------------------------------------
for d in "${LIB_INSTALL_DIRS[@]}"; do
    [ -d "$d" ] || die "DeepStream lib dir not found: $d (is DS ${DS_VERSION} installed?)"
done
[ -d "$SAMPLE_DIR" ] || die "sample build dir not found: $SAMPLE_DIR"
command -v make >/dev/null 2>&1 || die "'make' not found"
[ -x "/usr/local/cuda-${CUDA_VER}/bin/nvcc" ] \
    || die "nvcc not found for CUDA_VER=${CUDA_VER} at /usr/local/cuda-${CUDA_VER}/bin/nvcc"
[ -f "$SRC_DIR/Makefile" ] || die "source Makefile not found in $SRC_DIR"

# --- stage the patched source into the sample build dir ----------------------
log "Staging patched source into $SAMPLE_DIR"
cp -f "$SRC_DIR"/sequence_image_process.cpp \
      "$SRC_DIR"/sequence_image_process.h \
      "$SRC_DIR"/seq_process_common.h \
      "$SRC_DIR"/sequence_preprocess_kernel.cu \
      "$SRC_DIR"/sequence_preprocess_kernel.h \
      "$SRC_DIR"/Makefile \
      "$SAMPLE_DIR"/

# --- build -------------------------------------------------------------------
log "Building libnvds_custom_sequence_preprocess.so (CUDA_VER=${CUDA_VER})"
make -C "$SAMPLE_DIR" clean            # removes stale *.o and the .so
CUDA_VER="$CUDA_VER" NVDS_VERSION="$DS_VERSION" make -C "$SAMPLE_DIR" all

# --- install to both lib dirs ------------------------------------------------
log "Installing libnvds_custom_sequence_preprocess.so"
for dir in "${LIB_INSTALL_DIRS[@]}"; do
    cp -f "$SAMPLE_DIR/libnvds_custom_sequence_preprocess.so" "$dir/"
done

# --- verify ------------------------------------------------------------------
for dir in "${LIB_INSTALL_DIRS[@]}"; do
    lib="$dir/libnvds_custom_sequence_preprocess.so"
    [ -s "$lib" ] || die "install failed: $lib not present"
    if command -v grep >/dev/null 2>&1 && grep -q getUnitKey "$lib"; then
        log "OK: $lib (patched symbols present)"
    else
        log "WARNING: $lib built but patched symbols not detected"
    fi
done

log "Done."
