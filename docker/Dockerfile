# Build arguments for DeepStream version, container flavor and SDK folder.
ARG DS_VERSION=6.1.1
ARG DS_FLAVOR=devel
ARG DS_FOLDER=6.1

FROM nvcr.io/nvidia/deepstream:${DS_VERSION}-${DS_FLAVOR}

# Re-declare build args after FROM.
ARG DS_VERSION
ARG DS_FLAVOR
ARG DS_FOLDER

# Make version available at runtime
ENV DEEPSTREAM_VERSION=${DS_VERSION}

# Install additional multimedia modules.
RUN for dir in /opt/nvidia/deepstream/deepstream-${DS_FOLDER} /opt/nvidia/deepstream/deepstream; do \
        if [ -f "$dir/user_additional_install.sh" ]; then \
            cd "$dir" && ./user_additional_install.sh && break; \
        fi; \
    done

# Install build/runtime dependencies.
RUN apt-get update && apt-get install -y \
    python3-gi \
    python3-gi-cairo \
    python3-gst-1.0 \
    python3-dev \
    python3-pip \
    python3-venv \
    python-gi-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    git \
    cmake \
    g++ \
    build-essential \
    libglib2.0-dev \
    libglib2.0-dev-bin \
    libgstreamer1.0-dev \
    libtool \
    m4 \
    autoconf \
    automake \
    libgirepository1.0-dev \
    libcairo2-dev \
    meson \
    ninja-build \
    pkg-config \
    curl \
    apt-transport-https \
    ca-certificates && \
    if [ "$DS_VERSION" = "9.1" ]; then \
        apt-get install -y libgirepository-2.0-dev; \
    fi && \
    update-ca-certificates && \
    if python3 -m pip install --help | grep -q -- '--break-system-packages'; then \
        echo '--break-system-packages' > /tmp/pip_extra_flag; \
    else \
        touch /tmp/pip_extra_flag; \
    fi && \
    EXTRA_FLAG="$(cat /tmp/pip_extra_flag)" && \
    python3 -m pip install --no-cache-dir $EXTRA_FLAG build pycairo

# Determine SDK directory and clone deepstream_python_apps for ALL versions.
RUN if [ -d /opt/nvidia/deepstream/deepstream-${DS_FOLDER} ]; then \
        SDK_DIR="/opt/nvidia/deepstream/deepstream-${DS_FOLDER}"; \
    else \
        SDK_DIR="/opt/nvidia/deepstream/deepstream"; \
    fi && \
    mkdir -p "$SDK_DIR/sources" && \
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git \
        "$SDK_DIR/sources/deepstream_python_apps" && \
    cd "$SDK_DIR/sources/deepstream_python_apps" && \
    case "$DS_VERSION" in \
        6.1.1) git checkout v1.1.4 ;; \
        8.0) git checkout v1.2.2 ;; \
    esac && \
    git submodule update --init

# Install DeepStream Python bindings (pyds).
RUN EXTRA_FLAG="$(cat /tmp/pip_extra_flag)" && \
    PYDS_URL="" && \
    case "$DS_VERSION" in \
        6.1.1) PYDS_URL="https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.4/pyds-1.1.4-py3-none-linux_x86_64.whl" ;; \
        8.0) PYDS_URL="https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_x86_64.whl" ;; \
    esac && \
    if [ -n "$PYDS_URL" ]; then \
        PYDS_FILE="/tmp/$(basename "$PYDS_URL")" && \
        curl -fSL -o "$PYDS_FILE" "$PYDS_URL" && \
        python3 -m pip install $EXTRA_FLAG --no-deps "$PYDS_FILE"; \
    else \
        if [ -d /opt/nvidia/deepstream/deepstream-${DS_FOLDER} ]; then \
            SDK_DIR="/opt/nvidia/deepstream/deepstream-${DS_FOLDER}"; \
        else \
            SDK_DIR="/opt/nvidia/deepstream/deepstream"; \
        fi && \
        cd "$SDK_DIR/sources/deepstream_python_apps/bindings/3rdparty/gstreamer/subprojects/gst-python" && \
        meson setup build && \
        cd build && \
        ninja && \
        ninja install && \
        cd "$SDK_DIR/sources/deepstream_python_apps/bindings" && \
        export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) && \
        export CMAKE_ARGS="-DDS_PATH=$SDK_DIR" && \
        python3 -m build && \
        cd dist && \
        python3 -m pip install $EXTRA_FLAG --no-deps ./pyds-*.whl; \
    fi

# Install application dependencies.
RUN EXTRA_FLAG="$(cat /tmp/pip_extra_flag)" && \
    python3 -m pip install $EXTRA_FLAG \
    pika \
    jsonlib-python3 \
    protobuf \
    opencv-python \
    mysql-connector-python \
    requests

# Build and install the patched action-recognition custom sequence library.
# This is specific to DeepStream 8.0 (the 3D action-recognition sample with the
# custom sequence preprocess only ships there). The build_sequence_lib.sh script
# stages the patched source into the sample build dir, compiles
# libnvds_custom_sequence_preprocess.so, and installs it into both DeepStream lib
# directories. Build the image from the repo root so the COPY paths below are
# relative to the repo (e.g. `docker build -f docker/Dockerfile .`).
COPY custom_sequence_preprocess/ /opt/action-rec-build/custom_sequence_preprocess/
COPY docker/build_sequence_lib.sh /opt/action-rec-build/docker/build_sequence_lib.sh

RUN chmod +x /opt/action-rec-build/docker/build_sequence_lib.sh && \
    if [ "$DS_VERSION" = "8.0" ]; then \
        DS_VERSION=8.0 CUDA_VER=12.8 /opt/action-rec-build/docker/build_sequence_lib.sh; \
    else \
        echo "Skipping action-recognition sequence lib build (DeepStream ${DS_VERSION} does not use it)"; \
    fi && \
    rm -rf /opt/action-rec-build

# DeepStream 9.1:
# sudo docker build -f docker/Dockerfile \
#   --build-arg DS_VERSION=9.1 \
#   --build-arg DS_FLAVOR=triton-multiarch \
#   --build-arg DS_FOLDER=9.1 \
#   -t deepstreamsolutiondocker:9.1 .

# DeepStream 8.0 (also builds and installs the patched action-recognition
# custom sequence library):
# sudo docker build -f docker/Dockerfile \
#   --build-arg DS_VERSION=8.0 \
#   --build-arg DS_FLAVOR=gc-triton-devel \
#   --build-arg DS_FOLDER=8.0 \
#   -t deepstreamsolutiondocker:8.0 .

# DeepStream 6.1.1:
# sudo docker build -f docker/Dockerfile \
#   --build-arg DS_VERSION=6.1.1 \
#   --build-arg DS_FLAVOR=devel \
#   --build-arg DS_FOLDER=6.1 \
#   -t deepstreamsolutiondocker:6.1.1 .