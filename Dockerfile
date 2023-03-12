FROM nvcr.io/nvidia/deepstream:6.1.1-devel

RUN cd /opt/nvidia/deepstream/deepstream-6.1 && \
    ./user_additional_install.sh

RUN apt-get update && apt-get install -y \ 
    python3-gi python3-dev python3-gst-1.0 python-gi-dev git python-dev \
    python3 python3-pip python3.8-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev \
    git

RUN cd sources && \
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git && \ 
    cd deepstream_python_apps/ && \ 
    git checkout v1.1.4 && \ 
    git submodule update --init

RUN apt-get install -y apt-transport-https ca-certificates -y && \
    update-ca-certificates && \
    cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/3rdparty/gst-python/ && \
    ./autogen.sh && \
    make && \
    make install

RUN cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    pip install ./pyds-1.1.4-py3-none*.whl

RUN pip install pika jsonlib-python3 protobuf opencv-python mysql-connector-python


