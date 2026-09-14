Setting up the environment:

1) Before running the nvidia docker, you'll need to install NVIDIA Container Toolkit.
Refer to this page: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

2) Then run the following commands to run my docker:

# Allow local root access to the X server
xhost +local:root

# Run the container (from the DeepstreamSolution repo dir, so $(pwd) binds the repo to /workspace)
sudo docker run --gpus all -it --rm \
  --net=host \
  --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --mount type=bind,src="$(pwd)",target=/workspace \
  cpeeris/deepstreamsolutiondocker:8.0  # 6.1.1 and 9.1 also supported

## Building the Docker images

The `docker/` folder contains everything needed to build the images from source
via a single shared `docker/Dockerfile`:

```
docker/
  Dockerfile            # shared multi-version build definition
  build_images.sh       # builds all (or selected) DeepStream images
  build_sequence_lib.sh # compiles + installs the patched action-recognition
                        # custom sequence library (used by the DS 8.0 build)
```

Build all three images (9.1, 8.0, 6.1.1) from the repo root:

```
sudo ./docker/build_images.sh
```

Or build only selected versions:

```
sudo ./docker/build_images.sh 9.1 8.0
```

Each image is tagged `deepstreamsolutiondocker:<version>`. The DS 8.0 build also
compiles and installs the patched per-object action-recognition library (see the
"Per-object action recognition" note below) into `/opt/nvidia/deepstream/deepstream/lib/`,
so the per-object pipeline works out of the box in that image.

Equivalent manual commands for reference:

```
# DeepStream 9.1
sudo docker build -f docker/Dockerfile \
  --build-arg DS_VERSION=9.1 \
  --build-arg DS_FLAVOR=triton-multiarch \
  --build-arg DS_FOLDER=9.1 \
  -t deepstreamsolutiondocker:9.1 .

# DeepStream 8.0 (also builds + installs the patched action-recognition library)
sudo docker build -f docker/Dockerfile \
  --build-arg DS_VERSION=8.0 \
  --build-arg DS_FLAVOR=gc-triton-devel \
  --build-arg DS_FOLDER=8.0 \
  -t deepstreamsolutiondocker:8.0 .

# DeepStream 6.1.1
sudo docker build -f docker/Dockerfile \
  --build-arg DS_VERSION=6.1.1 \
  --build-arg DS_FLAVOR=devel \
  --build-arg DS_FOLDER=6.1 \
  -t deepstreamsolutiondocker:6.1.1 .
```

3) Setup mysql (https://phoenixnap.com/kb/install-mysql-ubuntu-20-04)

Step 1: Update/Upgrade Package Repository
sudo apt update
sudo apt upgrade

Step 2: Install MySQL
sudo apt install mysql-server
mysql --version

Step 3: Securing MySQL
sudo mysql_secure_installation
- enter and renter password (we use Password!23)

Step 4: Check if MySQL Service Is Running
sudo systemctl status mysql

Step 5: Log in to MySQL Server
sudo mysql -u root


4) Install RabbitMQ

Step 1: Install Erlang

sudo apt update
sudo apt install curl software-properties-common apt-transport-https lsb-release
curl -fsSL https://packages.erlang-solutions.com/ubuntu/erlang_solutions.asc | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/erlang.gpg
echo "deb https://packages.erlang-solutions.com/ubuntu $(lsb_release -cs) contrib" | sudo tee /etc/apt/sources.list.d/erlang.list
sudo apt update
sudo apt install erlang

Step 2: Add RabbitMQ Repository to Ubuntu

curl -s https://packagecloud.io/install/repositories/rabbitmq/rabbitmq-server/script.deb.sh | sudo bash

Step 3: Install RabbitMQ Server
sudo apt update
sudo apt install rabbitmq-server
systemctl status rabbitmq-server.service
systemctl is-enabled rabbitmq-server.service

If the service is disabled, enable it:
sudo systemctl enable rabbitmq-server

## Action Recognition pipeline

`detector_tracker_classifier_actionRec_deepstream_8.json` adds a 3D action
recognition stage (resnet18_3d_rgb_hmdb5_32, TAO ActionRecognitionNet) to the
existing detect -> track -> classify pipeline:

```
streammux -> pgie_detector -> tracker -> nvdspreprocess -> sgie_actionrec_3d
         -> sgie_classifier (vehicle make) -> sgie_classifier (vehicle type)
         -> nvvidconv -> nvosd -> filesink
```

Run it inside the docker:

```
cd /workspace/
python3 pipeline_launcher.py detector_tracker_classifier_actionRec_deepstream_8.json
```

### Model setup

The 3D action model (`resnet18_3d_rgb_hmdb5_32.etlt`) is downloaded into
`models/` and the TensorRT engine (`models/resnet18_3d_rgb_hmdb5_32.etlt_b4_gpu0_fp16.engine`)
is built automatically by nvinfer on first run. Both are git-ignored.

Source on NGC:
```
ngc registry model download-version nvidia/tao/actionrecognitionnet:deployable_v1.0
```
(the `.etlt` for both 2D and 3D variants; the config uses the 3D one).

### Measuring per-model inference time

The pipeline now measures and prints a per-model inference timing summary
automatically when the run finishes (on End-of-stream). Every `nvinfer` element
(detector, action recognition, and the vehicle make/type classifiers) gets a
src-pad probe that accumulates the interval between successive output buffers;
when the pipeline exits it prints a table like:

```
===== Per-model inference timing =====
  pgie_detector_1                     batch_size=4   per_batch= 32.8 ms  per_image=  8.20 ms  (min=  0.40 max=404.32 ms, n=1441)
  sgie_actionrec_3d_2                 batch_size=4   per_batch= 32.6 ms  per_image=  8.15 ms  (min=  0.77 max= 45.46 ms, n=1441)
  sgie_classifier_3                   batch_size=4   per_batch= 32.6 ms  per_image=  8.15 ms  (min=  0.77 max= 45.45 ms, n=1441)
  sgie_classifier_4                   batch_size=4   per_batch= 32.6 ms  per_image=  8.15 ms  (min=  0.77 max= 45.40 ms, n=1441)
=====================================
```

One src-pad buffer corresponds to one inference batch, so `per_batch` is the
measured interval between output batches and `per_image` is that interval divided
by the model's `batch-size` (the amortized cost per image in a batch of N).

This is implemented in `Modules/inference_engine_builder.py` (the probe
accumulation) and printed from `Modules/pipeline_builder.py`. No setup is
required — just run the pipeline:

```
cd /workspace
python3 pipeline_launcher.py configs/detector_tracker_classifier_actionRec_deepstream_8.json
```

The reported value is the per-model processing cadence in the running pipeline
(the interval between output batches), reflecting latency including that model's
pre/post-processing. For the raw GPU-only TensorRT inference time of a single
model, bypass the pipeline with `trtexec` on the built engine:

```
/usr/src/tensorrt/bin/trtexec --loadEngine=models/resnet18_3d_rgb_hmdb5_32.etlt_b4_gpu0_fp16.engine
```

> Note: DeepStream's `NVDS_ENABLE_LATENCY_MEASUREMENT` /
> `NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT` env vars only work with NVIDIA's
> C++ `deepstream-app` / sample apps; they are ignored by custom Python
> pipelines, which is why timing is now done in the probe code above.

### Important limitation: frame-mode vs per-object

The stock `libnvds_custom_sequence_preprocess.so` keys temporal sequences by the
ROI bounding-box position. It therefore only works reliably in **full-frame mode**
(`process-on-frame=1`, the sample's native mode), where each source has one fixed
ROI. This is what the shipped config uses, and it runs to completion.

Per-tracking-target action recognition (`process-on-frame=0`, `process-on-all-objects=1`)
with this stock library does **not** complete sequence batches for moving objects
(ROI position changes every frame), so the pipeline stalls. To get per-object
action labels, the library must be reworked to correlate ROIs by tracker object-id
instead of by bounding-box position.

The patched source lives in `custom_sequence_preprocess/` in this repo (version
controlled; built by `docker/build_sequence_lib.sh`). For the DS 8.0 image it is
compiled and installed automatically at image build time, so the per-object
config (`detector_tracker_classifier_actionRec_deepstream_8.json`) works out of
the box using `process-on-frame=0` / `process-on-all-objects=1`. To rebuild the
library manually inside a running 8.0 container instead:

```
cd /workspace
docker/build_sequence_lib.sh
```

## Triton Inference Server pipeline

The pipeline can run its inference models on a remote/test NVIDIA Triton Inference
Server instead of local `nvinfer` elements. The remote backend is selected with
a single top-level switch in the application JSON:

```json
"inference_backend": "triton",              // or "nvinfer" (default)
"triton": { "server_url": "localhost:8001", "protocol_type": "grpc" }
```

When `"triton"` is set, `Utils/config.py` writes `nvinferserver` config files
instead of `nvinfer` ones, in the DeepStream 8.0 protobuf format
(`infer_config { ... }` + `input_control { ... }`), and `PipelineBuilder` uses
`Modules/triton_inference_engine_builder.py` which creates `nvinferserver`
GStreamer elements. The local `nvinfer` path is unchanged.

Shipped config: `configs/detector_tracker_classifier_lpr_triton_deepstream_8.json`
(detector -> tracker -> LPD -> LPR -> vehicle-make/type classifiers). It omits
the 3D action-recognition engine because Triton cannot serve TAO `.etlt` models
(those stay on `nvinfer` with the `_actionRec_` configs).

### How to run it

```bash
# 0. Build a Triton-flavored image (plain "devel" images have NO nvinferserver plugin)
./docker/build_images.sh 8.0            # 9.1 also works

# 1. Start the container (host networking so the pipeline and Triton share localhost:8001)
docker run --gpus all --net=host --privileged -it \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
  --mount type=bind,src="$(pwd)",target=/workspace \
  cpeeris/deepstreamsolutiondocker:8.0

# 2. Inside the container: generate the Triton model repository from the SAME JSON
cd /workspace
python3 Utils/generate_triton_model_repo.py \
  configs/detector_tracker_classifier_lpr_triton_deepstream_8.json \
  --repo /workspace/triton/model_repo \
  --backend onnxruntime --execution-accelerator tensorrt

#    The models are SYMLINKED into the repo (1/model.onnx -> original path), not
#    copied: the detector/make/type models live under the DeepStream samples and
#    LPD/LPR under /workspace/models, and Triton loads them through the symlinks.
#    Pass --copy to force real copies for filesystems that lack symlinks.
#
#    ONNX models are served by the onnxruntime backend using its TensorRT
#    execution provider (--execution-accelerator tensorrt, the default). This
#    routes Convs through TensorRT and avoids onnxruntime's cuDNN-frontend
#    heuristic planner, which fails ("CUDNN_FE failure 8: HEURISTIC_QUERY_FAILED")
#    on Ampere GPUs when plain CUDA EP is used. Pass --execution-accelerator cuda
#    or none to change that. Explicit input/output signatures are derived from the
#    ONNX graph when the `onnx` python package is installed (pip install onnx);
#    otherwise Triton derives them from the model at load time.

# 3. Start Triton Inference Server (gRPC on 8001) in the background
nohup tritonserver --model-repository=/workspace/triton/model_repo \
  --http-port 8000 --grpc-port 8001 > /tmp/triton.log 2>&1 &
#    wait until the log shows every model READY (TRT-EP engines are built on the
#    first load, which can take a minute or two).
#    stopping/restarting: pkill -x tritonserver   (NOT pkill -f, which matches its own
#    shell). Triton honors exit_timeout (~30s), so wait before checking:
#      pgrep -x tritonserver        # empty output = stopped
#      for p in 8000 8001 8002; do (echo > /dev/tcp/127.0.0.1/$p) 2>/dev/null \
#        && echo "$p busy" || echo "$p free"; done   # all "free" = ports released

# 4. Run the pipeline (client-side nvinferserver configs are generated into a
#    temp dir by Utils/config.py, exactly like the nvinfer ones)
python3 pipeline_launcher.py configs/detector_tracker_classifier_lpr_triton_deepstream_8.json
```

Verification: you should see "create triton inference" per engine, per-model
timing with real batch counts (not "no buffers observed"), vehicle make/type
labels (`['bmw', 'sedan']`, ...) on stdout and `output.mp4` written. Re-run any
`_deepstream_8.json` config afterwards to confirm the local `nvinfer` path is
unaffected.

### Backends and caveats

- Models are served with the `onnxruntime` backend by default (uses each
  engine's `onnx-file`). Newer Triton images call it `onnxruntime_onnx`; pass
  `--backend onnxruntime_onnx` there. To serve pre-built TensorRT engines instead:
  `--backend tensorrt_plan` (uses `model-engine-file`). `config.pbtxt` is written
  with explicit I/O signatures from the ONNX graph (field names are output above)
  and the accelerator requested via `optimization.execution_accelerators`.
- The DeepStream client parses outputs by the exact `output-blob-names` from the
  JSON, so the served model's graph outputs must match them. Triton's onnxruntime
  backend derives those names from the model graph (or from the explicit config).
- `protocol_type` values: `grpc` (port 8001, recommended) or `http` (port 8000).
- Client config generation (`Utils/config.py`) drops model-file keys (`onnx-file`,
  `model-engine-file`, `int8-calib-file`, `tlt-encoded-model`) because the server
  loads those from its own model repository. `gie-unique-id` maps to `unique_id`.
- Custom output parsers: if an engine sets `custom-lib-path` (and the file exists
  on the client), the config emits `custom_lib { path: ... }` plus
  `custom_parse_classifier_func`/`custom_parse_bbox_func` so the `nvinferserver`
  postprocessor uses the same custom parser the nvinfer path does. This is how
  LPR plates are decoded (see `notes/lpr_lpd_model_download.txt` STEP 3 to build
  `libnvdsinfer_custom_impl_lpr.so`, which is NOT bundled in the DS 8.0 image).
  Without it the LPR engine still runs but produces no plate text.
- The temp config files must survive for the whole pipeline run: `nvinferserver`
  reads `config-file-path` at pipeline start (unlike `nvinfer`, which loads at
  element creation). `PipelineBuilder` therefore cleans them up after the run.


