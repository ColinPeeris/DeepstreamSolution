"""Standalone single-file benchmark: the full DeepStream LPR pipeline, nvinfer vs triton.

Reproduces, as one self-contained file, the ``lpr`` family of
``benchmarks/run_benchmark.py`` / ``benchmarks/make_configs.py``: the complete
5-model License Plate Recognition pipeline plus the NvDCF tracker and OSD::

    streammux -> pgie_detector_1 (TrafficCamNet) -> nvtracker -> sgie_lpd_2
              -> sgie_lpr_3 -> sgie_classifier_5 (VehicleMakeNet)
              -> sgie_classifier_6 (VehicleTypeNet) -> nvvidconv -> nvosd
              -> tee -> queue -> fakesink

The point of this script is to check whether the crossover reported in
``benchmarks/experiment_report.txt`` reproduces -- nvinfer faster at 1 stream,
Triton faster at 16/32 streams -- now that the identical 5-model chain (not a
chain of identical detectors) is the workload.

Both backends run the identical five-stage chain and identical model files:

- pgie_detector_1    TrafficCamNet  FP16  batch 16  full-frame detector
- sgie_lpd_2         LPDNet         INT8  batch 16  plate detection (min 40x30)
- sgie_lpr_3         LPRNet         FP16  batch 16  plate OCR + custom NVPlate
                    parser (``NvDsInferParseCustomNVPlate``)
- sgie_classifier_5  VehicleMakeNet FP16  batch 16  vehicle make (min 64x64)
- sgie_classifier_6  VehicleTypeNet FP16  batch 16  vehicle type (min 64x64)

plus `config_tracker_NvDCF_perf.yml` and the OSD/fakesink branch, matching the
original benchmark methodology: streammux batch-size = min(streams, 16), model
batch-size 16 everywhere, per-image = per-batch / 16, fps counted from actual
decoded frames. Engine build time is never measured: every run deserializes the
prebuilt ``_b16_*`` engine files and, with ``--warmup``, any one-off build is
absorbed before the measured repeats.

Fully self-contained: no dependency on this repository or ``benchmarks/`` (the
lprnet integration even re-derives the Triton model-repo pbtxt including its
INT32 ``tf_op_layer_ArgMax`` output). Copy this one file onto any machine with
the DeepStream container (python3 + pygobject + gst-python, DeepStream plugins
on ``GST_PLUGIN_PATH``, Triton binary, plus the sample models and the LP
weight/engine files) and run::

    python3 lpr_benchmark.py [--streams 1 16 32] [--repeats 3] [--warmup]

Artifacts go under OUT (default ``./lpr_benchmark_out/``):

- ``configs/<stage>_{nvinfer,nvinferserver}.txt``   per-stage engine configs
- ``triton/model_repo/...``                         generated Triton model repo
- ``logs/``                                         per-run logs (warmup/reps,
                                                    triton_server.log)
- ``results.csv``                                   one row per measured run
- ``summary.txt``                                   mean fps + per-model per_image

Run isolation / exit codes:

- parent: builds configs + model repo, starts/stops Triton, times children
- child (invoked with ``--child RUN.json``): builds the GStreamer pipeline,
  prints ``End-of-stream`` (+ frame count + per-stage timing), exits 0 on
  success, 2 on a GStreamer bus ERROR
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

DS_SAMPLES = '/opt/nvidia/deepstream/deepstream/samples'
DS_LIB = '/opt/nvidia/deepstream/deepstream/lib'
SAMPLES_MODELS = os.path.join(DS_SAMPLES, 'models')
DEFAULT_VIDEO = os.path.join(DS_SAMPLES, 'streams', 'sample_720p.h264')
DEFAULT_LP_DIR = '/workspace/models/LP'
CUSTOM_LPR_LIB = os.path.join(DS_LIB, 'libnvdsinfer_custom_impl_lpr.so')
TRACKER_LIB = os.path.join(DS_LIB, 'libnvds_nvmultiobjecttracker.so')
TRACKER_CFG = os.path.join(REPO_ROOT, 'config_tracker_NvDCF_perf.yml')
DEFAULT_TRITON_BIN = '/opt/tritonserver/bin/tritonserver'

BACKENDS = ('nvinfer', 'triton')
STREAM_SIZES = (1, 16, 32)
MODEL_BATCH = 16
STAGE_KINDS = ('pgie', 'lpd', 'lpr', 'make', 'type')

# Per-stage model files (defaults in the DeepStream sample install / repo LP
# dir; override --samples-models / --lp-dir when copying the script elsewhere).
MODEL_FILES = {
    'resnet18_trafficcamnet': {
        'onnx': os.path.join(SAMPLES_MODELS, 'Primary_Detector',
                             'resnet18_trafficcamnet_pruned.onnx'),
        'engine': os.path.join(SAMPLES_MODELS, 'Primary_Detector',
                               'resnet18_trafficcamnet_pruned.onnx_b16_gpu0_fp16.engine'),
        'labels': os.path.join(SAMPLES_MODELS, 'Primary_Detector', 'labels.txt'),
    },
    'lpdnet': {
        'dir': 'LPD', 'file': 'LPDNet_usa_pruned_tao5.onnx',
        'engine_file': 'LPDNet_usa_pruned_tao5.onnx_b16_gpu0_int8.engine',
        'calib': 'usa_cal_10.1.0.bin', 'labels': 'usa_lpd_label.txt',
    },
    'lprnet': {
        'dir': 'LPR', 'file': 'us_lprnet_baseline18_deployable.onnx',
        'engine_file': 'us_lprnet_baseline18_deployable.onnx_b16_gpu0_fp16.engine',
        'labels': 'dict_us.txt',
    },
    'vehiclemakenet': {
        'onnx': os.path.join(SAMPLES_MODELS, 'Secondary_VehicleMake',
                             'resnet18_vehiclemakenet_pruned.onnx'),
        'engine': os.path.join(SAMPLES_MODELS, 'Secondary_VehicleMake',
                               'resnet18_vehiclemakenet_pruned.onnx_b16_gpu0_fp16.engine'),
        'labels': os.path.join(SAMPLES_MODELS, 'Secondary_VehicleMake', 'labels.txt'),
    },
    'vehicletypenet': {
        'onnx': os.path.join(SAMPLES_MODELS, 'Secondary_VehicleTypes',
                             'resnet18_vehicletypenet_pruned.onnx'),
        'engine': os.path.join(SAMPLES_MODELS, 'Secondary_VehicleTypes',
                               'resnet18_vehicletypenet_pruned.onnx_b16_gpu0_fp16.engine'),
        'labels': os.path.join(SAMPLES_MODELS, 'Secondary_VehicleTypes', 'labels.txt'),
    },
}

# Triton model-repo tensor specs mirroring triton/model_repo/*/config.pbtxt.
# lprnet's input is named "image_input" and its ArgMax output is TYPE_INT32.
TRITON_MODEL_SPECS = {
    'resnet18_trafficcamnet': {
        'input': ('input_1:0', '3, 544, 960', 'TYPE_FP32'),
        'outputs': [('output_cov/Sigmoid:0', '4, 34, 60', 'TYPE_FP32'),
                    ('output_bbox/BiasAdd:0', '16, 34, 60', 'TYPE_FP32')],
    },
    'lpdnet': {
        'input': ('input_1:0', '3, 480, 640', 'TYPE_FP32'),
        'outputs': [('output_cov/Sigmoid:0', '1, 30, 40', 'TYPE_FP32'),
                    ('output_bbox/BiasAdd:0', '4, 30, 40', 'TYPE_FP32')],
    },
    'lprnet': {
        'input': ('image_input', '3, 48, 96', 'TYPE_FP32'),
        'outputs': [('tf_op_layer_ArgMax', '24', 'TYPE_INT32'),
                    ('tf_op_layer_Max', '24', 'TYPE_FP32')],
    },
    'vehiclemakenet': {
        'input': ('input_1:0', '3, 224, 224', 'TYPE_FP32'),
        'outputs': [('predictions/Softmax:0', '20', 'TYPE_FP32')],
    },
    'vehicletypenet': {
        'input': ('input_1:0', '3, 224, 224', 'TYPE_FP32'),
        'outputs': [('predictions/Softmax:0', '6', 'TYPE_FP32')],
    },
}


def model_paths(lp_dir: str, samples_models: str,
                lpd_precision: str = 'int8') -> Dict[str, Dict[str, str]]:
    """Resolve per-model onnx/engine/labels paths from the overridable roots.

    ``lpd_precision`` selects the prebuilt LPD engine (INT8 or FP16); it also
    tags the config with the matching ``network-mode`` (1 = INT8, 2 = FP16).
    """
    base = {name: dict(mod) for name, mod in MODEL_FILES.items()}
    base['resnet18_trafficcamnet'].update(
        onnx=os.path.join(samples_models, 'Primary_Detector',
                          'resnet18_trafficcamnet_pruned.onnx'),
        engine=os.path.join(samples_models, 'Primary_Detector',
                            'resnet18_trafficcamnet_pruned.onnx_b16_gpu0_fp16.engine'))
    base['vehiclemakenet'].update(
        onnx=os.path.join(samples_models, 'Secondary_VehicleMake',
                          'resnet18_vehiclemakenet_pruned.onnx'),
        engine=os.path.join(samples_models, 'Secondary_VehicleMake',
                            'resnet18_vehiclemakenet_pruned.onnx_b16_gpu0_fp16.engine'))
    base['vehicletypenet'].update(
        onnx=os.path.join(samples_models, 'Secondary_VehicleTypes',
                          'resnet18_vehicletypenet_pruned.onnx'),
        engine=os.path.join(samples_models, 'Secondary_VehicleTypes',
                            'resnet18_vehicletypenet_pruned.onnx_b16_gpu0_fp16.engine'))
    if lpd_precision == 'fp16':
        base['lpdnet']['engine_file'] = base['lpdnet']['engine_file'].replace(
            '_int8.', '_fp16.')
        base['lpdnet']['network_mode'] = 2
    else:
        base['lpdnet']['network_mode'] = 1
    for name in ('lpdnet', 'lprnet'):
        d = os.path.join(lp_dir, base[name]['dir'])
        base[name]['onnx'] = os.path.join(d, base[name]['file'])
        base[name]['engine'] = os.path.join(d, base[name]['engine_file'])
        base[name]['labels'] = os.path.join(d, base[name]['labels'])
    return base


def stage_properties(paths: Dict[str, Dict[str, str]],
                     lp_dir: str) -> List[Dict[str, Any]]:
    """The five per-stage property blocks (identical to the repo LPR configs)."""
    return [
        {
            'name': 'pgie_detector_1', 'model': 'resnet18_trafficcamnet',
            'kind': 'pgie', 'uid': 1, 'process-mode': 1,
            'prop': {
                'gpu-id': 0,
                'net-scale-factor': 0.0039215697906911373,
                'onnx-file': paths['resnet18_trafficcamnet']['onnx'],
                'model-engine-file': paths['resnet18_trafficcamnet']['engine'],
                'labelfile-path': paths['resnet18_trafficcamnet']['labels'],
                'batch-size': MODEL_BATCH,
                'network-mode': 2,
                'process-mode': 1,
                'model-color-format': 0,
                'num-detected-classes': 4,
                'interval': 0,
                'gie-unique-id': 1,
                'output-blob-names': 'output_bbox/BiasAdd:0;output_cov/Sigmoid:0',
            },
            'class_attrs': {'pre-cluster-threshold': 0.2, 'eps': 0.2,
                            'group-threshold': 1},
        },
        {
            'name': 'sgie_lpd_2', 'model': 'lpdnet', 'kind': 'lpd',
            'uid': 2, 'process-mode': 2,
            'prop': {
                'gpu-id': 0,
                'net-scale-factor': 0.0039215697906911373,
                'model-color-format': 0,
                'onnx-file': paths['lpdnet']['onnx'],
                'model-engine-file': paths['lpdnet']['engine'],
                'labelfile-path': paths['lpdnet']['labels'],
                'batch-size': MODEL_BATCH,
                'network-mode': paths['lpdnet'].get('network_mode', 1),
                'num-detected-classes': 1,
                'process-mode': 2,
                'interval': 0,
                'gie-unique-id': 2,
                'network-type': 0,
                'operate-on-gie-id': 1,
                'operate-on-class-ids': 0,
                'cluster-mode': 3,
                'output-blob-names': 'output_cov/Sigmoid:0;output_bbox/BiasAdd:0',
                'input-object-min-height': 30,
                'input-object-min-width': 40,
            },
            'class_attrs': {'pre-cluster-threshold': 0.3, 'roi-top-offset': 0,
                            'roi-bottom-offset': 0, 'detected-min-w': 0,
                            'detected-min-h': 0, 'detected-max-w': 0,
                            'detected-max-h': 0},
        },
        {
            'name': 'sgie_lpr_3', 'model': 'lprnet', 'kind': 'lpr',
            'uid': 3, 'process-mode': 2,
            'prop': {
                'gpu-id': 0,
                'net-scale-factor': 0.00392156862745098,
                'model-color-format': 0,
                'onnx-file': paths['lprnet']['onnx'],
                'model-engine-file': paths['lprnet']['engine'],
                'labelfile-path': paths['lprnet']['labels'],
                'batch-size': MODEL_BATCH,
                'network-mode': 2,
                'num-detected-classes': 3,
                'gie-unique-id': 3,
                'network-type': 1,
                'process-mode': 2,
                'operate-on-gie-id': 2,
                'output-blob-names': 'tf_op_layer_ArgMax;tf_op_layer_Max',
                'parse-classifier-func-name': 'NvDsInferParseCustomNVPlate',
                'custom-lib-path': CUSTOM_LPR_LIB,
            },
            'class_attrs': {'threshold': 0.5},
        },
        {
            'name': 'sgie_classifier_5', 'model': 'vehiclemakenet',
            'kind': 'make', 'uid': 5, 'process-mode': 2,
            'prop': {
                'gpu-id': 0,
                'net-scale-factor': 1,
                'onnx-file': paths['vehiclemakenet']['onnx'],
                'model-engine-file': paths['vehiclemakenet']['engine'],
                'labelfile-path': paths['vehiclemakenet']['labels'],
                'batch-size': MODEL_BATCH,
                'network-mode': 2,
                'input-object-min-width': 64,
                'input-object-min-height': 64,
                'process-mode': 2,
                'model-color-format': 1,
                'gie-unique-id': 5,
                'operate-on-gie-id': 1,
                'operate-on-class-ids': 0,
                'is-classifier': 1,
                'output-blob-names': 'predictions/Softmax:0',
                'classifier-async-mode': 1,
                'classifier-threshold': 0.51,
            },
            'class_attrs': {},
        },
        {
            'name': 'sgie_classifier_6', 'model': 'vehicletypenet',
            'kind': 'type', 'uid': 6, 'process-mode': 2,
            'prop': {
                'gpu-id': 0,
                'net-scale-factor': 1,
                'onnx-file': paths['vehicletypenet']['onnx'],
                'model-engine-file': paths['vehicletypenet']['engine'],
                'labelfile-path': paths['vehicletypenet']['labels'],
                'batch-size': MODEL_BATCH,
                'network-mode': 2,
                'input-object-min-width': 64,
                'input-object-min-height': 64,
                'process-mode': 2,
                'model-color-format': 1,
                'gie-unique-id': 6,
                'operate-on-gie-id': 1,
                'operate-on-class-ids': 0,
                'is-classifier': 1,
                'output-blob-names': 'predictions/Softmax:0',
                'classifier-async-mode': 1,
                'classifier-threshold': 0.51,
            },
            'class_attrs': {},
        },
    ]
    for st in stages:
        if st['model'] == 'lpdnet' and \
           int(st['prop'].get('network-mode', 1)) == 1:
            st['prop']['int8-calib-file'] = os.path.join(
                lp_dir, 'LPD', 'usa_cal_10.1.0.bin')
    return stages


def _property_lines(prop: Dict[str, Any]) -> List[str]:
    return ['%s=%s' % (k, prop[k]) for k in prop]


def write_nvinfer_config(path: str, prop: Dict[str, Any],
                         class_attrs: Optional[Dict[str, Any]] = None) -> None:
    with open(path, 'w') as f:
        f.write('[property] \n')
        f.write('\n'.join(_property_lines(prop)) + '\n')
        f.write('\n')
        if class_attrs:
            f.write('[class-attrs-all] \n')
            f.write('\n'.join(_property_lines(class_attrs)) + '\n')


def _triton_postprocess_lines(prop: Dict[str, Any],
                              class_attrs: Dict[str, Any]) -> List[str]:
    """Render the DS8 ``postprocess`` block (detection or classification)."""
    lines: List[str] = []
    is_classifier = (int(prop.get('is-classifier', 0)) == 1
                     or int(prop.get('network-type', 0)) == 1)
    custom_lib = prop.get('custom-lib-path')
    lib_ok = bool(custom_lib and os.path.isfile(custom_lib))
    if is_classifier:
        threshold = prop.get('classifier-threshold')
        if threshold is None and class_attrs:
            threshold = class_attrs.get('threshold')
        if threshold is not None:
            lines.append('    classification {')
            lines.append('      threshold: %s' % threshold)
            custom_func = prop.get('parse-classifier-func-name')
            if custom_func and lib_ok:
                lines.append('      custom_parse_classifier_func: "%s"' % custom_func)
            lines.append('    }')
    elif 'num-detected-classes' in prop:
        lines.append('    detection {')
        lines.append('      num_detected_classes: %s' % prop['num-detected-classes'])
        pre_threshold = (class_attrs or {}).get('pre-cluster-threshold')
        if pre_threshold is not None:
            lines.append('      per_class_params {')
            lines.append('        key: 0')
            lines.append('        value { pre_threshold: %s }' % pre_threshold)
            lines.append('      }')
            lines.append('      nms {')
            lines.append('        confidence_threshold: %s' % pre_threshold)
            lines.append('        topk: 20')
            lines.append('        iou_threshold: 0.5')
            lines.append('      }')
        custom_func = prop.get('parse-bbox-func-name')
        if custom_func and lib_ok:
            lines.append('      custom_parse_bbox_func: "%s"' % custom_func)
        lines.append('    }')
    return lines


def write_nvinferserver_config(path: str, prop: Dict[str, Any],
                               class_attrs: Optional[Dict[str, Any]],
                               model_name: str, server_url: str) -> None:
    """Render a gst-nvinferserver config (DeepStream-8 protobuf format).

    Covers full-frame detectors, ROI detectors (LPD with cluster-mode/batch),
    ROI classifiers (vehiclemake/vehicletypes) and the LPRNet classifier with a
    custom parse function read through a ``custom_lib`` block -- mirroring
    ``Utils/config.py`` so both backends share one property block.
    """
    out_names = [n.strip() for n in str(prop['output-blob-names']).split(';') if n.strip()]
    process_mode: int = int(prop.get('process-mode', 1))
    color_format = int(prop.get('model-color-format', 0))
    lines: List[str] = ['infer_config {']
    lines.append('  unique_id: %s' % prop.get('gie-unique-id', 1))
    if 'gpu-id' in prop:
        lines.append('  gpu_ids: [%s]' % prop['gpu-id'])
    lines.append('  max_batch_size: %s' % prop.get('batch-size', 1))
    lines.append('  backend {')
    lines.append('    outputs: [')
    for i, blob_name in enumerate(out_names):
        comma = ',' if i < len(out_names) - 1 else ''
        lines.append('      {name: "%s"}%s' % (blob_name, comma))
    lines.append('    ]')
    lines.append('    triton {')
    lines.append('      model_name: "%s"' % model_name)
    lines.append('      version: -1')
    lines.append('      grpc {')
    lines.append('        url: "%s"' % server_url)
    lines.append('        enable_cuda_buffer_sharing: true')
    lines.append('      }')
    lines.append('    }')
    lines.append('    output_mem_type: MEMORY_TYPE_CPU')
    lines.append('  }')
    custom_lib = prop.get('custom-lib-path')
    if custom_lib and os.path.isfile(custom_lib):
        lines.append('  custom_lib {')
        lines.append('    path: "%s"' % custom_lib)
        lines.append('  }')
    lines.append('  preprocess {')
    lines.append('    network_format: %s'
                 % ('IMAGE_FORMAT_BGR' if color_format == 1 else 'IMAGE_FORMAT_RGB'))
    lines.append('    tensor_order: TENSOR_ORDER_LINEAR')
    lines.append('    maintain_aspect_ratio: 0')
    lines.append('    frame_scaling_hw: FRAME_SCALING_HW_DEFAULT')
    lines.append('    frame_scaling_filter: 1')
    if 'net-scale-factor' in prop:
        lines.append('    normalize {')
        lines.append('      scale_factor: %s' % prop['net-scale-factor'])
        lines.append('    }')
    lines.append('  }')
    lines.append('  postprocess {')
    if 'labelfile-path' in prop:
        lines.append('    labelfile_path: "%s"' % prop['labelfile-path'])
    lines.extend(_triton_postprocess_lines(prop, class_attrs or {}))
    lines.append('  }')
    lines.append('}')
    lines.append('input_control {')
    lines.append('  process_mode: %s'
                 % ('PROCESS_MODE_FULL_FRAME' if process_mode == 1
                    else 'PROCESS_MODE_CLIP_OBJECTS'))
    lines.append('  operate_on_gie_id: %s' % prop.get('operate-on-gie-id', -1))
    if 'operate-on-class-ids' in prop:
        class_ids = [x.strip() for x in str(prop['operate-on-class-ids']).split(',')
                     if x.strip()]
        lines.append('  operate_on_class_ids: [%s]' % ', '.join(class_ids))
    if 'interval' in prop and process_mode == 1:
        lines.append('  interval: %s' % prop['interval'])
    if 'classifier-async-mode' in prop:
        lines.append('  async_mode: %s'
                     % ('true' if int(prop['classifier-async-mode']) == 1 else 'false'))
    if process_mode == 2 and ('input-object-min-width' in prop
                              or 'input-object-min-height' in prop):
        lines.append('  object_control {')
        lines.append('    bbox_filter {')
        if 'input-object-min-width' in prop:
            lines.append('      min_width: %s' % prop['input-object-min-width'])
        if 'input-object-min-height' in prop:
            lines.append('      min_height: %s' % prop['input-object-min-height'])
        lines.append('    }')
        lines.append('  }')
    lines.append('}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def model_repo_config_pbtxt(spec: Dict[str, Any], max_batch_size: int) -> str:
    input_name, input_dims, input_dtype = spec['input']
    lines = ['backend: "onnxruntime"',
             'max_batch_size: %d' % max_batch_size,
             'input {',
             '    name: "%s"' % input_name,
             '    data_type: %s' % input_dtype,
             '    dims: [%s]' % input_dims,
             '  }']
    for out_name, out_dims, out_dtype in spec['outputs']:
        lines.append('output {')
        lines.append('    name: "%s"' % out_name)
        lines.append('    data_type: %s' % out_dtype)
        lines.append('    dims: [%s]' % out_dims)
        lines.append('  }')
    lines.append('optimization {')
    lines.append('  execution_accelerators {')
    lines.append('    gpu_execution_accelerator: [')
    lines.append('      {')
    lines.append('        name: "tensorrt"')
    lines.append('        parameters { key: "precision_mode" value: "FP16" }')
    lines.append('      }')
    lines.append('    ]')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def ensure_model_repo(repo_dir: str, model_onnx: Dict[str, str],
                      max_batch_size: int, copy_models: bool) -> None:
    """Build the Triton repo ``config.pbtxt`` + model.onnx per served model."""
    for model_name, onnx_path in model_onnx.items():
        model_dir = os.path.join(repo_dir, model_name)
        version_dir = os.path.join(model_dir, '1')
        if os.path.isfile(os.path.join(model_dir, 'config.pbtxt')):
            continue
        os.makedirs(version_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'config.pbtxt'), 'w') as f:
            f.write(model_repo_config_pbtxt(TRITON_MODEL_SPECS[model_name],
                                            max_batch_size))
        target = os.path.join(version_dir, 'model.onnx')
        if os.path.isfile(target):
            continue
        if copy_models:
            shutil.copyfile(onnx_path, target)
        else:
            os.symlink(onnx_path, target)


def triton_running() -> bool:
    return subprocess.run(['pgrep', '-x', 'tritonserver'],
                          capture_output=True).returncode == 0


def stop_triton(timeout: int = 120) -> None:
    if not triton_running():
        return
    print('Stopping Triton server ...')
    subprocess.run(['pkill', '-x', 'tritonserver'], check=True)
    deadline = time.time() + timeout
    while triton_running() and time.time() < deadline:
        time.sleep(2)
    if triton_running():
        raise RuntimeError('Triton server did not stop within %ds' % timeout)
    print('Triton server stopped.')


def start_triton(repo_dir: str, log_path: str, triton_bin: str,
                 timeout: int = 600) -> None:
    if triton_running():
        print('Triton server already running.')
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print('Starting Triton server (repo=%s) ...' % repo_dir)
    log = open(log_path, 'w')
    subprocess.Popen(
        [triton_bin, '--model-repository', repo_dir,
         '--grpc-port', '8001', '--http-port', '8000'],
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    deadline = time.time() + timeout
    ready = False
    while time.time() < deadline:
        if triton_running():
            try:
                with open(log_path) as f:
                    if 'Started GRPCInferenceService' in f.read():
                        ready = True
                        break
            except OSError:
                pass
        time.sleep(2)
    if not ready:
        raise RuntimeError('Triton server did not become ready within %ds '
                           '(see %s)' % (timeout, log_path))
    print('Triton server ready.')


def _gst_symbols():
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import GLib, Gst
    return GLib, Gst


def _child_pipeline(run: Dict[str, Any]) -> None:
    """Build and run the full LPR pipeline (child process)."""
    GLib, Gst = _gst_symbols()
    try:
        import pyds
        unified_mem = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
    except Exception:
        unified_mem = 3  # NVBUF_MEM_CUDA_UNIFIED

    def _time_batches(pad, info, data):
        now = time.monotonic()
        if data['last'] is not None:
            data['total_ms'] += (now - data['last']) * 1000.0
            data['n'] += 1
        data['last'] = now
        return Gst.PadProbeReturn.OK

    backend = run['backend']
    streams = run['streams']
    mux_batch = run['mux_batch']
    batch = run['batch']

    Gst.init(None)
    pipeline = Gst.Pipeline()

    streammux = Gst.ElementFactory.make('nvstreammux', 'streammux')
    if not streammux:
        raise RuntimeError('Unable to create nvstreammux')
    pipeline.add(streammux)
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', mux_batch)
    streammux.set_property('batched-push-timeout', 4000000)

    frame_count = {'val': 0}

    def _count(pad, info, data):
        data['val'] += 1
        return Gst.PadProbeReturn.OK

    for i in range(streams):
        source = Gst.ElementFactory.make('filesrc', 'src-%d' % i)
        parser = Gst.ElementFactory.make('h264parse', 'parse-%d' % i)
        decoder = Gst.ElementFactory.make('nvv4l2decoder', 'dec-%d' % i)
        if not (source and parser and decoder):
            raise RuntimeError('Unable to create source/parser/decoder %d' % i)
        for el in (source, parser, decoder):
            pipeline.add(el)
        source.set_property('location', run['video'])
        source.link(parser)
        parser.link(decoder)
        decoder.get_static_pad('src').link(streammux.get_request_pad('sink_%d' % i))
        decoder.get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER, _count,
                                                frame_count)

    factory = 'nvinferserver' if backend == 'triton' else 'nvinfer'
    gie_els = []
    prev = streammux
    gies = run['gies']
    for spec in gies[0:1]:
        gie = Gst.ElementFactory.make(factory, spec['name'])
        if not gie:
            raise RuntimeError('Unable to create %s element "%s"'
                               % (backend, spec['name']))
        gie.set_property('config-file-path', spec['config'])
        pipeline.add(gie)
        if not prev.link(gie):
            raise RuntimeError('Failed to link %s -> %s'
                               % (prev.get_name(), spec['name']))
        prev = gie
        gie_els.append(gie)

    tracker = Gst.ElementFactory.make('nvtracker', 'tracker')
    if not tracker:
        raise RuntimeError('Unable to create nvtracker')
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    tracker.set_property('gpu_id', 0)
    tracker.set_property('ll-lib-file', run['tracker']['lib'])
    tracker.set_property('ll-config-file', run['tracker']['cfg'])
    pipeline.add(tracker)
    if not prev.link(tracker):
        raise RuntimeError('Failed to link %s -> tracker' % prev.get_name())
    prev = tracker

    for spec in gies[1:]:
        gie = Gst.ElementFactory.make(factory, spec['name'])
        if not gie:
            raise RuntimeError('Unable to create %s element "%s"'
                               % (backend, spec['name']))
        gie.set_property('config-file-path', spec['config'])
        pipeline.add(gie)
        if not prev.link(gie):
            raise RuntimeError('Failed to link %s -> %s'
                               % (prev.get_name(), spec['name']))
        prev = gie
        gie_els.append(gie)

    nvvidconv = Gst.ElementFactory.make('nvvideoconvert', 'convertor')
    capsfilter = Gst.ElementFactory.make('capsfilter', 'converter_caps')
    nvosd = Gst.ElementFactory.make('nvdsosd', 'onscreendisplay')
    tee = Gst.ElementFactory.make('tee', 'nvsink-tee')
    queue = Gst.ElementFactory.make('queue', 'queue_fake_sink')
    sink = Gst.ElementFactory.make('fakesink', 'fakesink')
    for el, name in ((nvvidconv, 'nvvideoconvert'), (capsfilter, 'capsfilter'),
                     (nvosd, 'nvdsosd'), (tee, 'tee'), (queue, 'queue'),
                     (sink, 'fakesink')):
        if not el:
            raise RuntimeError('Unable to create %s' % name)
    nvvidconv.set_property('nvbuf-memory-type', unified_mem)
    caps = Gst.Caps.from_string('video/x-raw(memory:NVMM), format=RGBA')
    capsfilter.set_property('caps', caps)

    for el in (nvvidconv, capsfilter, nvosd, tee, queue, sink):
        pipeline.add(el)
    if not (prev.link(nvvidconv) and nvvidconv.link(capsfilter)
            and capsfilter.link(nvosd) and nvosd.link(tee)):
        raise RuntimeError('Failed to link sink branch')
    tee_pad = tee.get_request_pad('src_%u')
    tee_link = tee_pad.link(queue.get_static_pad('sink'))
    if tee_link != Gst.PadLinkReturn.OK:
        raise RuntimeError('Failed to link tee -> queue (%s)' % tee_link)
    if not queue.link(sink):
        raise RuntimeError('Failed to link queue -> fakesink')

    timings: Dict[str, Dict[str, Any]] = {}
    for spec in gies:
        el = pipeline.get_by_name(spec['name'])
        data = {'kind': spec['kind'], 'last': None, 'total_ms': 0.0, 'n': 0}
        timings[spec['name']] = data
        el.get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER,
                                            _time_batches, data)

    state = {'error': None}
    loop = GLib.MainLoop()

    def _bus_call(bus, message, user_data):
        if message.type == Gst.MessageType.EOS:
            print('End-of-stream', flush=True)
            loop.quit()
            return True
        if message.type == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            state['error'] = str(err.message)
            sys.stderr.write('Error: %s\n' % state['error'])
            if dbg:
                sys.stderr.write('%s\n' % dbg)
            loop.quit()
            return True
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', _bus_call, loop)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    pipeline.set_state(Gst.State.NULL)

    if state['error'] is not None:
        print('ERROR', flush=True)
        sys.exit(2)

    print('PROCESSED_FRAMES=%d' % frame_count['val'], flush=True)
    for name, data in timings.items():
        if data['n'] > 0:
            per_batch = data['total_ms'] / data['n']
            per_image = per_batch / batch
            print('%s per_batch=%.2f ms per_image=%.2f ms (n=%d)'
                  % (name, per_batch, per_image, data['n']), flush=True)


def run_child(run_json: str) -> int:
    with open(run_json) as f:
        run = json.load(f)
    _child_pipeline(run)
    return 0


def run_one(script: str, run_json: str, log_path: str, cwd: str,
            timeout: int = 1800
            ) -> Tuple[float, int, Optional[int], Dict[str, float]]:
    t0 = time.monotonic()
    with open(log_path, 'w') as log:
        ret = subprocess.run(
            [sys.executable, '-u', script, '--child', run_json],
            stdout=log, stderr=subprocess.STDOUT, cwd=cwd, timeout=timeout)
    wall_s = time.monotonic() - t0
    with open(log_path) as f:
        content = f.read()
    if 'End-of-stream' not in content:
        raise RuntimeError('run never reached End-of-stream (see %s)' % log_path)
    frames = None
    m = re.search(r'PROCESSED_FRAMES=(\d+)', content)
    if m:
        frames = int(m.group(1))
    per_image: Dict[str, float] = {}
    for line in content.splitlines():
        m = re.match(r'(\S+) per_batch=[\d.]+ ms per_image=([\d.]+) ms \(n=\d+\)',
                     line.strip())
        if m:
            per_image[m.group(1)] = float(m.group(2))
    return wall_s, ret.returncode, frames, per_image


def select_stage_kinds(requested: str) -> List[str]:
    """Validate ``--stages`` and return the canonical ordered subset."""
    kinds = [k.strip() for k in requested.split(',') if k.strip()]
    if not kinds:
        raise ValueError('--stages is empty')
    unknown = [k for k in kinds if k not in STAGE_KINDS]
    if unknown:
        raise ValueError('unknown stage(s) in --stages: %s' % ', '.join(unknown))
    if 'pgie' not in kinds:
        raise ValueError('--stages must include pgie (every secondary stage '
                         'operates on its detections)')
    if 'lpr' in kinds and 'lpd' not in kinds:
        raise ValueError('--stages with lpr must include lpd (lpr operates on '
                         'lpd outputs)')
    return [k for k in STAGE_KINDS if k in kinds]


def bench(args: argparse.Namespace, out: str) -> List[Dict[str, Any]]:
    configs_dir = os.path.join(out, 'configs')
    logs_dir = os.path.join(out, 'logs')
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    paths = model_paths(args.lp_dir, args.samples_models, args.lpd_precision)
    stages = [s for s in stage_properties(paths, args.lp_dir)
              if s['kind'] in args.stage_kinds]
    print('  stages   : %s' % ' -> '.join(s['name'] for s in stages))

    # The custom NVPlate parser reads the plate dictionary from a bare
    # ``dict.txt`` in the working directory of the pipeline process. Mirror the
    # original benchmark (children ran with cwd=/workspace) by staging the LPR
    # dict into OUT and running every child from there.
    if 'lpr' in args.stage_kinds:
        dict_src = os.path.join(args.lp_dir, 'LPR', 'dict_us.txt')
        dict_dst = os.path.join(out, 'dict.txt')
        if os.path.isfile(dict_src):
            if not os.path.isfile(dict_dst) or \
               open(dict_src, 'rb').read() != open(dict_dst, 'rb').read():
                shutil.copyfile(dict_src, dict_dst)
        elif not os.path.isfile(dict_dst):
            raise RuntimeError('no LPR dictionary found for the NVPlate parser '
                               '(need dict_us.txt in %s or dict.txt in %s)'
                               % (args.lp_dir, out))

    repo_dir = os.path.join(out, 'triton', 'model_repo')
    model_onnx = {s['model']: paths[s['model']]['onnx'] for s in stages}
    ensure_model_repo(repo_dir, model_onnx, MODEL_BATCH, args.copy_models)
    print('  triton repo: %s (batch %d)' % (repo_dir, MODEL_BATCH))

    stage_configs: Dict[str, Dict[str, str]] = {}
    for st in stages:
        nvinfer_cfg = os.path.join(configs_dir, '%s_nvinfer.txt' % st['name'])
        triton_cfg = os.path.join(configs_dir, '%s_nvinferserver.txt' % st['name'])
        write_nvinfer_config(nvinfer_cfg, st['prop'], st['class_attrs'])
        write_nvinferserver_config(triton_cfg, st['prop'], st['class_attrs'],
                                   model_name=st['model'],
                                   server_url='localhost:8001')
        stage_configs[st['name']] = {'nvinfer': nvinfer_cfg, 'triton': triton_cfg}

    script = os.path.abspath(__file__)
    rows: List[Dict[str, Any]] = []
    for backend in BACKENDS:
        if args.manage_triton:
            if backend == 'triton':
                start_triton(repo_dir,
                             os.path.join(logs_dir, 'triton_server.log'),
                             args.triton_bin)
            else:
                stop_triton()
        for streams in args.streams:
            mux_batch = min(streams, MODEL_BATCH)
            gies = [{'name': st['name'], 'kind': st['kind'],
                     'config': stage_configs[st['name']][backend]}
                    for st in stages]
            run = {'backend': backend, 'streams': streams,
                   'batch': MODEL_BATCH, 'mux_batch': mux_batch,
                   'video': args.video, 'gies': gies,
                   'tracker': {'lib': TRACKER_LIB, 'cfg': TRACKER_CFG}}
            run_json = os.path.join(out, 'run_%s_%d.json' % (backend, streams))
            with open(run_json, 'w') as f:
                json.dump(run, f)
            print('--- %s | %d streams (mux batch %d, model batch %d) ---'
                  % (backend, streams, mux_batch, MODEL_BATCH))
            if args.warmup:
                run_one(script, run_json,
                        os.path.join(logs_dir, '%s_%d_warmup.log'
                                     % (backend, streams)),
                        out)
                print('  warmup done')
            for rep in range(1, args.repeats + 1):
                log_path = os.path.join(logs_dir, '%s_%d_rep%d.log'
                                        % (backend, streams, rep))
                wall_s, rc, frames, per_image = run_one(script, run_json,
                                                        log_path, out)
                if rc != 0:
                    raise RuntimeError('run exited rc=%d (see %s)' % (rc, log_path))
                fps = frames / wall_s if frames else float('nan')
                row: Dict[str, Any] = {
                    'backend': backend, 'streams': streams, 'repeat': rep,
                    'wall_s': wall_s, 'fps': fps, 'frames': frames,
                }
                for name, val in per_image.items():
                    row['per_image_%s' % name] = val
                rows.append(row)
                print('  rep %d: wall=%.1fs fps=%.1f' % (rep, wall_s, fps))
    if args.manage_triton:
        stop_triton()
    return rows


def format_table(rows: List[Dict[str, Any]]) -> str:
    lines = ['%-8s %5s %10s %9s %4s' %
             ('backend', 'strms', 'wall_s', 'fps', 'rep')]
    for r in rows:
        lines.append('%-8s %5d %10.1f %9.1f %4d' % (
            r['backend'], r['streams'], r['wall_s_mean'],
            r['fps_mean'], r['repeats']))
    return '\n'.join(lines)


def write_results(out: str, rows: List[Dict[str, Any]],
                  stream_order: List[int]) -> None:
    aggregate: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        by_key.setdefault((r['backend'], r['streams']), []).append(r)
    for backend in BACKENDS:
        for streams in stream_order:
            group = by_key.get((backend, streams), [])
            if not group:
                continue
            mean = lambda k: sum(r[k] for r in group) / len(group)  # noqa: E731
            row: Dict[str, Any] = {
                'backend': backend, 'streams': streams, 'repeats': len(group),
                'wall_s_mean': mean('wall_s'), 'fps_mean': mean('fps'),
            }
            for name in [k for k in group[0] if k.startswith('per_image_')]:
                row[name] = mean(name)
            aggregate.append(row)
    csv_path = os.path.join(out, 'results.csv')
    fieldnames = ['backend', 'streams', 'repeat', 'wall_s', 'fps', 'frames']
    per_keys = sorted({k for r in rows for k in r if k.startswith('per_image_')})
    fieldnames += per_keys
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fieldnames})
    summary_path = os.path.join(out, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(format_table(aggregate) + '\n')
        for row in aggregate:
            for name in per_keys:
                f.write('%s/%d %s: %.2f ms\n'
                        % (row['backend'], row['streams'],
                           name[len('per_image_'):], row[name]))
    print('\nPer-run CSV: %s' % csv_path)
    print('Summary:     %s' % summary_path)
    print('\n===== Summary =====')
    print(format_table(aggregate))
    for row in aggregate:
        for name in per_keys:
            print('  %s/%d %s = %.2f ms'
                  % (row['backend'], row['streams'],
                     name[len('per_image_'):], row[name]))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--streams', nargs='+', type=int, default=list(STREAM_SIZES))
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--stages', default='pgie,lpd,lpr,make,type',
                    help='comma-separated subset of pipeline stages to keep, in '
                         'any order: %s (default all five); pgie is required and '
                         'must stay first, lpr pulls in lpd' % ', '.join(STAGE_KINDS))
    ap.add_argument('--lpd-precision', choices=('int8', 'fp16'), default='int8',
                    help='prebuilt LPD engine precision to load '
                         '(default int8; fp16 was built for the precision-sweep '
                         'experiments)')
    ap.add_argument('--video', default=DEFAULT_VIDEO)
    ap.add_argument('--lp-dir', default=DEFAULT_LP_DIR,
                    help='root dir holding the LPD/ and LPR/ model dirs '
                         '(default %s)' % DEFAULT_LP_DIR)
    ap.add_argument('--samples-models', default=SAMPLES_MODELS,
                    help='DeepStream samples/models root holding '
                         'Primary_Detector/, Secondary_VehicleMake/, '
                         'Secondary_VehicleTypes/')
    ap.add_argument('--triton-bin', default=DEFAULT_TRITON_BIN)
    ap.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'lpr_benchmark_out'))
    ap.add_argument('--copy-models', action='store_true',
                    help='copy the ONNX into the model repo instead of symlinking')
    ap.add_argument('--manage-triton', action='store_true', default=True,
                    help='start/stop the Triton server around runs as needed')
    ap.add_argument('--no-manage-triton', dest='manage_triton', action='store_false')
    ap.add_argument('--warmup', action='store_true',
                    help='run each config once unmeasured first '
                         '(absorbs engine compilation, Triton cold start)')
    ap.add_argument('--child', help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        return run_child(args.child)

    try:
        args.stage_kinds = select_stage_kinds(args.stages)
    except ValueError as e:
        ap.error(str(e))

    paths = model_paths(args.lp_dir, args.samples_models, args.lpd_precision)
    if not os.path.isfile(args.video):
        ap.error('--video not found: %s' % args.video)
    needed_models = {s['model'] for s in stage_properties(paths, args.lp_dir)
                     if s['kind'] in args.stage_kinds}
    for name in needed_models:
        mod = paths[name]
        for key in ('onnx', 'engine', 'labels'):
            p = mod.get(key)
            if not p:
                ap.error('missing %s path for %s' % (key, name))
            if not os.path.isfile(p):
                ap.error('%s %s not found: %s' % (name, key, p))
    if 'lpr' in args.stage_kinds and not os.path.isfile(CUSTOM_LPR_LIB):
        print('WARNING: custom LPR parse lib not found: %s (lpr sgie will run '
              'without plate parsing)' % CUSTOM_LPR_LIB)
    for p, label in ((TRACKER_LIB, 'tracker lib'), (TRACKER_CFG, 'tracker config')):
        if not os.path.isfile(p):
            ap.error('%s not found: %s' % (label, p))

    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    print('Standalone LPR benchmark (nvinfer vs triton)')
    print('  pipeline: streammux -> pgie -> tracker -> lpd -> lpr -> '
          'vehiclemake -> vehicletypes -> nvvidconv -> nvosd -> fakesink')
    print('  video    : %s' % args.video)
    print('  streams  : %s' % ' '.join(str(s) for s in args.streams))
    print('  lpd      : %s' % args.lpd_precision)
    rows = bench(args, out_root)
    write_results(out_root, rows, args.streams)
    return 0


if __name__ == '__main__':
    sys.exit(main())