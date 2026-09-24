"""Standalone single-file benchmark: DeepStream ``nvinfer`` vs ``nvinferserver``.

Measures end-to-end throughput for a TrafficCamNet primary-detector pipe
(no tracker, no OSD) across 16 and 32 identical H.264 streams, with an
optional number of *detector* instances chained after the first one.

Only three sample models exist and the two tiny secondary classifiers (224x224
vehiclemake/vehicletypes) proved too light to move the needle, so every extra
stage is a fresh instance of the primary detector itself: full 544x960
resnet18_trafficcamnet detection on every batched frame. Same ONNX per stage,
but the pipeline sees N independent engine instances sharing the GPU -- nvinfer
instantiates each in-process (serialized GPU scheduling) while Triton serves
them all through one async server-side engine, so growing N is exactly where
the two backends diverge.

   --extra-models N  chain N extra detector instances after the base pgie
                     (N=0 is the single-model baseline; N=6 => 7 engines)
   --sweep [--sweep-max M]
                     run --extra-models 0..M back-to-back and merge summaries

The single model used is part of the DeepStream sample install (no downloads):

- samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx

Fully self-contained: no dependency on this repository or ``benchmarks/``.
Copy this one file onto any machine with the DeepStream container (python3 +
pygobject + gst-python bindings and the DeepStream plugins on
``GST_PLUGIN_PATH``) and run::

    python3 detector_benchmark.py [--extra-models 6] [--repeats 3] [--warmup]

The script generates its own artifacts under OUT (default
``./detector_benchmark_out/``); with ``--sweep`` results land in
``OUT/extra0|extra1|...|extraM`` and a combined ``sweep_summary.txt``:

- ``configs/{pgie,pgie2,..,pgieN}_b32_{nvinfer,nvinferserver}.txt``
  (per-stream-count configs: batch == streams, so ``b16`` and ``b32`` variants)
- ``triton/model_repo_b{16,32}/...``                Triton model repositories
  (one per batch, config.pbtxt + model.onnx, symlinked from the install by
  default)
- ``logs/``                                      per-run logs (engine build, timing)
- ``results.csv``                                one row per measured run
- ``summary.txt``                                mean fps / per-image per config

Apples-to-apples guarantees - both backends are given the identical workload:

- model batch = the number of streams, for both backends: nvinfer ``batch-size``
  = nvinferserver ``max_batch_size`` = streams (16 -> batch-16 engines, 32 ->
  batch-32 engines), and the streammux flushes every muxed batch of ``streams``
  frames through the engine in one shot
- FP16 (network-mode 2 / TensorRT FP16 execution accelerator)
- every chained stage runs the identical detector config (same batch, FP16,
  full-frame process mode), so stages differ only in their unique id / element
- same input video, same N copies via separate decoders into one streammux
- engines are (re)built only during ``--warmup``; every measured run deserializes
  the existing engine file, so engine-build time is never counted

Diagnostics: every run is one fresh child process so GPU/library state (engine
builds, Triton warm-up) never leaks between measured runs; ``--warmup`` absorbs
one-off engine compilation. Each child prints the counted frame total, so a
different test video than ``sample_720p.h264`` is measured correctly.

Run isolation / exit codes:

- parent: builds configs + model repo, starts/stops Triton, times children
- child (invoked with ``--child RUN.json``): builds the GStreamer pipeline,
  prints ``End-of-stream`` (+ frame count + per-GIE timing), exits 0 on success,
  2 on a GStreamer bus ERROR
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
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DS_SAMPLES = '/opt/nvidia/deepstream/deepstream/samples'
DEFAULT_VIDEO = os.path.join(DS_SAMPLES, 'streams', 'sample_720p.h264')
DEFAULT_ONNX = os.path.join(DS_SAMPLES, 'models', 'Primary_Detector',
                            'resnet18_trafficcamnet_pruned.onnx')
DEFAULT_LABELS = os.path.join(DS_SAMPLES, 'models', 'Primary_Detector', 'labels.txt')
DEFAULT_TRITON_BIN = '/opt/tritonserver/bin/tritonserver'

BACKENDS = ('nvinfer', 'triton')
STREAM_SIZES = (16, 32)

# Triton model-repo tensor specs for the sample models (from the DeepStream
# triton samples / the repo's own generated repository).
TRITON_MODEL_SPECS = {
    'resnet18_trafficcamnet': {
        'onnx': DEFAULT_ONNX,
        'input': ('input_1:0', '3, 544, 960'),
        'outputs': [('output_cov/Sigmoid:0', '4, 34, 60'),
                    ('output_bbox/BiasAdd:0', '16, 34, 60')],
    },
}


def _property_lines(prop: Dict[str, Any]) -> List[str]:
    return ['%s=%s' % (k, prop[k]) for k in prop]


def write_nvinfer_config(path: str, prop: Dict[str, Any],
                         class_attrs: Optional[Dict[str, Any]] = None) -> None:
    """Render a gst-nvinfer config file (detector or classifier)."""
    with open(path, 'w') as f:
        f.write('[property] \n')
        f.write('\n'.join(_property_lines(prop)) + '\n')
        f.write('\n')
        if class_attrs:
            f.write('[class-attrs-all] \n')
            f.write('\n'.join(_property_lines(class_attrs)) + '\n')


def write_nvinferserver_config(path: str, prop: Dict[str, Any],
                               class_attrs: Optional[Dict[str, Any]],
                               model_name: str, server_url: str) -> None:
    """Render a gst-nvinferserver config (DeepStream-8 protobuf format).

    Handles both the primary full-frame detector (``process-mode: 1``) and the
    secondary ROI classifiers (``process-mode: 2``), mirroring the format the
    repo's ``Utils/config.py`` produces so both backends share one property
    block.
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
    is_classifier = int(prop.get('is-classifier', 0)) == 1
    if is_classifier:
        threshold = prop.get('classifier-threshold')
        if threshold is not None:
            lines.append('    classification {')
            lines.append('      threshold: %s' % threshold)
            lines.append('    }')
    elif 'num-detected-classes' in prop:
        lines.append('    detection {')
        lines.append('      num_detected_classes: %s' % prop['num-detected-classes'])
        pre = (class_attrs or {}).get('pre-cluster-threshold')
        if pre is not None:
            lines.append('      per_class_params {')
            lines.append('        key: 0')
            lines.append('        value { pre_threshold: %s }' % pre)
            lines.append('      }')
            lines.append('      nms {')
            lines.append('        confidence_threshold: %s' % pre)
            lines.append('        topk: 20')
            lines.append('        iou_threshold: 0.5')
            lines.append('      }')
        lines.append('    }')
    lines.append('  }')
    lines.append('}')
    lines.append('input_control {')
    lines.append('  process_mode: %s'
                 % ('PROCESS_MODE_FULL_FRAME' if process_mode == 1
                    else 'PROCESS_MODE_CLIP_OBJECTS'))
    lines.append('  operate_on_gie_id: %s' % prop.get('operate-on-gie-id', -1))
    if 'operate-on-class-ids' in prop:
        class_ids = [x.strip() for x in str(prop['operate-on-class-ids']).split(',') if x.strip()]
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


def gie_properties(onnx_path: str, labels_path: str,
                   gie_unique_id: int = 1, batch: int = 16) -> Dict[str, Any]:
    """Primary-detector property block shared by both backend configs.

    ``model-engine-file`` is the canonical DeepStream path (the ONNX path plus
    the ``_b<batch>_gpu<gpu>_<precision>.engine`` suffix); nvinfer always
    serializes a freshly built engine there and deserializes from it on later
    runs, so the first (warm-up) run builds it and the rest are fast. Chained
    detector instances share the same ONNX/engine *file* but each deserializes
    its own engine context (they differ only by ``gie-unique-id``).
    """
    engine_path = '%s_b%d_gpu0_fp16.engine' % (onnx_path, batch)
    return {
        'gpu-id': 0,
        'net-scale-factor': 0.0039215697906911373,
        'onnx-file': onnx_path,
        'model-engine-file': engine_path,
        'labelfile-path': labels_path,
        'batch-size': batch,
        'network-mode': 2,
        'process-mode': 1,
        'model-color-format': 0,
        'num-detected-classes': 4,
        'interval': 0,
        'gie-unique-id': gie_unique_id,
        'output-blob-names': 'output_bbox/BiasAdd:0;output_cov/Sigmoid:0',
    }


CLASS_ATTRS = {
    'pre-cluster-threshold': 0.2,
    'eps': 0.2,
    'group-threshold': 1,
}


def model_repo_config_pbtxt(model_name: str, max_batch_size: int) -> str:
    """Triton model config: onnxruntime backend + TensorRT FP16 accelerator."""
    spec = TRITON_MODEL_SPECS[model_name]
    input_name, input_dims = spec['input']
    lines = ['name: "%s"' % model_name,
             'backend: "onnxruntime"',
             'max_batch_size: %d' % max_batch_size,
             'input {',
             '    name: "%s"' % input_name,
             '    data_type: TYPE_FP32',
             '    dims: [%s]' % input_dims,
             '  }']
    for out_name, out_dims in spec['outputs']:
        lines.append('output {')
        lines.append('    name: "%s"' % out_name)
        lines.append('    data_type: TYPE_FP32')
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


def ensure_model_repo(repo_dir: str, model_names: List[str], max_batch_size: int,
                      copy_models: bool) -> str:
    """Create Triton model-repository entries (no-op when already present)."""
    for name in model_names:
        model_dir = os.path.join(repo_dir, name)
        version_dir = os.path.join(model_dir, '1')
        if os.path.isfile(os.path.join(model_dir, 'config.pbtxt')):
            continue
        os.makedirs(version_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'config.pbtxt'), 'w') as f:
            f.write(model_repo_config_pbtxt(name, max_batch_size))
        onnx_path = TRITON_MODEL_SPECS[name]['onnx']
        target = os.path.join(version_dir, 'model.onnx')
        if copy_models:
            shutil.copyfile(onnx_path, target)
        else:
            os.symlink(onnx_path, target)
    return repo_dir


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


def start_triton(repo_dir: str, log_path: str, triton_bin: str, timeout: int = 300) -> None:
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


def _child_pipeline(backend: str, streams: int, batch: int, video: str,
                    gies: List[Dict[str, str]]) -> None:
    """Build and run one streammux -> GIE chain -> fakesink pipeline (child)."""
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import GLib, Gst

    def _time_batches(pad, info, data):
        now = time.monotonic()
        if data['last'] is not None:
            data['total_ms'] += (now - data['last']) * 1000.0
            data['n'] += 1
        data['last'] = now
        return Gst.PadProbeReturn.OK

    Gst.init(None)
    pipeline = Gst.Pipeline()

    streammux = Gst.ElementFactory.make('nvstreammux', 'streammux')
    if not streammux:
        raise RuntimeError('Unable to create nvstreammux')
    pipeline.add(streammux)
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', batch)
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
        source.set_property('location', video)
        source.link(parser)
        parser.link(decoder)
        decoder.get_static_pad('src').link(streammux.get_request_pad('sink_%d' % i))
        decoder.get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER, _count, frame_count)

    factory = 'nvinferserver' if backend == 'triton' else 'nvinfer'
    prev = streammux
    for spec in gies:
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

    sink = Gst.ElementFactory.make('fakesink', 'fakesink')
    if not sink:
        raise RuntimeError('Unable to create fakesink')
    sink.set_property('sync', False)
    pipeline.add(sink)
    if not prev.link(sink):
        raise RuntimeError('Failed to link %s -> fakesink' % prev.get_name())

    timings: Dict[str, Dict[str, Any]] = {}
    for spec in gies:
        gie = pipeline.get_by_name(spec['name'])
        data = {'kind': spec['kind'], 'last': None, 'total_ms': 0.0, 'n': 0}
        timings[spec['name']] = data
        gie.get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER,
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
    for data in timings.values():
        if data['n'] > 0:
            per_batch = data['total_ms'] / data['n']
            per_image = per_batch / batch
            print('%s per_batch=%.2f ms per_image=%.2f ms (n=%d)'
                  % (data['kind'], per_batch, per_image, data['n']), flush=True)


def run_child(run_json: str) -> int:
    with open(run_json) as f:
        run = json.load(f)
    _child_pipeline(run['backend'], run['streams'], run['batch'], run['video'],
                    run['gies'])
    return 0


def run_one(script: str, run_json: str, log_path: str, timeout: int = 1800
            ) -> Tuple[float, int, Optional[int], Optional[float]]:
    t0 = time.monotonic()
    with open(log_path, 'w') as log:
        ret = subprocess.run(
            [sys.executable, '-u', script, '--child', run_json],
            stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    wall_s = time.monotonic() - t0
    with open(log_path) as f:
        content = f.read()
    if 'End-of-stream' not in content:
        raise RuntimeError('run never reached End-of-stream (see %s)' % log_path)
    frames = None
    m = re.search(r'PROCESSED_FRAMES=(\d+)', content)
    if m:
        frames = int(m.group(1))
    per_image = None
    m = re.search(r'PGIE per_batch=[\d.]+ ms per_image=([\d.]+) ms', content)
    if m:
        per_image = float(m.group(1))
    return wall_s, ret.returncode, frames, per_image


def format_table(rows: List[Dict[str, Any]]) -> str:
    lines = ['%-8s %5s %9s %10s %9s %4s %10s' %
             ('backend', 'extra', 'strms', 'wall_s', 'fps', 'rep', 'per_img_ms')]
    for r in rows:
        lines.append('%-8s %5d %9d %10.1f %9.1f %4d %10.3f' % (
            r['backend'], r.get('extra_models', 0), r['streams'], r['wall_s_mean'],
            r['fps_mean'], r['repeats'], r.get('per_image') or float('nan')))
    return '\n'.join(lines)


def bench(args: argparse.Namespace, extra_models: int, out: str,
          triton_persist: bool) -> List[Dict[str, Any]]:
    """Run the 1/16/32-stream matrix for one ``extra_models`` count."""
    configs_dir = os.path.join(out, 'configs')
    logs_dir = os.path.join(out, 'logs')
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    stages: List[Dict[str, Any]] = [
        {'kind': 'PGIE', 'name': 'pgie_detector_1',
         'model': 'resnet18_trafficcamnet',
         'cfg': 'pgie'},
    ]
    for i in range(extra_models):
        stages.append({
            'kind': 'PGIE%d' % (i + 2),
            'name': 'pgie_detector_%d' % (i + 2),
            'model': 'resnet18_trafficcamnet',
            'cfg': 'pgie%d' % (i + 2),
        })
    for stage in stages:
        stage['uid'] = int(stage['name'].rsplit('_', 1)[1])

    repo_models = ['resnet18_trafficcamnet']
    available = [m for m in repo_models if os.path.isfile(TRITON_MODEL_SPECS[m]['onnx'])]

    names = ' + '.join(s['name'] for s in stages)
    print()
    print('=== extra-models=%d | %s ===' % (extra_models, names))
    print('  video      : %s' % args.video)
    print('  streams    : %s' % ' '.join(str(s) for s in args.streams))

    script = os.path.abspath(__file__)
    rows: List[Dict[str, Any]] = []
    for streams in args.streams:
        batch = streams
        cfg_suffix = 'b%d' % batch
        stage_configs: Dict[str, Dict[str, str]] = {}
        for stage in stages:
            uid = stage['uid']
            prop = gie_properties(args.onnx, args.labels, uid, batch)
            nvinfer_cfg = os.path.join(
                configs_dir, '%s_%s_nvinfer.txt' % (stage['cfg'], cfg_suffix))
            triton_cfg = os.path.join(
                configs_dir, '%s_%s_nvinferserver.txt' % (stage['cfg'], cfg_suffix))
            write_nvinfer_config(nvinfer_cfg, prop, CLASS_ATTRS)
            write_nvinferserver_config(triton_cfg, prop, CLASS_ATTRS,
                                       model_name=stage['model'],
                                       server_url='localhost:8001')
            stage_configs[stage['kind']] = {'nvinfer': nvinfer_cfg, 'triton': triton_cfg}
        repo_dir = os.path.join(out, 'triton', 'model_repo_%s' % cfg_suffix)
        ensure_model_repo(repo_dir, available, batch, args.copy_models)
        print('  batch=%d (streams=%d)  triton repo: %s' % (batch, streams, repo_dir))

        for backend in BACKENDS:
            if args.manage_triton:
                if backend == 'triton':
                    start_triton(repo_dir,
                                 os.path.join(logs_dir, 'triton_server_%s.log' % cfg_suffix),
                                 args.triton_bin)
                elif not triton_persist:
                    stop_triton()
            gies = [{'kind': s['kind'], 'name': s['name'],
                     'config': stage_configs[s['kind']][backend]} for s in stages]
            run = {'backend': backend, 'streams': streams, 'batch': batch,
                   'video': args.video, 'gies': gies}
            run_json = os.path.join(out, 'run_%s_%d.json' % (backend, streams))
            with open(run_json, 'w') as f:
                json.dump(run, f)
            print('--- %s | %d streams (batch %d) ---' % (backend, streams, batch))
            if args.warmup:
                run_one(script, run_json,
                        os.path.join(logs_dir, '%s_%d_warmup.log' % (backend, streams)))
                print('  warmup done')
            for rep in range(1, args.repeats + 1):
                log_path = os.path.join(logs_dir, '%s_%d_rep%d.log'
                                        % (backend, streams, rep))
                wall_s, rc, frames, per_image = run_one(script, run_json, log_path)
                if rc != 0:
                    raise RuntimeError('run exited rc=%d (see %s)' % (rc, log_path))
                fps = frames / wall_s if frames else float('nan')
                rows.append({
                    'backend': backend, 'streams': streams, 'batch': batch,
                    'repeat': rep, 'wall_s': wall_s, 'fps': fps,
                    'frames': frames, 'per_image': per_image,
                })
                print('  rep %d: wall=%.1fs fps=%.1f' % (rep, wall_s, fps))
        if args.manage_triton and not triton_persist:
            stop_triton()

    by_key: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(BACKENDS.index(row['backend']), row['streams'])].append(row)
    aggregated: List[Dict[str, Any]] = []
    for bi, backend in enumerate(BACKENDS):
        for streams in args.streams:
            group = by_key.get((bi, streams), [])
            if not group:
                continue
            aggregated.append({
                'backend': backend,
                'streams': streams,
                'batch': streams,
                'repeats': len(group),
                'wall_s_mean': sum(r['wall_s'] for r in group) / len(group),
                'fps_mean': sum(r['fps'] for r in group) / len(group),
                'per_image': sum(r['per_image'] for r in group if r['per_image'])
                              / max(1, sum(1 for r in group if r['per_image'])),
            })

    print('  %s extra-model count summary' % extra_models)
    print(format_table([{**r, 'extra_models': extra_models} for r in aggregated]))

    csv_path = os.path.join(out, 'results.csv')
    fieldnames = ['backend', 'streams', 'batch', 'repeat', 'wall_s', 'fps',
                  'frames', 'per_image']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    summary_path = os.path.join(out, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(format_table([{**r, 'extra_models': extra_models}
                              for r in aggregated]) + '\n')
    print('  Per-run CSV: %s' % csv_path)
    print('  Summary:     %s' % summary_path)
    return rows


def write_sweep(summary_path: str, csv_path: str,
                all_rows: List[Dict[str, Any]],
                aggregated: List[Dict[str, Any]]) -> None:
    with open(summary_path, 'w') as f:
        f.write('%-8s %5s %5s %9s %10s %4s %10s\n' %
                ('backend', 'extra', 'strms', 'wall_s', 'fps', 'rep', 'per_img_ms'))
        for r in aggregated:
            f.write('%-8s %5d %5d %9.1f %10.1f %4d %10.3f\n' % (
                r['backend'], r['extra_models'], r['streams'], r['wall_s_mean'],
                r['fps_mean'], r['repeats'], r.get('per_image') or float('nan')))
        f.write('\n%s\n' % format_table(aggregated))
    fieldnames = ['extra_models', 'backend', 'streams', 'batch', 'repeat',
                  'wall_s', 'fps', 'frames', 'per_image']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--streams', nargs='+', type=int, default=list(STREAM_SIZES))
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--extra-models', type=int, default=0,
                    help='number of extra detector instances chained after the '
                         'base pgie (same trafficcamnet model, fresh engine per '
                         'stage; N=0 is the single-model baseline; default 0)')
    ap.add_argument('--sweep', action='store_true',
                    help='run extra-models 0..--sweep-max back-to-back under '
                         'OUT/extra<count> and write a merged sweep_summary.txt')
    ap.add_argument('--sweep-max', type=int, default=6,
                    help='highest extra-models count used by --sweep (default 6)')
    ap.add_argument('--video', default=DEFAULT_VIDEO)
    ap.add_argument('--onnx', default=DEFAULT_ONNX)
    ap.add_argument('--labels', default=DEFAULT_LABELS)
    ap.add_argument('--triton-bin', default=DEFAULT_TRITON_BIN)
    ap.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'detector_benchmark_out'))
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

    if args.extra_models < 0:
        ap.error('--extra-models must be >= 0')

    paths = [('--video', args.video), ('--onnx', args.onnx), ('--labels', args.labels)]
    for flag, path in paths:
        if not os.path.isfile(path):
            ap.error('%s not found: %s (override with %s)' % (flag, path, flag))

    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    print('Standalone detector benchmark (nvinfer vs triton)')
    print('  pipeline: streammux -> pgie -> pgie2..pgieN+1 -> fakesink '
          '(chained detector instances, same trafficcamnet model; no tracker/OSD)')
    print('  video    : %s' % args.video)

    if args.sweep:
        all_rows: List[Dict[str, Any]] = []
        sweep_counts = range(0, args.sweep_max + 1)
        for extra in sweep_counts:
            out_k = os.path.join(out_root, 'extra%d' % extra)
            os.makedirs(out_k, exist_ok=True)
            rows = bench(args, extra_models=extra, out=out_k,
                         triton_persist=False)
            all_rows.extend({'extra_models': extra, **r} for r in rows)
        if args.manage_triton:
            stop_triton()
        aggregated: List[Dict[str, Any]] = []
        for extra in sweep_counts:
            group = [r for r in all_rows if r['extra_models'] == extra]
            for backend in BACKENDS:
                for streams in args.streams:
                    reps = [r for r in group
                            if r['backend'] == backend and r['streams'] == streams]
                    if not reps:
                        continue
                    aggregated.append({
                        'backend': backend, 'streams': streams, 'batch': streams,
                        'extra_models': extra, 'repeats': len(reps),
                        'wall_s_mean': sum(r['wall_s'] for r in reps) / len(reps),
                        'fps_mean': sum(r['fps'] for r in reps) / len(reps),
                        'per_image': sum(r['per_image'] for r in reps
                                         if r['per_image'])
                                      / max(1, sum(1 for r in reps
                                                   if r['per_image'])),
                    })
        write_sweep(os.path.join(out_root, 'sweep_summary.txt'),
                    os.path.join(out_root, 'sweep_results.csv'),
                    all_rows, aggregated)
        print('\n===== Sweep summary =====')
        print(format_table(aggregated))
        print('\nSweep summary: %s' % os.path.join(out_root, 'sweep_summary.txt'))
        print('Sweep CSV:     %s' % os.path.join(out_root, 'sweep_results.csv'))
    else:
        bench(args, extra_models=args.extra_models, out=out_root,
              triton_persist=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())