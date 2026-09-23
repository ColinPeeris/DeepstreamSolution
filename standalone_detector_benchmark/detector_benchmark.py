"""Standalone single-file benchmark: DeepStream ``nvinfer`` vs ``nvinferserver``.

Measures end-to-end throughput for a TrafficCamNet primary-detector pipe
(no tracker, no secondary GIEs) across 1, 16 and 32 identical H.264 streams.

Fully self-contained: no dependency on this repository or ``benchmarks/``.
Copy this one file onto any machine with the DeepStream container (python3 +
pygobject + gst-python bindings and the DeepStream plugins on
``GST_PLUGIN_PATH``) and run::

    python3 detector_benchmark.py [--repeats 3] [--warmup]

The script generates its own artifacts under OUT (default
``./detector_benchmark_out/``):

- ``configs/pgie_nvinfer.txt``          gst-nvinfer plugin config
- ``configs/pgie_nvinferserver.txt``    gst-nvinferserver plugin config
- ``triton/model_repo/...``             Triton model repository (config.pbtxt
  + model.onnx, symlinked from the DeepStream install by default)
- ``logs/``                             per-run logs (engine build, timing)
- ``results.csv``                       one row per measured run
- ``summary.txt``                       mean fps / per-image per config

Apples-to-apples guarantees - both backends are given the identical workload:

- streammux batch size = ``min(streams, 16)`` (frames are batched before the GIE)
- model batch size 16 (nvinfer ``batch-size`` = nvinferserver ``max_batch_size``)
- FP16 (network-mode 2 / TensorRT FP16 execution accelerator)
- same input video, same N copies via separate decoders into one streammux

Diagnostics: every run is one fresh child process so GPU/library state (engine
builds, Triton warm-up) never leaks between measured runs; ``--warmup`` absorbs
one-off engine compilation. Each child prints the counted frame total, so a
different test video than ``sample_720p.h264`` is measured correctly.

Run isolation / exit codes:

- parent: builds configs + model repo, starts/stops Triton, times children
- child (invoked with ``--child RUN.json``): builds the GStreamer pipeline,
  prints ``End-of-stream`` (+ frame count + GIE timing), exits 0 on success,
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

DEFAULT_VIDEO = '/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264'
DEFAULT_ONNX = ('/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/'
                'resnet18_trafficcamnet_pruned.onnx')
DEFAULT_LABELS = ('/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/'
                  'labels.txt')
DEFAULT_TRITON_BIN = '/opt/tritonserver/bin/tritonserver'

BACKENDS = ('nvinfer', 'triton')
STREAM_SIZES = (1, 16, 32)

BATCH_SIZE = 16


def _property_lines(prop: Dict[str, Any]) -> List[str]:
    return ['%s=%s' % (k, prop[k]) for k in prop]


def write_nvinfer_config(path: str, prop: Dict[str, Any],
                         class_attrs: Dict[str, Any]) -> None:
    """Render the gst-nvinfer config file exactly as the repo's generator."""
    with open(path, 'w') as f:
        f.write('[property] \n')
        f.write('\n'.join(_property_lines(prop)) + '\n')
        f.write('\n')
        f.write('[class-attrs-all] \n')
        f.write('\n'.join(_property_lines(class_attrs)) + '\n')


def write_nvinferserver_config(path: str, prop: Dict[str, Any],
                               class_attrs: Dict[str, Any], model_name: str,
                               server_url: str) -> None:
    """Render the gst-nvinferserver config (DeepStream-8 protobuf format).

    Mirrors the format produced by Utils/config.py for the primary detector so
    the inference context is identical to the nvinfer path's property block.
    """
    out_names = [n.strip() for n in str(prop['output-blob-names']).split(';') if n.strip()]
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
                 % ('IMAGE_FORMAT_BGR' if int(prop.get('model-color-format', 0)) == 1
                    else 'IMAGE_FORMAT_RGB'))
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
    lines.append('    detection {')
    lines.append('      num_detected_classes: %s' % prop['num-detected-classes'])
    pre = class_attrs.get('pre-cluster-threshold')
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
    lines.append('  process_mode: PROCESS_MODE_FULL_FRAME')
    lines.append('  operate_on_gie_id: -1')
    lines.append('  interval: %s' % prop.get('interval', 0))
    lines.append('}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def gie_properties(onnx_path: str, labels_path: str) -> Dict[str, Any]:
    """Primary-detector property block shared by both backend configs.

    ``model-engine-file`` is the canonical DeepStream path (the ONNX path plus
    the ``_b<batch>_gpu<gpu>_<precision>.engine`` suffix); mvinfer always
    serializes a freshly built engine there and deserializes from it on later
    runs, so the first (warm-up) run builds it and the rest are fast.
    """
    engine_path = '%s_b%d_gpu0_fp16.engine' % (onnx_path, BATCH_SIZE)
    return {
        'gpu-id': 0,
        'net-scale-factor': 0.0039215697906911373,
        'onnx-file': onnx_path,
        'model-engine-file': engine_path,
        'labelfile-path': labels_path,
        'batch-size': BATCH_SIZE,
        'network-mode': 2,
        'process-mode': 1,
        'model-color-format': 0,
        'num-detected-classes': 4,
        'interval': 0,
        'gie-unique-id': 1,
        'output-blob-names': 'output_bbox/BiasAdd:0;output_cov/Sigmoid:0',
    }


CLASS_ATTRS = {
    'pre-cluster-threshold': 0.2,
    'eps': 0.2,
    'group-threshold': 1,
}


def model_repo_config_pbtxt() -> str:
    """Triton model config: onnxruntime backend + TensorRT FP16 accelerator."""
    return ('name: "resnet18_trafficcamnet"\n'
            'backend: "onnxruntime"\n'
            'max_batch_size: 16\n'
            'input {\n'
            '    name: "input_1:0"\n'
            '    data_type: TYPE_FP32\n'
            '    dims: [3, 544, 960]\n'
            '  }\n'
            'output {\n'
            '    name: "output_cov/Sigmoid:0"\n'
            '    data_type: TYPE_FP32\n'
            '    dims: [4, 34, 60]\n'
            '  }\n'
            'output {\n'
            '    name: "output_bbox/BiasAdd:0"\n'
            '    data_type: TYPE_FP32\n'
            '    dims: [16, 34, 60]\n'
            '  }\n'
            'optimization {\n'
            '  execution_accelerators {\n'
            '    gpu_execution_accelerator: [\n'
            '      {\n'
            '        name: "tensorrt"\n'
            '        parameters { key: "precision_mode" value: "FP16" }\n'
            '      }\n'
            '    ]\n'
            '  }\n'
            '}\n')


def ensure_model_repo(repo_dir: str, onnx_path: str, copy_models: bool) -> str:
    """Create the Triton model repository (no-op when it already exists)."""
    model_dir = os.path.join(repo_dir, 'resnet18_trafficcamnet')
    version_dir = os.path.join(model_dir, '1')
    if os.path.isfile(os.path.join(model_dir, 'config.pbtxt')):
        return repo_dir
    os.makedirs(version_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'config.pbtxt'), 'w') as f:
        f.write(model_repo_config_pbtxt())
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


def _child_pipeline(backend: str, streams: int, video: str, config_path: str) -> None:
    """Build and run one streammux -> GIE -> fakesink pipeline (child process)."""
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import GLib, Gst

    Gst.init(None)
    pipeline = Gst.Pipeline()

    streammux = Gst.ElementFactory.make('nvstreammux', 'streammux')
    if not streammux:
        raise RuntimeError('Unable to create nvstreammux')
    pipeline.add(streammux)
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', min(streams, BATCH_SIZE))
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

    gie = Gst.ElementFactory.make('nvinferserver' if backend == 'triton' else 'nvinfer',
                                  'pgie_detector_1')
    if not gie:
        raise RuntimeError('Unable to create %s element' % backend)
    gie.set_property('config-file-path', config_path)
    sink = Gst.ElementFactory.make('fakesink', 'fakesink')
    if not sink:
        raise RuntimeError('Unable to create fakesink')
    sink.set_property('sync', False)
    pipeline.add(gie)
    pipeline.add(sink)

    if not streammux.link(gie):
        raise RuntimeError('Failed to link streammux -> %s' % backend)
    if not gie.link(sink):
        raise RuntimeError('Failed to link %s -> fakesink' % backend)

    timing = {'last': None, 'total_ms': 0.0, 'n': 0}

    def _time_batches(pad, info, data):
        now = time.monotonic()
        if data['last'] is not None:
            data['total_ms'] += (now - data['last']) * 1000.0
            data['n'] += 1
        data['last'] = now
        return Gst.PadProbeReturn.OK

    gie.get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER, _time_batches, timing)

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
    if timing['n'] > 0:
        per_batch = timing['total_ms'] / timing['n']
        per_image = per_batch / BATCH_SIZE
        print('PGIE per_batch=%.2f ms per_image=%.2f ms (n=%d)'
              % (per_batch, per_image, timing['n']), flush=True)


def run_child(run_json: str) -> int:
    with open(run_json) as f:
        run = json.load(f)
    _child_pipeline(run['backend'], run['streams'], run['video'], run['config'])
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
    lines = ['%-8s %5s %10s %9s %4s %10s' %
             ('backend', 'strms', 'wall_s', 'fps', 'rep', 'per_img_ms')]
    for r in rows:
        lines.append('%-8s %5d %10.1f %9.1f %4d %10.3f' % (
            r['backend'], r['streams'], r['wall_s_mean'], r['fps_mean'],
            r['repeats'], r.get('per_image') or float('nan')))
    return '\n'.join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--streams', nargs='+', type=int, default=list(STREAM_SIZES))
    ap.add_argument('--repeats', type=int, default=3)
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

    for path, name in ((args.video, 'video'), (args.onnx, 'onnx'), (args.labels, 'labels')):
        if not os.path.isfile(path):
            ap.error('%s not found: %s (override with --%s)' % (name, path, name))

    out = os.path.abspath(args.out)
    configs_dir = os.path.join(out, 'configs')
    repo_dir = os.path.join(out, 'triton', 'model_repo')
    logs_dir = os.path.join(out, 'logs')
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    prop = gie_properties(args.onnx, args.labels)
    nvinfer_cfg = os.path.join(configs_dir, 'pgie_nvinfer.txt')
    triton_cfg = os.path.join(configs_dir, 'pgie_nvinferserver.txt')
    write_nvinfer_config(nvinfer_cfg, prop, CLASS_ATTRS)
    write_nvinferserver_config(triton_cfg, prop, CLASS_ATTRS,
                               model_name='resnet18_trafficcamnet',
                               server_url='localhost:8001')
    ensure_model_repo(repo_dir, args.onnx, args.copy_models)

    print('Standalone detector benchmark')
    print('  pipeline: streammux -> pgie_(nvinfer|nvinferserver) -> fakesink '
          '(no tracker/OSD)')
    print('  video     : %s' % args.video)
    print('  streams   : %s' % ' '.join(str(s) for s in args.streams))
    print('  configs   : %s' % configs_dir)
    print('  triton repo: %s' % repo_dir)

    script = os.path.abspath(__file__)
    rows: List[Dict[str, Any]] = []
    for backend in BACKENDS:
        for streams in args.streams:
            if args.manage_triton:
                if backend == 'triton':
                    start_triton(repo_dir,
                                 os.path.join(logs_dir, 'triton_server.log'),
                                 args.triton_bin)
                else:
                    stop_triton()
            cfg = triton_cfg if backend == 'triton' else nvinfer_cfg
            run = {'backend': backend, 'streams': streams, 'video': args.video,
                   'config': cfg}
            run_json = os.path.join(out, 'run_%s_%d.json' % (backend, streams))
            with open(run_json, 'w') as f:
                json.dump(run, f)
            print('=== %s | %d streams ===' % (backend, streams))
            if args.warmup:
                run_one(script, run_json,
                        os.path.join(logs_dir, '%s_%d_warmup.log' % (backend, streams)))
                print('  warmup done')
            for rep in range(1, args.repeats + 1):
                log_path = os.path.join(logs_dir, '%s_%d_rep%d.log' % (backend, streams, rep))
                wall_s, rc, frames, per_image = run_one(script, run_json, log_path)
                if rc != 0:
                    raise RuntimeError('run exited rc=%d (see %s)' % (rc, log_path))
                fps = frames / wall_s if frames else float('nan')
                rows.append({'backend': backend, 'streams': streams, 'repeat': rep,
                             'wall_s': wall_s, 'fps': fps, 'frames': frames,
                             'per_image': per_image})
                print('  rep %d: wall=%.1fs fps=%.1f' % (rep, wall_s, fps))

    if args.manage_triton:
        stop_triton()

    by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row['backend'], row['streams'])].append(row)
    aggregated: List[Dict[str, Any]] = []
    for backend in BACKENDS:
        for streams in args.streams:
            key = (backend, streams)
            if key not in by_key:
                continue
            group = by_key[key]
            aggregated.append({
                'backend': backend,
                'streams': streams,
                'repeats': len(group),
                'wall_s_mean': sum(r['wall_s'] for r in group) / len(group),
                'fps_mean': sum(r['fps'] for r in group) / len(group),
                'per_image': sum(r['per_image'] for r in group if r['per_image'])
                              / max(1, sum(1 for r in group if r['per_image'])),
            })

    print('\n===== Summary =====')
    print(format_table(aggregated))

    csv_path = os.path.join(out, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['backend', 'streams', 'repeat',
                                               'wall_s', 'fps', 'frames', 'per_image'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    summary_path = os.path.join(out, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(format_table(aggregated) + '\n')
    print('\nPer-run CSV: %s' % csv_path)
    print('Summary:     %s' % summary_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())