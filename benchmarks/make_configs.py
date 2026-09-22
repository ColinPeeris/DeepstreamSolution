"""Generate reproducible multi-stream benchmark configs.

Derives a matrix of configs from ''families'' of canonical configs
(``nvinfer`` and ``triton`` backends) for 1, 16 and 32 identical streams.

Families:

- ``lpr``: the full LPR pipeline (pgie detector + LPD + LPR + make/type
  classifiers + tracker) from ``configs/detector_tracker_classifier_lpr_*``.
- ``detector_tracker``: the Primary_Detector (resnet18 TrafficCamNet) GIE +
  tracker only, from ``configs/detector_tracker{,_triton}.json``. Isolates the
  traffic-camnet inference backend cost (no secondary GIEs, no custom libs).

Differences vs. the originals, applied to every generated config so both
backends are measured on an identical pipeline:

- ``sink_type`` is ``fakesink``: the ``filesink`` branch adds a CPU
  ``avenc_mpeg4`` encoder that bottlenecks at 16-32 streams and would mask the
  inference-backend difference this benchmark is measuring.
- ``va_filters`` are emptied: the VA event filters publish per-frame metadata
  to RabbitMQ (``send_metadata`` in ``va_filter_builder.py``), adding unrelated
  per-frame latency/noise.
- ``streammux_batch_size`` is set to ``min(streams, 16)`` so the stream-muxer
  actually batches 16 frames per buffer and the GIE models run at their
  configured ``batch-size: 16`` (matches the Triton repo max_batch_size).

Usage (repo root)::

    python3 benchmarks/make_configs.py [--family lpr|detector_tracker]

Generated files are named ``<family>_<backend>_<streams>.json``.
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(REPO_ROOT, 'configs')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')

FAMILIES: Dict[str, Dict[str, Any]] = {
    'lpr': {
        'canonical': {
            'nvinfer': 'detector_tracker_classifier_lpr_deepstream_8.json',
            'triton': 'detector_tracker_classifier_lpr_triton_deepstream_8.json',
        },
        'description': 'full LPR pipeline (pgie + LPD + LPR + make/type classifiers)',
    },
    'detector_tracker': {
        'canonical': {
            'nvinfer': 'detector_tracker.json',
            'triton': 'detector_tracker_triton.json',
        },
        'description': 'primary TrafficCamNet detector + tracker only',
    },
}

VIDEO_FILE = '/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264'
STREAM_SIZES = [1, 16, 32]
FRAMES_PER_STREAM = 1441  # verified frame count of VIDEO_FILE


def make_config(family: str, backend: str, streams: int) -> Dict[str, Any]:
    canonical = FAMILIES[family]['canonical'][backend]
    with open(os.path.join(CONFIGS_DIR, canonical)) as f:
        cfg: Dict[str, Any] = json.load(f)

    batch_size = min(streams, 16)
    if streams == 1:
        cfg['video_source'] = {
            'filename': VIDEO_FILE,
            'streammux_batch_size': batch_size,
        }
    else:
        cfg['video_source'] = {
            'filenames': [VIDEO_FILE] * streams,
            'streammux_batch_size': batch_size,
        }
    cfg['va_filters'] = []
    cfg['sink_type'] = 'fakesink'
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family', choices=sorted(FAMILIES),
                    default=None, help='generate only this family (default: all)')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    paths: List[str] = []
    families = [args.family] if args.family else sorted(FAMILIES)
    for family in families:
        for backend in ('nvinfer', 'triton'):
            for streams in STREAM_SIZES:
                cfg = make_config(family, backend, streams)
                out = os.path.join(OUT_DIR, '%s_%s_%d.json'
                                   % (family, backend, streams))
                with open(out, 'w') as f:
                    json.dump(cfg, f, indent=2)
                paths.append(out)
    print('Wrote %d configs to %s' % (len(paths), OUT_DIR))
    for path in paths:
        print('  %s' % os.path.relpath(path, REPO_ROOT))


if __name__ == '__main__':
    sys.exit(main())