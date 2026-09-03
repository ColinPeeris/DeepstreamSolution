"""Per-model inference timing helpers.

The nvinfer element does not expose a simple Gst property for per-model
inference time in recent DeepStream, and DeepStream's C++ latency-measurement
env vars only work with NVIDIA's ``deepstream-app``/sample apps -- not custom
Python pipelines. These helpers measure per-model timing directly by attaching a
``BUFFER`` probe to each nvinfer's src pad.

One src-pad buffer corresponds to one inference batch, so the measured interval
is a per-batch time; dividing it by the model's ``batch-size`` yields an
approximate per-image time.
"""

import time
from typing import Dict, List, Optional

from gi.repository import Gst


class InferenceTimingMonitor:
    """Times each nvinfer element and prints a summary at the end of the run.

    Attach the monitor to a list of ``nvinfer`` elements (the pipeline's
    inference engines) with :meth:`attach`, let the pipeline run, then call
    :meth:`print_summary` after the run finishes (on End-of-stream).

    Example::

        monitor = InferenceTimingMonitor()
        monitor.attach(inference_engines)
        # ... run the pipeline ...
        monitor.print_summary()
    """

    def __init__(self) -> None:
        """Initialize an empty timing monitor."""
        self._stats: List[Dict] = []

    def attach(self, inference_engines: List[Gst.Element]) -> None:
        """Attach per-model timing probes to every nvinfer element.

        Adds a ``BUFFER`` probe to each nvinfer's src pad that measures the
        wall-clock interval between successive output buffers. Statistics,
        including the model's configured batch size, are accumulated per model.

        Args:
            inference_engines: The list of ``nvinfer`` elements to time.
        """
        self._stats = []
        for engine in inference_engines:
            src_pad = engine.get_static_pad("src")
            try:
                batch_size = engine.get_property("batch-size")
            except Exception:
                batch_size = 1
            stats = {
                'name': engine.get_name(),
                'batch_size': batch_size,
                'count': 0,
                'total_ms': 0.0,
                'min_ms': None,
                'max_ms': 0.0,
                'last_t': None,
            }
            self._stats.append(stats)
            src_pad.add_probe(Gst.PadProbeType.BUFFER, self._probe, stats)

    def _probe(self, pad, info, stats) -> Gst.PadProbeReturn:
        """Per-buffer callback that accumulates per-batch interval timings."""
        now = time.time()
        if stats['last_t'] is not None:
            delta_ms = (now - stats['last_t']) * 1000.0
            stats['count'] += 1
            stats['total_ms'] += delta_ms
            if stats['min_ms'] is None or delta_ms < stats['min_ms']:
                stats['min_ms'] = delta_ms
            if delta_ms > stats['max_ms']:
                stats['max_ms'] = delta_ms
        stats['last_t'] = now
        return Gst.PadProbeReturn.OK

    def print_summary(self) -> None:
        """Print a per-model inference timing summary.

        Reports, for each nvinfer, the average per-batch interval and the
        approximate per-image time (per-batch interval divided by the model's
        ``batch-size``). A batch of size N means N images share one inference
        batch, so per-image reflects the amortized model cost per frame. Models
        with no observed buffers are reported as ``N/A``.
        """
        print("\n===== Per-model inference timing =====")
        for stats in self._stats:
            if stats['count'] > 0:
                avg_batch_ms = stats['total_ms'] / stats['count']
                batch_size = stats['batch_size'] or 1
                avg_per_image_ms = avg_batch_ms / batch_size
                print(f"  {stats['name']:<38} "
                      f"batch_size={batch_size:<3} "
                      f"per_batch={avg_batch_ms:7.1f} ms  "
                      f"per_image={avg_per_image_ms:7.2f} ms  "
                      f"(min={stats['min_ms']:6.2f} max={stats['max_ms']:6.2f} ms, n={stats['count']})")
            else:
                print(f"  {stats['name']:<38} N/A (no buffers observed)")
        print("=====================================")
