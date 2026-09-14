#!/usr/bin/env python3
"""Generate a Triton Inference Server model repository from an application JSON.

Reads the same ``inference_engines`` list the pipeline consumes (so the client
config and the served models can never drift apart) and, for each engine, writes:

    <repo>/<model_name>/config.pbtxt
    <repo>/<model_name>/1/model.<onnx|plan>

The model name comes from the engine's ``model_name`` key, else from its model
file basename. ONNX models are served with the ``onnxruntime`` backend by
default; pass ``--backend tensorrt_plan`` to serve pre-built ``.engine`` files
with the TensorRT backend instead. Engines whose model cannot be served by the
chosen backend (e.g. a TAO ``.etlt`` on onnxruntime) are skipped with a warning.

The model file is symlinked into the repository (``1/model.onnx`` ->
original path) rather than copied, so the repo never duplicates the gigabytes of
model bytes already present under ``models/`` or the DeepStream samples; Triton
follows the symlink and loads the original file. Pass ``--copy`` to fall back to
real file copies for filesystems that do not support symlinks.

When the ``onnx`` python package is available the ``config.pbtxt`` is written
with explicit input/output signatures read from the model graph (so they always
match the ``output-blob-names`` the DeepStream client parses); otherwise Triton
derives them from the model at load time. ``--execution-accelerator tensorrt``
(the default) requests the onnxruntime TensorRT execution provider, which avoids
onnxruntime's cuDNN-frontend planner on Ampere GPUs.

Once generated, point ``tritonserver`` at the repository, e.g.:

    tritonserver --model-repository=<repo> --grpc-port 8001

and launch the pipeline with a ``"inference_backend": "triton"`` config.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

from Utils.config import infer_triton_model_name

log = logging.getLogger('triton-model-repo')

# onnx.TensorProto element_type -> Triton data_type
_ONNX_TO_TRITON_DTYPE = {
    1: 'TYPE_FP32',    # FLOAT
    2: 'TYPE_UINT8',
    3: 'TYPE_INT8',
    4: 'TYPE_UINT16',
    5: 'TYPE_INT16',
    6: 'TYPE_INT32',
    7: 'TYPE_INT64',
    9: 'TYPE_BOOL',
    10: 'TYPE_FP16',   # FLOAT16
    11: 'TYPE_FP64',   # DOUBLE
    12: 'TYPE_UINT32',
    13: 'TYPE_UINT64',
}


def _model_source(engine: Dict[str, Any], backend: str) -> Optional[str]:
    """Pick the model file to serve for an engine, or ``None`` if unsupported."""
    prop = engine.get('property', {})
    if backend == 'tensorrt_plan':
        return prop.get('model-engine-file') or prop.get('onnx-file')
    onnx_file = prop.get('onnx-file')
    if onnx_file:
        return onnx_file
    engine_file = prop.get('model-engine-file')
    if engine_file:
        log.warning("engine '%s' has an engine file but no onnx-file; "
                    "serving the .engine with the onnxruntime backend as-is", engine['name'])
        return engine_file
    return None


def _triton_dtype(elem_type: int) -> str:
    return _ONNX_TO_TRITON_DTYPE.get(elem_type, 'TYPE_FP32')


def _onnx_io_specs(onnx, model_path: str):
    """Read the ONNX graph's input/output signatures if possible."""
    model = onnx.load(model_path, load_external_data=False)
    graph = model.graph

    def _tensor_spec(value_info):
        name = value_info.name
        tensor_type = value_info.type.tensor_type
        dims = []
        # Onnxruntime with max_batch_size>0 expects dims without the batch dim,
        # so drop the leading (batch) dimension.
        for i, dim in enumerate(tensor_type.shape.dim):
            if i == 0:
                continue
            dims.append(dim.dim_value if dim.dim_value > 0 else -1)
        return {
            'name': name,
            'data_type': _triton_dtype(tensor_type.elem_type),
            'dims': dims,
        }

    return ([_tensor_spec(vi) for vi in graph.input],
            [_tensor_spec(vi) for vi in graph.output])


def write_config_pbtxt(model_path: str,
                       model_name: str,
                       backend: str,
                       max_batch_size: int,
                       accelerator: str = 'tensorrt') -> None:
    """Write ``config.pbtxt`` for one served model."""
    if backend == 'tensorrt_plan':
        # Older Triton builds reject the ``auto_complete_config`` field, so for
        # TensorRT plans we emit the minimal config and let the backend derive
        # the I/O signature from the engine at load time.
        contents = ("name: \"{name}\"\n"
                    "backend: \"tensorrt\"\n"
                    "platform: \"tensorrt_plan\"\n"
                    "max_batch_size: {max_batch_size}\n"
                    ).format(name=model_name, max_batch_size=max_batch_size)
    else:
        try:
            import onnx  # type: ignore
        except ImportError:
            onnx = None
        inputs, outputs = [], []
        if onnx is not None:
            try:
                inputs, outputs = _onnx_io_specs(onnx, model_path)
            except Exception as exc:  # malformed/unreadable model
                log.warning("could not parse %s (%s); emitting minimal config "
                            "and letting the backend auto-complete", model_path, exc)
        # ``auto_complete_config`` is not valid in older Triton ModelConfig
        # protos; when the model cannot be parsed we simply omit the I/O blocks
        # and let the onnxruntime backend derive them from the graph. The
        # ``backend`` name is taken from ``--backend`` (older Triton images name
        # it ``onnxruntime``, newer ones ``onnxruntime_onnx``).
        lines = ["name: \"{name}\"".format(name=model_name),
                 "backend: \"{backend}\"".format(backend=backend),
                 "max_batch_size: {max_batch_size}".format(max_batch_size=max_batch_size)]
        if inputs and outputs:
            for kind, specs in (('input', inputs), ('output', outputs)):
                for spec in specs:
                    lines.append("{kind} {{".format(kind=kind))
                    lines.append("    name: \"{name}\"".format(name=spec['name']))
                    lines.append("    data_type: {data_type}".format(data_type=spec['data_type']))
                    lines.append("    dims: [{dims}]".format(
                        dims=', '.join(str(d) for d in spec['dims']) if spec['dims'] else '1'))
                    lines.append("  }")
        if accelerator:
            lines.append("optimization {")
            lines.append("  execution_accelerators {")
            lines.append("    gpu_execution_accelerator: [")
            lines.append("      {")
            lines.append("        name: \"{accelerator}\"".format(accelerator=accelerator))
            lines.append("        parameters { key: \"precision_mode\" value: \"FP16\" }")
            lines.append("      }")
            lines.append("    ]")
            lines.append("  }")
            lines.append("}")
        contents = "\n".join(lines) + "\n"
    return contents


def _link_model(model_source: str, model_dir: str, backend: str,
                copy: bool = False) -> str:
    os.makedirs(model_dir, exist_ok=True)
    if backend == 'tensorrt_plan':
        target = os.path.join(model_dir, 'model.plan')
    elif model_source.endswith('.onnx'):
        target = os.path.join(model_dir, 'model.onnx')
    else:
        target = os.path.join(model_dir, 'model.' + (model_source.rsplit('.', 1)[-1] or 'bin'))
    if copy:
        shutil.copy(model_source, target)
        return target
    source_abs = os.path.abspath(model_source)
    if os.path.islink(target) and os.readlink(target) == source_abs:
        return target
    if os.path.lexists(target):
        os.remove(target)
    os.symlink(source_abs, target)
    return target


def generate_model_repo(config_path: str,
                        repo_root: str,
                        backend: str = 'onnxruntime',
                        force: bool = False,
                        accelerator: str = 'tensorrt',
                        copy: bool = False) -> List[str]:
    """Generate the repository; returns the list of served model names.

    Args:
        config_path: Application JSON config (same one the pipeline reads).
        repo_root: Directory to generate the repository in.
        backend: Triton backend name written into ``config.pbtxt``.
        force: Overwrite existing ``config.pbtxt``/model files.
        accelerator: ONNX Runtime execution accelerator.
        copy: Real file copies instead of symlinks to the original models.
    """
    if backend not in ('onnxruntime', 'onnxruntime_onnx', 'tensorrt_plan'):
        raise ValueError("backend must be 'onnxruntime', 'onnxruntime_onnx' or 'tensorrt_plan', got %r"
                         % backend)
    if accelerator not in ('', 'none', 'tensorrt', 'cuda'):
        raise ValueError("accelerator must be 'none', 'cuda' or 'tensorrt', got %r" % accelerator)
    with open(config_path) as json_file:
        data = json.load(json_file)
    engines = data.get('inference_engines', [])
    if not engines:
        raise RuntimeError("no inference_engines found in %s" % config_path)

    os.makedirs(repo_root, exist_ok=True)
    served = []
    for engine in engines:
        name = engine['name']
        model_name = infer_triton_model_name(engine)
        model_source = _model_source(engine, backend)
        if not model_source or not os.path.isfile(model_source):
            log.warning("engine '%s': no model file found to serve, skipping", name)
            continue
        model_root = os.path.join(repo_root, model_name)
        os.makedirs(model_root, exist_ok=True)
        config_path_out = os.path.join(model_root, 'config.pbtxt')
        if os.path.exists(config_path_out) and not force:
            log.info("engine '%s': %s exists, skipping (use --force to overwrite)", name, config_path_out)
            served.append(model_name)
            continue
        prop = engine.get('property', {})
        max_batch_size = int(prop.get('batch-size', 1))
        accelerator_eff = '' if backend == 'tensorrt_plan' or accelerator in ('', 'none') else accelerator
        contents = write_config_pbtxt(model_source, model_name, backend, max_batch_size,
                                      accelerator=accelerator_eff)
        with open(config_path_out, 'w') as f:
            f.write(contents)
        version_dir = os.path.join(model_root, '1')
        linked = _link_model(model_source, version_dir, backend, copy=copy)
        log.info("engine '%s' -> model '%s' (backend=%s, batch=%d)", name, model_name, backend, max_batch_size)
        log.info("  config.pbtxt: %s", config_path_out)
        log.info("  model file:   %s", linked)
        served.append(model_name)
    return served


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Triton Inference Server model repository from an application JSON.")
    parser.add_argument('config', help='application JSON config (same one the pipeline reads)')
    parser.add_argument('--repo', required=True, help='model repository directory to generate')
    parser.add_argument('--backend', default='onnxruntime',
                        choices=['onnxruntime', 'onnxruntime_onnx', 'tensorrt_plan'],
                        help="Triton backend name to write into config.pbtxt "
                             "(default: onnxruntime, the name used by DeepStream's bundled Triton; "
                             "newer Triton images use onnxruntime_onnx instead)")
    parser.add_argument('--execution-accelerator', default='tensorrt',
                        choices=['none', 'cuda', 'tensorrt'],
                        help="ONNX Runtime execution accelerator requested via "
                             "optimization.execution_accelerators (default: tensorrt; "
                             "'none' omits the optimization block)")
    parser.add_argument('--force', action='store_true',
                        help='overwrite existing config.pbtxt/model files')
    parser.add_argument('--copy', action='store_true',
                        help='copy model files into the repo instead of symlinking them')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    served = generate_model_repo(args.config, args.repo, backend=args.backend, force=args.force,
                                 accelerator=args.execution_accelerator, copy=args.copy)
    if not served:
        log.error("no models were generated; fix the warnings above")
        sys.exit(1)
    log.info("generated %d model(s) in %s", len(served), args.repo)
    log.info("start the server with: tritonserver --model-repository=%s --grpc-port 8001", args.repo)


if __name__ == '__main__':
    main()