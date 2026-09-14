import os
import json
from typing import Any, Dict, List, Optional, Tuple


def infer_triton_model_name(inference_engine_config: Dict[str, Any]) -> str:
    """Derive the Triton model-repository name for an inference engine entry.

    Prefers an explicit ``model_name`` key, then the engine's model-file
    basename, then falls back to the pipeline ``name``.

    Args:
        inference_engine_config: A dict describing a single engine, with at
            least a ``name`` key, and optionally a ``model_name`` key.

    Returns:
        The model name to use in the Triton model repository.
    """
    explicit = inference_engine_config.get('model_name')
    if explicit:
        return explicit
    property_block = inference_engine_config.get('property', {})
    for key in ('onnx-file', 'tlt-encoded-model', 'model-engine-file'):
        path = property_block.get(key)
        if path:
            return os.path.splitext(os.path.basename(path))[0]
    return inference_engine_config.get('name', 'model')


class Config:
    """Parses a JSON application config and prepares DeepStream element configs.

    For every entry in the supplied JSON's ``inference_engines`` list the class
    writes a plain-text inference-engine config file (used via
    ``config-file-path``). The file format depends on the selected backend:

    - ``inference_backend: "nvinfer"`` (default): a ``nvinfer`` config file with
      a ``[property]`` (and optional ``class-attrs-all``) block.
    - ``inference_backend: "triton"``: a ``nvinferserver`` config file with a
      ``[property]`` block pointing at the Triton server (``server-url``,
      ``model-name``, ``model-version``, ``protocol-type``) and an
      ``[infer-config]`` block carrying the model/inference context (derived
      from the same ``property`` data as the nvinfer path).

    If an entry also declares a ``preprocess`` block, a matching
    ``nvdspreprocess`` config file is written as well and the entry is annotated
    with a ``preprocess_config`` key so the builder can wire the element in.

    The processed result is exposed through :meth:`get_config`, which the
    pipeline builders consume.
    """

    def __init__(self, file_name: str, temp_directory: str) -> None:
        """Load and process the application JSON config.

        Args:
            file_name: Path to the JSON application config file.
            temp_directory: Directory where generated inference-engine config
                files are written.
        """
        self.temp_directory: str = temp_directory
        self.config: Dict[str, Any] = {}
        data: Optional[Dict[str, Any]] = None
        with open(file_name) as json_file:
            data = json.load(json_file)

        assert data is not None
        self.config['inference_backend'] = data.get('inference_backend', 'nvinfer')
        self.config['triton'] = data.get('triton', {})
        self.config['inference_engines'] = []
        for inference_engine in data['inference_engines']:
            if self.config['inference_backend'] == 'triton':
                self.create_triton_inference_config_file(inference_engine)
            else:
                self.create_inference_config_file(inference_engine)
        self.config['video_source'] = data['video_source']
        self.config['sink_type'] = data['sink_type']
        self.config['tracker'] = data['tracker']
        self.config['va_filters'] = data['va_filters']

    def get_config(self) -> Dict[str, Any]:
        """Return the processed configuration consumed by the pipeline builders.

        Returns:
            The processed config dict with ``inference_engines``,
            ``video_source``, ``sink_type``, ``tracker``, ``va_filters``,
            ``inference_backend`` and ``triton`` keys.
        """
        return self.config

    def create_inference_config_file(self, inference_engine_config: Dict[str, Any]) -> None:
        """Write ``nvinfer`` config files for a single inference engine entry.

        Writes an ``nvinfer`` config file derived from the entry's ``property``
        (and optional ``class-attrs-all``) block. If the entry has a
        ``preprocess`` block, also writes an ``nvdspreprocess`` config file and
        records its path under the ``preprocess_config`` key of the resulting
        engine entry.

        Args:
            inference_engine_config: A dict describing a single engine. Must have
                a ``name`` key; ``property`` is required to produce a config file.
        """
        name: str = inference_engine_config['name']
        file_name: str = os.path.join(self.temp_directory, name + 'config.txt')

        if 'property' not in inference_engine_config:
            print(file_name + " could not be created due to missing data in configuration file")
            return

        sections: List[Tuple[str, Dict[str, Any]]] = [('property', inference_engine_config['property'])]
        if 'class-attrs-all' in inference_engine_config:
            sections.append(('class-attrs-all', inference_engine_config['class-attrs-all']))
        self._write_config_file(file_name, sections)

        inference_engine: Dict[str, Any] = {
            "name": name,
            "spec_file": file_name,
            "batch_size": inference_engine_config['property'].get('batch-size', 1)
        }
        self._finalize_engine_entry(name, inference_engine_config, inference_engine)
        print(name)

    def create_triton_inference_config_file(self, inference_engine_config: Dict[str, Any]) -> None:
        """Write an ``nvinferserver`` config file for a single engine entry.

        Derives the config from the same ``property`` (and ``class-attrs-all``)
        data the nvinfer path uses, rendered in the DeepStream-8 protobuf format
        that ``gst-nvinferserver`` expects (an ``infer_config { ... }`` block
        plus ``input_control { ... }``). The server connection settings come
        from the top-level ``triton`` block.

        Model-file keys (``onnx-file``, ``tlt-encoded-model``,
        ``model-engine-file``, ``int8-calib-file``) are dropped -- the Triton
        server loads those from its own model repository.

        Args:
            inference_engine_config: A dict describing a single engine, with at
                least a ``name`` key and a ``property`` block.
        """
        name: str = inference_engine_config['name']
        file_name: str = os.path.join(self.temp_directory, name + 'config.txt')

        if 'property' not in inference_engine_config:
            print(file_name + " could not be created due to missing data in configuration file")
            return

        contents: str = self._triton_infer_config_contents(inference_engine_config)
        with open(file_name, 'w') as f:
            f.write(contents)

        inference_engine: Dict[str, Any] = {
            "name": name,
            "spec_file": file_name,
            "batch_size": inference_engine_config['property'].get('batch-size', 1)
        }
        self._finalize_engine_entry(name, inference_engine_config, inference_engine)
        print(name)

    def _triton_infer_config_contents(self, inference_engine_config: Dict[str, Any]) -> str:
        """Render the DeepStream-8 ``nvinferserver`` config file for an engine."""
        triton_config: Dict[str, Any] = self.config.get('triton', {})
        prop: Dict[str, Any] = inference_engine_config['property']
        class_attrs: Dict[str, Any] = inference_engine_config.get('class-attrs-all', {})

        server_url: str = triton_config.get('server_url', 'localhost:8001')
        enable_cuda_share: bool = bool(triton_config.get('enable_cuda_buffer_sharing', True))
        use_http: bool = str(triton_config.get('protocol_type', 'grpc')).lower() == 'http'
        model_name: str = infer_triton_model_name(inference_engine_config)
        try:
            model_version: int = int(inference_engine_config.get('model_version', -1) or -1)
        except (TypeError, ValueError):
            model_version = -1

        lines: List[str] = ['infer_config {']
        lines.append('  unique_id: %s' % prop.get('gie-unique-id', 1))
        if 'gpu-id' in prop:
            lines.append('  gpu_ids: [%s]' % prop['gpu-id'])
        lines.append('  max_batch_size: %s' % prop.get('batch-size', 1))

        lines.append('  backend {')
        if prop.get('output-blob-names'):
            out_names = [n.strip() for n in str(prop['output-blob-names']).split(';') if n.strip()]
            lines.append('    outputs: [')
            for i, blob_name in enumerate(out_names):
                comma = ',' if i < len(out_names) - 1 else ''
                lines.append('      {name: "%s"}%s' % (blob_name, comma))
            lines.append('    ]')
        lines.append('    triton {')
        lines.append('      model_name: "%s"' % model_name)
        lines.append('      version: %d' % model_version)
        if use_http:
            lines.append('      http {')
            lines.append('        url: "%s"' % server_url)
            lines.append('      }')
        else:
            lines.append('      grpc {')
            lines.append('        url: "%s"' % server_url)
            lines.append('        enable_cuda_buffer_sharing: %s'
                         % ('true' if enable_cuda_share else 'false'))
            lines.append('      }')
        lines.append('    }')
        # Custom (CPU) postprocessors such as the NVPlate LPR parser read the
        # output tensors through host pointers, so request CPU output memory.
        lines.append('    output_mem_type: MEMORY_TYPE_CPU')
        lines.append('  }')

        custom_lib: Optional[str] = prop.get('custom-lib-path')
        if custom_lib and os.path.isfile(custom_lib):
            lines.append('  custom_lib {')
            lines.append('    path: "%s"' % custom_lib)
            lines.append('  }')
        elif custom_lib:
            print("Warning: custom-lib-path %s not found; skipping custom parser "
                  "for engine %s" % (custom_lib, inference_engine_config.get('name', '?')))

        lines.append('  preprocess {')
        color_format = prop.get('model-color-format', 0)
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
        if prop.get('labelfile-path'):
            lines.append('    labelfile_path: "%s"' % prop['labelfile-path'])
        lines.extend(self._triton_postprocess_lines(prop, class_attrs))
        lines.append('  }')
        lines.append('}')

        lines.append('input_control {')
        process_mode: int = int(prop.get('process-mode', 1))
        lines.append('  process_mode: %s'
                     % ('PROCESS_MODE_FULL_FRAME' if process_mode == 1 else 'PROCESS_MODE_CLIP_OBJECTS'))
        lines.append('  operate_on_gie_id: %s'
                     % self._triton_signed_int(prop.get('operate-on-gie-id', -1)))
        if 'operate-on-class-ids' in prop:
            class_ids = [x.strip() for x in str(prop['operate-on-class-ids']).replace(';', ',').split(',') if x.strip()]
            lines.append('  operate_on_class_ids: [%s]' % ', '.join(class_ids))
        if 'interval' in prop and process_mode == 1:
            lines.append('  interval: %s' % prop['interval'])
        if 'classifier-async-mode' in prop:
            lines.append('  async_mode: %s'
                         % ('true' if int(prop['classifier-async-mode']) == 1 else 'false'))
        if process_mode == 2 and ('input-object-min-width' in prop or 'input-object-min-height' in prop):
            lines.append('  object_control {')
            lines.append('    bbox_filter {')
            if 'input-object-min-width' in prop:
                lines.append('      min_width: %s' % prop['input-object-min-width'])
            if 'input-object-min-height' in prop:
                lines.append('      min_height: %s' % prop['input-object-min-height'])
            lines.append('    }')
            lines.append('  }')
        lines.append('}')
        return '\n'.join(lines) + '\n'

    @staticmethod
    def _triton_signed_int(value: Any) -> int:
        """Return ``value`` as a non-negative int, or -1 when absent/invalid."""
        try:
            v = int(value)
            return v if v >= 0 else -1
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _triton_postprocess_lines(prop: Dict[str, Any],
                                  class_attrs: Dict[str, Any]) -> List[str]:
        """Render the ``postprocess`` block (detection or classification)."""
        lines: List[str] = []
        is_classifier: bool = int(prop.get('is-classifier', 0)) == 1 or int(prop.get('network-type', 0)) == 1
        if is_classifier:
            threshold = prop.get('classifier-threshold')
            if threshold is None and class_attrs:
                threshold = class_attrs.get('threshold')
            if threshold is not None:
                lines.append('    classification {')
                lines.append('      threshold: %s' % threshold)
                custom_func = prop.get('parse-classifier-func-name')
                if custom_func and prop.get('custom-lib-path') and os.path.isfile(prop['custom-lib-path']):
                    lines.append('      custom_parse_classifier_func: "%s"' % custom_func)
                lines.append('    }')
        elif 'num-detected-classes' in prop:
            lines.append('    detection {')
            lines.append('      num_detected_classes: %s' % prop['num-detected-classes'])
            pre_threshold = class_attrs.get('pre-cluster-threshold')
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
            if custom_func and prop.get('custom-lib-path') and os.path.isfile(prop['custom-lib-path']):
                lines.append('      custom_parse_bbox_func: "%s"' % custom_func)
            lines.append('    }')
        return lines

    def _finalize_engine_entry(self,
                               name: str,
                               inference_engine_config: Dict[str, Any],
                               inference_engine: Dict[str, Any]) -> None:
        """Append a processed engine entry, adding any preprocess config.

        If the engine entry declares a ``preprocess`` block, writes the
        ``nvdspreprocess`` config file and records its path on the entry under
        the ``preprocess_config`` key, then appends the entry to the processed
        ``inference_engines`` list.

        Args:
            name: The engine name, used for the preprocess file name.
            inference_engine_config: The raw engine entry from the JSON config.
            inference_engine: The processed engine entry dict to append.
        """
        if 'preprocess' in inference_engine_config:
            preprocess_file_name: str = os.path.join(self.temp_directory, name + 'preprocess.txt')
            preprocess_sections: List[Tuple[str, Dict[str, Any]]] = [
                (section_name, section)
                for section_name, section in inference_engine_config['preprocess'].items()
            ]
            self._write_config_file(preprocess_file_name, preprocess_sections)
            inference_engine["preprocess_config"] = preprocess_file_name
        self.config['inference_engines'].append(inference_engine)

    def _write_config_file(self, file_name: str, sections: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Write a DeepStream textual config file from named sections.

        Each section is rendered as ``[section_name]`` followed by one
        ``key=value`` line per entry, separated by blank lines.

        Args:
            file_name: Path of the file to write.
            sections: Ordered list of ``(section_name, options)`` where ``options``
                is a dict mapping each config key to its value.
        """
        with open(file_name, 'w') as f:
            for section_name, section in sections:
                f.write('[' + section_name + '] \n')
                for key in section:
                    f.write(key + "=" + str(section[key]) + '\n')
                f.write('\n')