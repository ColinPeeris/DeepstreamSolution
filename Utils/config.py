import os
import json
from typing import Any, Dict, List, Optional, Tuple


class Config:
    """Parses a JSON application config and prepares DeepStream element configs.

    For every entry in the supplied JSON's ``inference_engines`` list the class
    writes a plain-text ``nvinfer`` config file (used via ``config-file-path``).
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
            temp_directory: Directory where generated nvinfer/nvdspreprocess
                config files are written.
        """
        self.temp_directory: str = temp_directory
        self.config: Dict[str, Any] = {}
        data: Optional[Dict[str, Any]] = None
        with open(file_name) as json_file:
            data = json.load(json_file)

        assert data is not None
        self.config['inference_engines'] = []
        for inference_engine in data['inference_engines']:
            self.create_inference_config_file(inference_engine)
        self.config['video_source'] = data['video_source']
        self.config['sink_type'] = data['sink_type']
        self.config['tracker'] = data['tracker']
        self.config['va_filters'] = data['va_filters']

    def get_config(self) -> Dict[str, Any]:
        """Return the processed configuration consumed by the pipeline builders.

        Returns:
            The processed config dict with ``inference_engines``, ``video_source``,
            ``sink_type``, ``tracker`` and ``va_filters`` keys.
        """
        return self.config

    def create_inference_config_file(self, inference_engine_config: Dict[str, Any]) -> None:
        """Write config files for a single inference engine entry.

        Writes an ``nvinfer`` config file derived from the entry's ``property``
        (and optional ``class-attrs-all``) block. If the entry has a ``preprocess``
        block, also writes an ``nvdspreprocess`` config file and records its path
        under the ``preprocess_config`` key of the resulting engine entry.

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
            "spec_file": file_name
        }

        if 'preprocess' in inference_engine_config:
            preprocess_file_name: str = os.path.join(self.temp_directory, name + 'preprocess.txt')
            preprocess_sections: List[Tuple[str, Dict[str, Any]]] = [
                (section_name, section)
                for section_name, section in inference_engine_config['preprocess'].items()
            ]
            self._write_config_file(preprocess_file_name, preprocess_sections)
            inference_engine["preprocess_config"] = preprocess_file_name

        self.config['inference_engines'].append(inference_engine)
        print(name)

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
