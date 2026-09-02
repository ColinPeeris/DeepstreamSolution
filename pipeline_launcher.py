#!/usr/bin/env python3

import sys
from Modules.pipeline_builder import PipelineBuilder


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python pipeline_launcher.py <config_file.json>\n")
        return 1
    pipeline = PipelineBuilder(config_file_name=sys.argv[1])
    pipeline.run()


if __name__ == '__main__':
    sys.exit(main())
