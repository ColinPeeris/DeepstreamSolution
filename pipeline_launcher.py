#!/usr/bin/env python3

import sys
from Modules.pipeline_builder import PipelineBuilder


def main():
    pipeline = PipelineBuilder()
    pipeline.run()


if __name__ == '__main__':
    sys.exit(main())
