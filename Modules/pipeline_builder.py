import sys
import gi
import tempfile
import shutil

from Modules.video_source_builder import VideoSourceBuilder
from Modules.inference_engine_builder import InferenceEngineBuilder
from Modules.pipeline_sink_builder import PipelineSinkBuilder
from Modules.va_filter_builder import VAFilterBuilder
from Modules.tracker_builder import TrackerBuilder
from Utils.config import Config

sys.path.append('/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/apps/')
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst                                     # noqa: E402
from common.bus_call import bus_call                                    # noqa: E402


class PipelineBuilder:
    def __init__(self, config_file_name):
        temp_directory = tempfile.mkdtemp(dir=".")
        config = Config(config_file_name, temp_directory)
        # Standard GStreamer initialization
        Gst.init(None)
        self.pipeline = self.create_pipline()

        video_source = VideoSourceBuilder(pipeline=self.pipeline, config=config.get_config())
        tracker = TrackerBuilder(pipeline=self.pipeline, config=config.get_config()).get_tracker()
        inference_engine = InferenceEngineBuilder(pipeline=self.pipeline,
                                                  streammux=video_source.get_stream_mux(),
                                                  tracker=tracker,
                                                  config=config.get_config())
        pipeline_sink = PipelineSinkBuilder(pipeline=self.pipeline,
                                            inference_engine=inference_engine.get_last_inference_engine(),
                                            config=config.get_config())
        VAFilterBuilder(pipeline_sink_pad=pipeline_sink.get_pipeline_sink_pad(), config=config.get_config())

        self.create_event_loop(self.pipeline)
        shutil.rmtree(temp_directory)

    def create_pipline(self):
        # Create gstreamer elements
        # Create Pipeline element that will form a connection of other elements
        print("Creating Pipeline \n ")
        pipeline = Gst.Pipeline()

        if not pipeline:
            sys.stderr.write(" Unable to create Pipeline \n")

        return pipeline

    def create_event_loop(self, pipeline):
        # create an event loop and feed gstreamer bus mesages to it
        self.loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, self.loop)

    def run(self):
        # start play back and listen to events
        print("Starting pipeline \n")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except Exception:
            pass
        # cleanup
        self.pipeline.set_state(Gst.State.NULL)
