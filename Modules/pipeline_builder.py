import sys
import gi
import tempfile
import shutil

from Modules.video_source_builder import VideoSourceBuilder
from Modules.inference_engine_builder import InferenceEngineBuilder
from Modules.triton_inference_engine_builder import TritonInferenceEngineBuilder
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
        temp_directory = tempfile.mkdtemp()
        config = Config(config_file_name, temp_directory)
        # Standard GStreamer initialization
        Gst.init(None)
        self.pipeline = self.create_pipline()

        video_source = VideoSourceBuilder(pipeline=self.pipeline, config=config.get_config())
        tracker = TrackerBuilder(pipeline=self.pipeline, config=config.get_config()).get_tracker()
        inference_engine_builder = TritonInferenceEngineBuilder \
            if config.get_config().get('inference_backend', 'nvinfer') == 'triton' \
            else InferenceEngineBuilder
        inference_engine = inference_engine_builder(pipeline=self.pipeline,
                                                    streammux=video_source.get_stream_mux(),
                                                    tracker=tracker,
                                                    config=config.get_config())
        inference_engine.attach_inference_timing()
        self.inference_engine = inference_engine
        pipeline_sink = PipelineSinkBuilder(pipeline=self.pipeline,
                                            inference_engine=inference_engine.get_last_inference_engine(),
                                            config=config.get_config())
        VAFilterBuilder(pipeline_sink_pad=pipeline_sink.get_pipeline_sink_pad(),
                        config=config.get_config())

        self.create_event_loop(self.pipeline)
        self.temp_directory = temp_directory

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
        # Report per-model inference timing now that the pipeline has finished.
        self.inference_engine.print_inference_timing()
        # Generated per-engine config files are no longer needed once the
        # pipeline stops (nvinferserver reads them at start time, so they must
        # live for the whole run).
        shutil.rmtree(self.temp_directory)
        # cleanup
        self.pipeline.set_state(Gst.State.NULL)
