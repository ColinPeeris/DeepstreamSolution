
import sys
from gi.repository import GLib, Gst

class VideoSourceBuilder():
    def __init__(self, pipeline, config):
        self.streammux = None
        filename = config['video_source']['filename']
        streammux = self.create_source(pipeline, filename=filename)

    def get_stream_mux(self):
        return self.streammux

    def create_source(self, pipeline, filename):
        # Source element for reading from the file
        print("Creating Source \n ")
        source = Gst.ElementFactory.make("filesrc", "file-source")
        if not source:
            sys.stderr.write(" Unable to create Source \n")

        # Since the data format in the input file is elementary h264 stream,
        # we need a h264parser
        print("Creating H264Parser \n")
        h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        if not h264parser:
            sys.stderr.write(" Unable to create h264 parser \n")

        # Use nvdec_h264 for hardware accelerated decode on GPU
        print("Creating Decoder \n")
        decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
        if not decoder:
            sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")

        # Create nvstreammux instance to form batches from one or more sources.
        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not self.streammux:
            sys.stderr.write(" Unable to create NvStreamMux \n")

        pipeline.add(source)
        pipeline.add(h264parser)
        pipeline.add(decoder)
        pipeline.add(self.streammux)

        print("Playing file %s " % filename)
        source.set_property('location', filename)
        self.streammux.set_property('width', 1920)
        self.streammux.set_property('height', 1080)
        self.streammux.set_property('batch-size', 1)
        self.streammux.set_property('batched-push-timeout', 4000000)

        source.link(h264parser)
        h264parser.link(decoder)

        sinkpad = self.streammux.get_request_pad("sink_0")
        if not sinkpad:
            sys.stderr.write(" Unable to get the sink pad of streammux \n")
        srcpad = decoder.get_static_pad("src")
        if not srcpad:
            sys.stderr.write(" Unable to get source pad of decoder \n")
        srcpad.link(sinkpad)