
import sys
import pyds
from gi.repository import Gst


class PipelineSinkBuilder:
    def __init__(self, pipeline, inference_engine, config):
        self.pipeline = pipeline
        self.osdsinkpad = None
        '''if config['sink_type'] == 'filesink':
            self.create_file_Sink(inference_engine)
        else:
            self.create_renderer_Sink(inference_engine)'''
        self.create_sink(inference_engine, config)

    def get_pipeline_sink_pad(self):
        return self.osdsinkpad

    def create_sink(self, inference_engine, config):    # the final one!
        # Use convertor to convert from NV12 to RGBA as required by nvosd
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not nvvidconv:
            sys.stderr.write(" Unable to create nvvidconv \n")

        # Create OSD to draw on the converted RGBA buffer
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not nvosd:
            sys.stderr.write(" Unable to create nvosd \n")

        tee = Gst.ElementFactory.make("tee", "nvsink-tee")
        if not tee:
            sys.stderr.write(" Unable to create tee \n")

        # changing the memory management to cuda_unified.
        # NVIDIA mentions this in the documentation for pyds.get_nvds_buf_surface.
        # We add this so that we can use the get_nvds_buf_surface call
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)

        self.pipeline.add(nvvidconv)
        self.pipeline.add(nvosd)
        self.pipeline.add(tee)

        inference_engine.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(tee)

        # if (msgconv is not None) and (msgbroker is not None):
        #    queue_msg = self.link_tee_to_queue(tee, "nvtee-que1")
        #    queue_msg.link(msgconv)
        #    msgconv.link(msgbroker)

        if config['sink_type'] == 'filesink':
            queue_file_source = self.link_tee_to_queue(tee, "queue_file_source")
            self.add_file_sink(queue_file_source)
        else:
            queue_fake_sink = self.link_tee_to_queue(tee, "queue_fake_sink")
            self.add_fake_sink(queue_fake_sink)

        self.osdsinkpad = nvosd.get_static_pad("sink")

    def link_tee_to_queue(self, tee, queue_name):
        print("Creating Queue for file sink \n")
        queue = Gst.ElementFactory.make("queue", queue_name)
        if not queue:
            sys.stderr.write(" Unable to create " + queue_name + " \n")
        self.pipeline.add(queue)

        print("Link Tee to Queue \n")
        queue_pad = queue.get_static_pad("sink")
        tee_pad = tee.get_request_pad('src_%u')
        if not tee_pad:
            sys.stderr.write("Unable to get request pads for file source \n")
        tee_pad.link(queue_pad)

        return queue

    def add_fake_sink(self, queue_fake_sink):
        print("Creating FakeSink \n")
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
        if not sink:
            sys.stderr.write(" Unable to create fakesink \n")

        self.pipeline.add(sink)
        queue_fake_sink.link(sink)

    def add_file_sink(self, queue_file_src):
        # Create another Converter
        nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
        if not nvvidconv2:
            sys.stderr.write(" Unable to create nvvidconv2 \n")
        # Create Capsfilter
        capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        if not capsfilter:
            sys.stderr.write(" Unable to create capsfilter \n")
        caps = Gst.Caps.from_string("video/x-raw, format=I420")
        capsfilter.set_property("caps", caps)
        # Create Encoder
        encoder = Gst.ElementFactory.make("avenc_mpeg4", "encoder")
        if not encoder:
            sys.stderr.write(" Unable to create encoder \n")
        encoder.set_property("bitrate", 2000000)
        # Create Code Parser
        codeparser = Gst.ElementFactory.make("mpeg4videoparse", "mpeg4-parser")
        if not codeparser:
            sys.stderr.write(" Unable to create code parser \n")
        # Create Container
        container = Gst.ElementFactory.make("qtmux", "qtmux")
        if not container:
            sys.stderr.write(" Unable to container \n")
        # Create Sink
        sink = Gst.ElementFactory.make('filesink', 'filesink')
        if not sink:
            sys.stderr.write(" Unable to create file sink \n")
        sink.set_property('location', './output.mp4')
        sink.set_property("sync", 1)
        sink.set_property("async", 0)

        self.pipeline.add(nvvidconv2)
        self.pipeline.add(capsfilter)
        self.pipeline.add(encoder)
        self.pipeline.add(codeparser)
        self.pipeline.add(container)
        self.pipeline.add(sink)
        queue_file_src.link(nvvidconv2)
        nvvidconv2.link(capsfilter)
        capsfilter.link(encoder)
        encoder.link(codeparser)
        codeparser.link(container)
        container.link(sink)
