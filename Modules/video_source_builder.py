
import sys
from typing import List

from gi.repository import Gst


class VideoSourceBuilder():
    def __init__(self, pipeline, config):
        self.streammux = None
        vcfg = config['video_source']
        # The pipeline can ingest several videos at once (e.g. one per camera).
        # ``filenames`` takes a list of paths; ``filename`` is kept for
        # single-stream configs and backwards compatibility.
        if 'filenames' in vcfg:
            filenames: List[str] = [str(name) for name in vcfg['filenames']]
        elif 'filename' in vcfg:
            filenames = [str(vcfg['filename'])]
        else:
            raise ValueError("video_source must define 'filename' or 'filenames'")
        if not filenames:
            raise ValueError("video_source.filenames is empty")

        # For multi-stream pipelines, batch as many frames as the number of
        # streams allows (capped at 16) so the GIE models' configured batch
        # size is actually exercised; single-stream runs keep batch-size 1 to
        # match the original behaviour.
        streammux_batch_size: int = int(vcfg.get(
            'streammux_batch_size', min(len(filenames), 16)))
        self.create_sources(pipeline, filenames=filenames,
                            streammux_batch_size=streammux_batch_size)

    def get_stream_mux(self):
        return self.streammux

    def create_sources(self, pipeline, filenames, streammux_batch_size):
        # Create nvstreammux instance to form batches from one or more sources.
        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not self.streammux:
            sys.stderr.write(" Unable to create NvStreamMux \n")
        pipeline.add(self.streammux)

        for stream_index, filename in enumerate(filenames):
            # Source element for reading from the file.
            print("Creating Source %d \n" % stream_index)
            source = Gst.ElementFactory.make("filesrc", "file-source-%d" % stream_index)
            if not source:
                sys.stderr.write(" Unable to create Source \n")

            # Since the data format in the input file is elementary h264 stream,
            # we need a h264parser.
            print("Creating H264Parser \n")
            h264parser = Gst.ElementFactory.make("h264parse", "h264-parser-%d" % stream_index)
            if not h264parser:
                sys.stderr.write(" Unable to create h264 parser \n")

            # Use nvdec_h264 for hardware accelerated decode on GPU.
            print("Creating Decoder \n")
            decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder-%d" % stream_index)
            if not decoder:
                sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")

            pipeline.add(source)
            pipeline.add(h264parser)
            pipeline.add(decoder)

            print("Playing file %s \n" % filename)
            source.set_property('location', filename)
            source.link(h264parser)
            h264parser.link(decoder)

            sinkpad = self.streammux.get_request_pad("sink_%d" % stream_index)
            if not sinkpad:
                sys.stderr.write(" Unable to get the sink pad of streammux \n")
            srcpad = decoder.get_static_pad("src")
            if not srcpad:
                sys.stderr.write(" Unable to get source pad of decoder \n")
            srcpad.link(sinkpad)

        print("Playing %d stream(s)" % len(filenames))
        self.streammux.set_property('width', 1920)
        self.streammux.set_property('height', 1080)
        self.streammux.set_property('batch-size', streammux_batch_size)
        self.streammux.set_property('batched-push-timeout', 4000000)
