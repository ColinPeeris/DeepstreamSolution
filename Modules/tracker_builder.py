import sys
from gi.repository import Gst


class TrackerBuilder:
    def __init__(self, pipeline, config):
        self.tracker = None
        self.pipeline = pipeline
        self.create_tracker()
        self.set_tracker_properties(config)

    def get_tracker(self):
        return self.tracker

    def create_tracker(self):
        self.tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not self.tracker:
            sys.stderr.write(" Unable to create tracker \n")
        self.pipeline.add(self.tracker)

    def set_tracker_properties(self, config):
        # Set properties of tracker
        for key in config['tracker']:
            if key == 'tracker-width':
                tracker_width = config['tracker'][key]
                self.tracker.set_property('tracker-width', tracker_width)
            if key == 'tracker-height':
                tracker_height = config['tracker'][key]
                self.tracker.set_property('tracker-height', tracker_height)
            if key == 'gpu-id':
                tracker_gpu_id = config['tracker'][key]
                self.tracker.set_property('gpu_id', tracker_gpu_id)
            if key == 'll-lib-file':
                tracker_ll_lib_file = config['tracker'][key]
                self.tracker.set_property('ll-lib-file', tracker_ll_lib_file)
            if key == 'll-config-file':
                tracker_ll_config_file = config['tracker'][key]
                self.tracker.set_property('ll-config-file', tracker_ll_config_file)
            if key == 'enable-batch-process':
                tracker_enable_batch_process = config['tracker'][key]
                self.tracker.set_property('enable_batch_process', tracker_enable_batch_process)
            if key == 'enable-past-frame':
                tracker_enable_past_frame = config['tracker'][key]
                self.tracker.set_property('enable_past_frame', tracker_enable_past_frame)
