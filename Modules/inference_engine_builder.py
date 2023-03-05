
import sys
from gi.repository import GLib, Gst

class InferenceEngineBuilder:
    def __init__(self, pipeline, streammux, tracker, config):
        self.pipeline = pipeline
        self.inference_engines = []
        assert len(config['inference_engines']) > 0
        for inference_engine_config in config['inference_engines']:
            self.inference_engines.append(self.create_inference(inference_engine_config['name'],
                                                                inference_engine_config['spec_file']))
        self.link_inference_engines(streammux, tracker)

    def get_last_inference_engine(self):
        assert len(self.inference_engines) > 0
        return self.inference_engines[-1]

    def create_inference(self, model_name, model_config_file):
        print("create inference")
        print("model_name:" + model_name)
        print("model_config_file:" + model_config_file)
        # Use nvinfer to run inferencing on decoder's output,
        # behaviour of inferencing is set through config file
        inference_engine = Gst.ElementFactory.make("nvinfer", model_name)
        if not inference_engine:
            sys.stderr.write(" Unable to create model engine " + model_name + "\n")

        inference_engine.set_property('config-file-path', model_config_file)
        self.pipeline.add(inference_engine)

        return inference_engine

    def link_inference_engines(self, streammux, tracker):
        first_model = True
        for index in range(len(self.inference_engines)):
            if first_model:
                first_model = False
                streammux.link(self.inference_engines[index])
                print("link streammux to inference engine " + str(index))
                if tracker is not None:
                    self.inference_engines[index].link(tracker)
                    print("link inference engine " + str(index) + " to tracker engine")
            else:
                if (tracker is not None) and (index -1 == 0):
                    tracker.link(self.inference_engines[index])
                    print("link tracker engine to inference engine " + str(index))
                else:
                    self.inference_engines[index-1].link(self.inference_engines[index])
                    print("link inference engine " + str(index-1) + " to inference engine " + str(index))
