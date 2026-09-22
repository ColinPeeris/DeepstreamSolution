import sys
from typing import Dict, List, Optional

from gi.repository import Gst

from Utils.timing import InferenceTimingMonitor

# A single engine's entry in the processed config's ``inference_engines`` list.
TritonInferenceEngineConfig = Dict


class TritonInferenceEngineBuilder:
    """Builds and links inference stages that run on a Triton Inference Server.

    Mirrors :class:`Modules.inference_engine_builder.InferenceEngineBuilder`
    but creates ``nvinferserver`` elements instead of locally-running ``nvinfer``
    elements. Each entry in the config's ``inference_engines`` list becomes an
    ``nvinferserver`` element driven by a generated config file (the
    DeepStream-8 ``infer_config``/``input_control`` protobuf format) that points
    at the Triton server (gRPC by default) and names the served model. An
    engine may also declare an optional ``preprocess`` block, in which case a
    ``nvdspreprocess`` element is inserted ahead of that engine.

    The resulting chain is::

        streammux -> engine[0] -> [tracker] -> [preprocess] -> engine[1]
                  -> [preprocess] -> engine[2] -> ...

    Because ``nvinferserver`` does not expose ``batch-size`` as a GStreamer
    property, the configured batch size is tracked from the config and handed to
    :class:`Utils.timing.InferenceTimingMonitor` so the timing summary stays
    accurate.
    """

    def __init__(self,
                 pipeline: Gst.Pipeline,
                 streammux: Gst.Element,
                 tracker: Optional[Gst.Element],
                 config: Dict) -> None:
        """Create the Triton inference elements and link them into the pipeline.

        Args:
            pipeline: The parent ``Gst.Pipeline`` the elements are added to.
            streammux: The ``nvstreammux`` element feeding the first engine.
            tracker: An optional ``nvtracker`` element, or ``None``.
            config: The application config dict; must contain a non-empty
                ``inference_engines`` list.
        """
        self.pipeline = pipeline
        self.inference_engines: List[Gst.Element] = []
        self.preprocess_engines: List[Optional[Gst.Element]] = []
        self.batch_sizes: List[int] = []
        self.inference_timing_monitor = InferenceTimingMonitor()
        self.chain_tail: Optional[Gst.Element] = None
        assert len(config['inference_engines']) > 0
        for inference_engine_config in config['inference_engines']:
            self.create_inference(inference_engine_config)
        self.link_inference_engines(streammux, tracker)

    def get_last_inference_engine(self) -> Gst.Element:
        """Return the last element of the inference/tracker chain.

        This is the final nvinferserver if any engine follows the tracker,
        else the tracker itself (the sink stage links from this element).

        Returns:
            The last ``Gst.Element`` in the chain.
        """
        assert self.chain_tail is not None
        return self.chain_tail

    def attach_inference_timing(self) -> None:
        """Attach per-model inference timing probes to every nvinferserver element.

        Delegates to :class:`Utils.timing.InferenceTimingMonitor`, passing the
        models' configured batch sizes since ``nvinferserver`` does not expose
        them as GStreamer properties.
        """
        self.inference_timing_monitor.attach(self.inference_engines, self.batch_sizes)

    def print_inference_timing(self) -> None:
        """Print a per-model inference timing summary at the end of the run.

        Delegates to :class:`Utils.timing.InferenceTimingMonitor`. Call after
        the pipeline has finished (End-of-stream).
        """
        self.inference_timing_monitor.print_summary()

    def create_inference(self, inference_engine_config: TritonInferenceEngineConfig) -> None:
        """Create the nvinferserver (and optional nvdspreprocess) elements.

        Args:
            inference_engine_config: A dict describing a single engine, with at
                least ``name`` and ``spec_file`` keys, and optionally a
                ``preprocess_config`` key.
        """
        model_name: str = inference_engine_config['name']
        model_config_file: str = inference_engine_config['spec_file']
        preprocess_config_file: Optional[str] = inference_engine_config.get('preprocess_config')
        self.batch_sizes.append(inference_engine_config.get('batch_size', 1))

        print("create triton inference")
        print("model_name:" + model_name)
        print("model_config_file:" + model_config_file)

        # Use nvdspreprocess to re-process object ROIs into a temporal sequence
        # tensor for sequence (action recognition) model inference.
        preprocess: Optional[Gst.Element] = None
        if preprocess_config_file is not None:
            preprocess = Gst.ElementFactory.make("nvdspreprocess", model_name + "_preprocess")
            if not preprocess:
                sys.stderr.write(" Unable to create nvdspreprocess for " + model_name + "\n")
            preprocess.set_property('config-file', preprocess_config_file)
            self.pipeline.add(preprocess)

        # Use nvinferserver to run inferencing on a remote Triton server.
        # Behaviour of inferencing is set through the generated config file.
        inference_engine = Gst.ElementFactory.make("nvinferserver", model_name)
        if not inference_engine:
            sys.stderr.write(" Unable to create nvinferserver for " + model_name + "\n")

        inference_engine.set_property('config-file-path', model_config_file)
        self.pipeline.add(inference_engine)

        self.inference_engines.append(inference_engine)
        self.preprocess_engines.append(preprocess)

    def link_inference_engines(self,
                               streammux: Gst.Element,
                               tracker: Optional[Gst.Element]) -> None:
        """Link the streammux, tracker, preprocess and nvinferserver elements.

        The first engine links from the streammux, then (if present) to the
        tracker. Every subsequent engine links from the previous element,
        inserting its associated ``nvdspreprocess`` element in between when one
        was created.

        Args:
            streammux: The ``nvstreammux`` element to start the chain from.
            tracker: An optional ``nvtracker`` element, or ``None``.
        """
        first: Gst.Element = self.inference_engines[0]
        ret = streammux.link(first)
        print("link streammux to inference engine 0 -> " + str(ret))

        prev: Gst.Element = first
        if tracker is not None:
            ret = first.link(tracker)
            print("link inference engine 0 to tracker engine -> " + str(ret))
            prev = tracker

        for index in range(1, len(self.inference_engines)):
            preprocess = self.preprocess_engines[index]
            inference_engine = self.inference_engines[index]
            if preprocess is not None:
                ret = prev.link(preprocess)
                print("link to preprocess engine " + str(index) + " -> " + str(ret))
                prev = preprocess
            ret = prev.link(inference_engine)
            print("link to inference engine " + str(index) + " -> " + str(ret))
            prev = inference_engine
        self.chain_tail = prev