
import sys
import pyds
import time
import pika     # pip install pika-1.3.1
import json     # pip install jsonlib-python3-1.6.1
from typing import Dict
from Modules.VAFilters.VehicleMonitorFilter import VehicleMonitorFilter
from Modules.VAFilters.ActionMonitorFilter import ActionMonitorFilter
from Modules.Utilities.roi import Point

from gi.repository import Gst
start = time.time()

available_va_filters = {
    'VehicleMonitorFilter': VehicleMonitorFilter(),
    'ActionMonitorFilter': ActionMonitorFilter()
}


class ObjInfo:
    def __init__(self, left, top, width, height, trackerID, objectLabel, classifierLabels):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.trackerID = trackerID
        self.objectLabel = objectLabel
        self.classifierLabels = classifierLabels

    def get_center_point(self) -> Point:
        return Point(x=self.left + self.width / 2, y=self.top + self.height / 2)

    def get_object_label(self):
        return self.objectLabel

    def get_object_classification(self):
        return self.classifierLabels

    def get_trackerID(self):
        return self.trackerID


class FrameMetadata:
    def __init__(self, frame, frame_number):
        self.objList = []
        self.frame = frame
        self.frame_number = frame_number
        self.va_output = {}     # this should be a dictionary: {'filter_name' : 'message_string'}

    def add_object(self, objInfo: ObjInfo):
        self.objList.append(objInfo)

    def get_object_list(self):
        return self.objList

    def get_frame_resolution(self):
        # returns heigth, width
        return self.frame.shape[0], self.frame.shape[1]

    def get_frame(self):
        return self.frame

    def convert_dict_to_msg_string(self, metadata: Dict):
        return json.dumps(metadata)

    def convert_msg_string_to_dict(self, msg_string: str):
        return json.loads(msg_string)

    def set_va_output(self, filter_name, filter_metadata: Dict):
        self.va_output[filter_name] = self.convert_dict_to_msg_string(filter_metadata)

    def get_va_output(self):
        return self.va_output


class VAFilterBuilder:
    def __init__(self, pipeline_sink_pad, config):
        self.va_filter_list = []
        self.create_VAFilters(pipeline_sink_pad, config)

    def convert_msg_string_to_dict(self, msg_string: str):
        return json.loads(msg_string)

    def get_classifier_data(self, obj_meta):
        classifier_data = []
        # Only vehicle supports secondary inference
        cls_meta = obj_meta.classifier_meta_list
        while cls_meta is not None:
            cls = pyds.NvDsClassifierMeta.cast(cls_meta.data)
            # type of pyds.GList
            info = cls.label_info_list
            while info is not None:
                label_meta = pyds.glist_get_nvds_label_info(info.data)
                classifier_data.append(label_meta.result_label)
                try:
                    info = info.next
                except StopIteration:
                    break
            try:
                cls_meta = cls_meta.next
            except StopIteration:
                break
        return classifier_data

    def va_filter_probe(self, pad, info, u_data):
        frame_number = 0
        # Intiallizing object counter with 0.
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return

        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting is done by pyds.NvDsFrameMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_number = frame_meta.frame_num
            frame_metadata = FrameMetadata(frame=pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id),
                                           frame_number=frame_number)
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                objInfo = ObjInfo(top=obj_meta.rect_params.top,
                                  left=obj_meta.rect_params.left,
                                  width=obj_meta.rect_params.width,
                                  height=obj_meta.rect_params.height,
                                  trackerID=obj_meta.object_id,
                                  objectLabel=obj_meta.obj_label,
                                  classifierLabels=self.get_classifier_data(obj_meta=obj_meta))
                frame_metadata.add_object(objInfo)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # Acquiring a display meta object. The memory ownership remains in
            # the C code so downstream plugins can still access it. Otherwise
            # the garbage collector will claim it when this probe function exits.
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            py_nvosd_text_params = display_meta.text_params[0]

            for va_filter in self.va_filter_list:
                va_filter.run_filter(frame_metadata)

            # get va output
            va_output = frame_metadata.get_va_output()
            if len(va_output) > 0:
                self.send_metadata(va_output)

            '''for filter_name in va_output:
                print(filter_name)
                metadata = self.convert_msg_string_to_dict(va_output[filter_name])
                for key in metadata:
                    print(metadata[key])'''

            # Now set the offsets where the string should appear
            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 12

            # Font , font-color and font-size
            py_nvosd_text_params.font_params.font_name = "Serif"
            py_nvosd_text_params.font_params.font_size = 10
            # set(red, green, blue, alpha); set to White
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

            # Text background color
            py_nvosd_text_params.set_bg_clr = 1
            # set(red, green, blue, alpha); set to Black
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
            # Using pyds.get_string() to get display_text as string
            # print(pyds.get_string(py_nvosd_text_params.display_text))
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        # past traking meta data
        return Gst.PadProbeReturn.OK

    def send_metadata(self, metadata: Dict):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()

        channel.queue_declare(queue='deepstreamSolution')

        json_object = json.dumps(metadata, indent=4)
        channel.basic_publish(exchange='', routing_key='deepstreamSolution', body=json_object)

        connection.close()

    def create_VAFilters(self, osdsinkpad, config):
        # Lets add probe to get informed of the meta data generated, we add probe to
        # the sink pad of the osd element, since by that time, the buffer would have
        # had got all the metadata.

        if not osdsinkpad:
            sys.stderr.write(" Unable to get sink pad of nvosd \n")

        for va_filter_name in config['va_filters']:
            assert va_filter_name in available_va_filters, \
                f'{va_filter_name} is not in: {", ".join(list(available_va_filters.keys()))}'
            self.va_filter_list.append(available_va_filters[va_filter_name])

        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.va_filter_probe, 0)
