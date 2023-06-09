
import time
import numpy as np
import cv2
from Modules.Utilities.roi import ROI
from Modules.Utilities.roi import Point
from typing import List

start = time.time()

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3


class Object:
    def __init__(self, objId):
        self.objId = objId
        self.insideROI = False
        self.alertTriggered = False
        self.roiEntry = None
        self.roiExit = None
        self.entryExitDetected = False


class VehicleHistoricalRecord:
    def __init__(self):
        self.objRecord = []

    def add_track_id(self, track_id):
        id_exists = False
        for i in range(len(self.objRecord)):
            if self.objRecord[i].objId == track_id:
                id_exists = True
                break

        if not id_exists:
            object = Object(track_id)
            self.objRecord.append(object)

    def objectEnteredROI(self, track_id, centerPoint: Point):
        for i in range(len(self.objRecord)):
            if self.objRecord[i].objId == track_id and not self.objRecord[i].insideROI:
                self.objRecord[i].insideROI = True
                self.objRecord[i].roiEntry = centerPoint
                break

    def triggerAlert(self, track_id, centerPoint: Point):
        for i in range(len(self.objRecord)):
            if self.objRecord[i].objId == track_id:
                if self.objRecord[i].insideROI and not self.objRecord[i].alertTriggered:
                    self.objRecord[i].insideROI = False
                    self.objRecord[i].roiExit = centerPoint
                    self.objRecord[i].alertTriggered = True
                    return True
        return False

    def getEntryAndExitPoints(self, track_id):
        for i in range(len(self.objRecord)):
            if self.objRecord[i].objId == track_id:
                return self.objRecord[i].roiEntry, self.objRecord[i].roiExit


class VehicleMonitorFilter:
    def __init__(self):
        self.filter_name = 'VehicleMonitorFilter'
        self.roi_defined = False
        self.vehicleHistoricalRecord = VehicleHistoricalRecord()

    def save_snapshot(self, file_name, frame, point: Point = None, points: List[Point] = None):
        frame_copy = np.array(frame, copy=True, order='C')
        frame_copy_cv = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
        frame_copy_cv = self.roi.drawROI(frame_copy_cv)

        if point is not None:
            frame_copy_cv = cv2.circle(frame_copy_cv, (int(point.x), int(point.y)), 2, (0, 0, 255), 2)

        if points is not None:
            for point in points:
                frame_copy_cv = cv2.circle(frame_copy_cv, (int(point.x), int(point.y)), 2, (0, 0, 255), 2)

        cv2.imwrite(file_name, frame_copy_cv)

    def defineROI(self, img_height, img_width):
        roi_points = [Point((0.4*img_width), 0), Point((0.6*img_width), 0),
                      Point((0.6*img_width), img_height), Point((0.4*img_width), img_height)]
        self.roi = ROI(roi_points)

    def run_filter(self, frame_metadata):
        if not self.roi_defined:
            img_height, img_width = frame_metadata.get_frame_resolution()
            self.defineROI(img_height, img_width)
            self.roi_defined = True

        for object in frame_metadata.get_object_list():
            object_id = object.get_trackerID()
            obj_label = object.get_object_label()
            self.vehicleHistoricalRecord.add_track_id(track_id=object_id)
            center_point = object.get_center_point()
            if self.roi.checkInside(center_point):
                self.vehicleHistoricalRecord.objectEnteredROI(track_id=object_id, centerPoint=center_point)
            else:
                if self.vehicleHistoricalRecord.triggerAlert(track_id=object_id, centerPoint=center_point):
                    entry_point, exit_point = self.vehicleHistoricalRecord.getEntryAndExitPoints(track_id=object_id)
                    entry_exit_points = []
                    entry_exit_points.append(entry_point)
                    entry_exit_points.append(exit_point)
                    horizontal_motion_dir, horizontal_motion = self.horizontal_motion_check(entry_exit_points)
                    if (horizontal_motion > 100):   # filter out cases where the horizotal motion is less than 100 pixels     # noqa: E501
                        metadata = {}
                        metadata['vehicleID'] = object_id
                        metadata['objLabel'] = obj_label
                        if horizontal_motion_dir == 1:
                            metadata['direction'] = "right moving"
                        elif horizontal_motion_dir == -1:
                            metadata['direction'] = "left moving"
                        else:
                            metadata['direction'] = "not moving"
                        # metadata['snapshot'] = self.take_snapshot(gst_buffer = gst_buffer,
                        #                                           frame_meta = frame_meta,
                        #                                           points = entry_exit_points)
                        print(metadata)
                        if object.get_object_label() == 'Car':
                            if object.get_object_classification() != []:
                                print(object.get_object_classification())
                        # self.save_snapshot(file_name = "ROIentryexit_frame_" +
                        #                                str(frame_metadata.frame_number) + "_" +
                        #                                str(object_id) + ".jpg",
                        #                    frame = frame_metadata.get_frame(), points = entry_exit_points)
                        # self.send_metadata(metadata)
                        frame_metadata.set_va_output(self.filter_name, metadata)

    def horizontal_motion_check(self, entry_exit_pt):
        # this method outputs:
        # 1 if the object moves to the right
        # -1 if the object moves to the left and
        # 0 if there is no horizontal motion
        assert len(entry_exit_pt) == 2
        horizontal_motion = entry_exit_pt[1].x - entry_exit_pt[0].x
        if (horizontal_motion) > 0:
            return 1, abs(horizontal_motion)
        elif (horizontal_motion) < 0:
            return -1, abs(horizontal_motion)
        else:
            return 0, 0
