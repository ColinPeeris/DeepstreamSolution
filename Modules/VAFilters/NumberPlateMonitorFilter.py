"""License plate (numberplate) recognition monitor for the LPR pipeline stage.

This filter runs after the LPD (License Plate Detection) and LPR (License Plate
Recognition) secondary stages. LPD attaches child ``license_plate`` objects to
each detected vehicle, and LPR writes the recognized plate characters into that
plate object's classifier meta. For each vehicle the filter reads the plate text,
builds a per-vehicle plate history, and reports the current and most frequent
(read "settled") plate.
"""

from typing import Any, Dict, List, Optional

# ``gie-unique-id`` of the LPD (License Plate Detection) engine in the config.
# Plate child objects carry this component id. Kept in sync with
# ``configs/detector_tracker_classifier_actionRec_lpr_deepstream_8.json``.
LPD_UNIQUE_ID: int = 2

# Usual LPD class label for the detected plate region (from usa_lpd_label.txt).
LPD_PLATE_LABELS: List[str] = ['license_plate']


class VehiclePlateRecord:
    """Tracks the plate readings observed for a single tracked vehicle."""

    def __init__(self, vehicle_id: int) -> None:
        self.vehicle_id: int = vehicle_id
        self.plates: List[str] = []  # most recent plate texts, oldest first

    def record(self, plate: str, max_history: int = 10) -> None:
        """Append a plate observation for this vehicle."""
        self.plates.append(plate)
        if len(self.plates) > max_history:
            self.plates.pop(0)

    def current_plate(self) -> Optional[str]:
        """Return the most recent plate text, or ``None`` if none recorded."""
        if not self.plates:
            return None
        return self.plates[-1]

    def primary_plate(self) -> Optional[str]:
        """Return the most frequently observed plate text in the history."""
        if not self.plates:
            return None
        counts: Dict[str, int] = {}
        for plate in self.plates:
            counts[plate] = counts.get(plate, 0) + 1
        return max(counts, key=counts.get)


class NumberPlateMonitorFilter:
    """Reports recognized license plates per tracked vehicle.

    The LPR stage runs as a per-object secondary classifier over the LPD plate
    detections, so each plate child object carries the recognized characters in
    its classifier meta (readable through the object's ``classifierLabels``).
    This filter collects those readings per vehicle, reports the current and
    primary (most frequent) plate, and latches the settled plate for each
    vehicle.
    """

    def __init__(self) -> None:
        self.filter_name: str = 'NumberPlateMonitorFilter'
        self.vehicleRecords: Dict[int, VehiclePlateRecord] = {}

    def _get_record(self, vehicle_id: int) -> VehiclePlateRecord:
        """Return the per-vehicle record for ``vehicle_id``, creating it if needed."""
        if vehicle_id not in self.vehicleRecords:
            self.vehicleRecords[vehicle_id] = VehiclePlateRecord(vehicle_id)
        return self.vehicleRecords[vehicle_id]

    @staticmethod
    def _is_plate_object(object_info: Any) -> bool:
        """Return ``True`` if ``object_info`` is an LPD plate child object.

        A plate object is recognized either by its component (GIE) id matching
        the LPD engine, or by its label being one of the known LPD plate labels.

        Args:
            object_info: An :class:`~Modules.va_filter_builder.ObjInfo`.

        Returns:
            ``True`` if the object was produced by the LPD stage.
        """
        if object_info.get_component_id() == LPD_UNIQUE_ID:
            return True
        if object_info.get_object_label() in LPD_PLATE_LABELS:
            return True
        return False

    @staticmethod
    def _plate_text(classifier_labels: Optional[List[str]]) -> Optional[str]:
        """Reconstruct the plate string from an object's classifier labels.

        The LPR custom parser writes the decoded plate characters as label
        results, so the plate text is the concatenation of those labels.

        Args:
            classifier_labels: The plate object's classifier label list.

        Returns:
            The joined plate string, or ``None`` if there is no text.
        """
        for label in classifier_labels or []:
            if label and label.strip():
                return ''.join(label.strip().split())
        return None

    def run_filter(self, frame_metadata: "FrameMetadata") -> None:
        """Process one frame of metadata and report each vehicle's plate.

        The LPD stage attaches each plate as a child ``lpd`` object carrying the
        recognized characters in its classifier meta. Because the DeepStream
        metadata may not populate the child's parent link, the plate is assigned
        to a vehicle geometrically: the vehicle whose bounding box contains the
        plate's centre point. The plate text is recorded per associated vehicle
        and emitted (current plate, primary plate, plate history) via ``print``
        and :meth:`set_va_output`. Plate objects with no recognised text or no
        containing vehicle are skipped.

        Args:
            frame_metadata: The :class:`FrameMetadata` for the current frame.
        """
        vehicles: List[Any] = []
        plates: List[Any] = []
        for object_info in frame_metadata.get_object_list():
            if self._is_plate_object(object_info):
                plates.append(object_info)
            elif self._is_vehicle_object(object_info):
                vehicles.append(object_info)

        for plate_info in plates:
            plate: Optional[str] = self._plate_text(plate_info.get_object_classification())
            if plate is None:
                continue

            vehicle_info = self._containing_vehicle(plate_info, vehicles)
            if vehicle_info is None:
                continue

            vehicle_id: int = vehicle_info.get_trackerID()
            record = self._get_record(vehicle_id)
            record.record(plate)

            metadata: Dict[str, Any] = {
                'vehicleID': vehicle_id,
                'objLabel': vehicle_info.get_object_label(),
                'plate': record.current_plate(),
                'primary_plate': record.primary_plate(),
                'plate_history': list(record.plates),
            }

            print(metadata)
            frame_metadata.set_va_output(self.filter_name, metadata)

    @staticmethod
    def _is_vehicle_object(object_info: Any) -> bool:
        """Return ``True`` if ``object_info`` represents a vehicle.

        Vehicles are the primary-detector objects (component 1) that a number
        plate would belong to. Matched by component id or a vehicle label.

        Args:
            object_info: An :class:`~Modules.va_filter_builder.ObjInfo`.

        Returns:
            ``True`` if the object is a vehicle.
        """
        if object_info.get_component_id() == 1:
            return True
        label = (object_info.get_object_label() or '').strip().lower()
        return label in ('car', 'vehicle', 'truck', 'bus', 'bicycle', 'motorbike', 'motorcycle')

    @staticmethod
    def _containing_vehicle(plate_info: Any, vehicles: List[Any]) -> Optional[Any]:
        """Return the vehicle whose bbox contains the plate's centre point.

        Uses the plate's centre point so a plate lying across two nearby
        vehicles still resolves to the correct one. Among candidates the
        smallest containing vehicle is preferred (best fit).

        Args:
            plate_info: The plate :class:`~Modules.va_filter_builder.ObjInfo`.
            vehicles: Candidate vehicle objects.

        Returns:
            The containing vehicle, or ``None`` if no vehicle contains the plate.
        """
        centre = plate_info.get_center_point()
        best = None
        best_area = None
        for vehicle in vehicles:
            left = vehicle.left
            top = vehicle.top
            right = left + vehicle.width
            bottom = top + vehicle.height
            if left <= centre.x <= right and top <= centre.y <= bottom:
                area = vehicle.width * vehicle.height
                if best_area is None or area < best_area:
                    best = vehicle
                    best_area = area
        return best
