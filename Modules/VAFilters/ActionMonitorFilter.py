"""Per-person action recognition monitor for the deepstream video-analytics pipeline.

This filter runs after the per-object action-recognition classifier stage. For
each tracked person it records the detected action over time, builds a per-person
history, and reports the current and primary (most frequent) actions. A
``fall_floor`` action additionally raises an alert.

The action labels are defined in :data:`ACTION_LABELS`.
"""

import time
from typing import Any, Dict, List, Optional

start: float = time.time()

ACTION_LABELS: List[str] = ['push', 'fall_floor', 'walk', 'run', 'ride_bike']


class PersonActionRecord:
    """Tracks what an individual tracked person has been doing over time."""

    def __init__(self, person_id: int) -> None:
        self.person_id: int = person_id
        self.object_label: Optional[str] = None
        self.actions: List[str] = []       # most recent action labels, oldest first
        self.confidences: List[float] = []  # confidence for each recorded action

    def record(
        self, object_label: str, action: str, confidence: float, max_history: int = 10
    ) -> None:
        """Append an action observation for this person."""
        self.object_label = object_label
        self.actions.append(action)
        self.confidences.append(confidence)
        if len(self.actions) > max_history:
            self.actions.pop(0)
            self.confidences.pop(0)

    def current_action(self) -> Optional[str]:
        """Return the most recent action label, or ``None`` if none recorded."""
        if not self.actions:
            return None
        return self.actions[-1]

    def current_confidence(self) -> float:
        """Return the confidence of the most recent action (0.0 if none)."""
        if not self.confidences:
            return 0.0
        return self.confidences[-1]

    def primary_action(self) -> Optional[str]:
        """Return the most frequently observed action in the history."""
        if not self.actions:
            return None
        counts: Dict[str, int] = {}
        for action in self.actions:
            counts[action] = counts.get(action, 0) + 1
        return max(counts, key=counts.get)


class ActionMonitorFilter:
    """Keeps tabs of what each person is doing from the action-recognition stage.

    The action-recognition stage runs as a per-object secondary classifier, so
    each tracked person carries an action label in its classifier meta (readable
    through the object's ``classifierLabels``). This filter records that action
    for each person over time, building a per-person action history, and reports
    the person's current and primary actions. A ``fall_floor`` action raises an
    alert.
    """

    def __init__(self) -> None:
        self.filter_name: str = 'ActionMonitorFilter'
        self.personRecords: Dict[int, PersonActionRecord] = {}

    def _get_record(self, person_id: int) -> PersonActionRecord:
        """Return the per-person record for ``person_id``, creating it if needed.

        Args:
            person_id: The tracker object id of the person.

        Returns:
            The :class:`PersonActionRecord` tracking this person's history.
        """
        if person_id not in self.personRecords:
            self.personRecords[person_id] = PersonActionRecord(person_id)
        return self.personRecords[person_id]

    @staticmethod
    def _classifier_action(classifier_labels: Optional[List[str]]) -> Optional[str]:
        """Pick the action label out of an object's classifier labels."""
        for label in classifier_labels or []:
            if label in ACTION_LABELS:
                return label
        return None

    def run_filter(self, frame_metadata: "FrameMetadata") -> None:
        """Process one frame of metadata and report each person's action.

        Iterates the frame's tracked objects, reads each person's action from the
        per-object classifier meta, records it into that person's history, and
        emits the aggregated metadata (current action, primary action, action
        history) via ``print`` and :meth:`set_va_output`. Persons without a
        recognized action are skipped.

        Args:
            frame_metadata: The :class:`FrameMetadata` for the current frame.
        """
        for object_info in frame_metadata.get_object_list():
            person_id: int = object_info.get_trackerID()
            obj_label: str = object_info.get_object_label()
            classifier_labels: List[str] = object_info.get_object_classification()
            action: Optional[str] = self._classifier_action(classifier_labels)
            if action is None:
                continue

            record = self._get_record(person_id)
            record.record(object_label=obj_label, action=action, confidence=0.0)

            metadata: Dict[str, Any] = {
                'personID': person_id,
                'objLabel': obj_label,
                'classifierLabels': classifier_labels,
                'action': record.current_action(),
                'primary_action': record.primary_action(),
                'action_history': list(record.actions),
            }

            if action == 'fall_floor':
                metadata['alert'] = 'PERSON FELL TO FLOOR'

            print(metadata)
            frame_metadata.set_va_output(self.filter_name, metadata)
