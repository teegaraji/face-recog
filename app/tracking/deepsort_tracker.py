from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortFaceTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=0.7)

    def update(self, detections, embeddings, frame=None):
        """
        detections: list of (x1, y1, x2, y2, conf)
        embeddings: list of np.array (embedding vector)
        frame: original image frame (np.ndarray)
        """
        # DeepSort expects: [ [x1, y1, x2, y2, conf, ...], ... ]
        # You may need to concatenate embedding to detection if required by your DeepSort version
        tracks = self.tracker.update_tracks(detections, embeddings, frame=frame)
        return tracks
