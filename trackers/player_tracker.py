from ultralytics import YOLO
import pickle
import cv2 
import sys
from utils.bbox_utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path=r"models\yolov8x.pt"):
        # Load a pretrained YOLO model for player tracking
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        # Implement logic to choose and filter players based on court keypoints
        player_detection_first_frame = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detection_first_frame)
        # Filter detections to keep only chosen players
        filtered_player_detections = []
        for frame_detections in player_detections:
            filtered_detections = {track_id: bbox for track_id, bbox in frame_detections.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_detections)

        return filtered_player_detections
    
    
    def choose_players(self, court_keypoints, player_detections):
        # Implement logic to choose players based on court keypoints
        distances = []
        for track_id, bbox in player_detections.items():
           player_center = get_center_of_bbox(bbox)

           # Calculate distance to each court keypoint and take the minimum
           min_distance = float('inf')
           # court_keypoints is a flat array [x0, y0, x1, y1, ...], iterate in pairs
           for i in range(0, len(court_keypoints), 2):
                keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, keypoint)
                if distance < min_distance:
                     min_distance = distance
           distances.append((track_id, min_distance))
        # Sort players by distance to court keypoints and select the closest two
        distances.sort(key=lambda x: x[1])
        # choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(self, frames,read_from_stubs= False, stubs_path= None):

        # Check if we need to read from stubs
        # stubs_path is the path to the pickle file where detections are stored
        # If read_from_stubs is True and stubs_path is provided, load detections from the pickle file
        if read_from_stubs and stubs_path is not None:
            with open(stubs_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        
        # Detect players in each frame and store the results
        player_detections = []
        for frame in frames:
            detections = self.detect_frame(frame)
            player_detections.append(detections)

        # Save detections to stubs if path is provided
        if  stubs_path is not None:
            with open(stubs_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    """
    notice 
    there are two kinds of IDs here:
    1. Tracking ID: This ID is assigned to each detected player and is used to track the player across frames.
    It helps in maintaining the identity of the player as they move around in the video.
    2. Class ID: This ID represents the type of object detected (e.g., person, car, dog, etc.).
    """

    """ Detect players in a single frame and return a dictionary of tracking IDs and bounding boxes """
    def detect_frame(self, frame):
        # Perform detection on a single frame
        results = self.model.track(frame,persist=True)[0]
        # id name dict for class names
        id_name_dict = results.names
        player_dict = {}
        # Iterate through detected boxes and filter for players
        for box in results.boxes:
            # Get tracking ID and bounding box coordinates
            # box.id is a tensor, convert it to int
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            # Get class ID and name
            object_cls_id = box.cls.tolist()[0]
            # Check if the detected object is a person (class ID 0 in COCO dataset)
            object_class_name = id_name_dict[object_cls_id]
            if object_class_name == 'person':
                player_dict[track_id] = result

        return player_dict
    
    def draw_bboxes(self,video_frames, player_detections):
        import cv2
        out_video_frame = []
        for frame, detections in zip(video_frames, player_detections):
            for track_id, bbox in detections.items():
                x1, y1, x2, y2 = map(int, bbox)
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put tracking ID text
                cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            out_video_frame.append(frame)
        return out_video_frame