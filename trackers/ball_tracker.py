from ultralytics import YOLO
import pickle
import pandas as pd
import constants
from utils import (get_center_of_bbox, measure_distance, get_foot_position, get_closest_keypoint_index)
class BallTracker:
    def __init__(self, model_path=r"models\training_runs_detect_train5_weights_last.pt"):
        # Load a pretrained YOLO model for player tracking
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_detections):
        ball_positions = [x.get(1, [None, None, None, None]) for x in ball_detections]
        # Convert to DataFrame for easy interpolation
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions.interpolate(method='linear', limit_direction='both', inplace=True)
        # Fill any remaining NaN values (e.g., at the start or end) using forward and backward fill
        df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self, frames,read_from_stubs= False, stubs_path= None):
        ball_detections = []
        # Check if we need to read from stubs
        # stubs_path is the path to the pickle file where detections are stored
        # If read_from_stubs is True and stubs_path is provided, load detections from the pickle file
        if read_from_stubs and stubs_path is not None:
            with open(stubs_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections
        
        # Detect players in each frame and store the results
        for frame in frames:
            detections = self.detect_frame(frame)
            ball_detections.append(detections)

        # Save detections to stubs if path is provided
        if  stubs_path is not None:
            with open(stubs_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

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
        results = self.model.predict(frame, conf =0.15)[0]
  
        ball_dict = {}
        for box in results.boxes:
           
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict
    
    def draw_bboxes(self,video_frames, player_detections):
        import cv2
        out_video_frame = []
        for frame, detections in zip(video_frames, player_detections):
            for track_id, bbox in detections.items():
                x1, y1, x2, y2 = map(int, bbox)
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # Put tracking ID text
                cv2.putText(frame, f'Tennis ball', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,36,12), 2)
            out_video_frame.append(frame)
        return out_video_frame
    

    

