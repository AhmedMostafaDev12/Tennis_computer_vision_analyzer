from utils import (read_video, save_video, measure_distance,convert_pixel_distance_to_meters,draw_player_stats)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2 
from mini_court import MiniCourt
import constants
from copy import deepcopy
import pandas as pd 
def main():
    video_path = "input_videos/input_video.mp4"
    output_path = "output_videos/output_video.avi"

    # Read video frames
    frames = read_video(video_path)
    player_tracker = PlayerTracker(model_path=r'models/yolov8x.pt')
    player_detections = player_tracker.detect_frames(frames, read_from_stubs=True,
                                                    stubs_path='tracker_stubs/player_detections.pkl')
    
    
    ball_tracker = BallTracker(model_path=r'models/training_runs_detect_train5_weights_last.pt')
    ball_detections = ball_tracker.detect_frames(frames, read_from_stubs=True
                                                 , stubs_path='tracker_stubs/ball_detections.pkl')
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    

    court_line_detector = CourtLineDetector(model_path=r'models\training_keypoint_model.pth')
    court_keypoints = court_line_detector.predict(frames[0])

    # Draw court keypoints
    mini_court = MiniCourt(frames[0])

    # choose players - must be done before mini court conversion
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Get ball shot frames
    ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert to mini court coordinates
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, ball_detections, court_keypoints)

    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]

    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame - start_frame)/24  #24 fbs

        # get distance covered by the ball in mini court coordinates
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 
        
        # speed of the ball km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # player who has the ball 
        player_positions =  player_mini_court_detections[start_frame]
        player_shot_ball = player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                 ball_mini_court_detections[start_frame][1]))

        # opponent player speed 
        opponent_player_id = 1 if player_shot_ball ==2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 
        
        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

        # Update stats
        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] +=1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']


    # Draw detections on frames
    output_video = player_tracker.draw_bboxes(frames, player_detections)
    output_video = ball_tracker.draw_bboxes(output_video, ball_detections)
    output_video = court_line_detector.draw_keypoints_on_video(output_video, court_keypoints)



    # Draw Mini Court
    output_video = mini_court.draw_mini_court(output_video)
    output_video = mini_court.draw_points_on_mini_court(output_video,player_mini_court_detections)
    output_video = mini_court.draw_points_on_mini_court(output_video,ball_mini_court_detections, color=(0,255,255))    

    # Draw Player Stats
    output_video = draw_player_stats(output_video,player_stats_data_df)

    ## Draw frame number on top left corner 
    for idx, frame in enumerate(output_video):
        cv2.putText(frame, f'Frame: {idx+1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video, output_path)

if __name__ == "__main__":
    main()