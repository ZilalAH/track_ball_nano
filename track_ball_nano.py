import cv2
import numpy as np
import os
import time
from ultralytics import YOLO

# --- Configuration for Jetson Nano ---
MODEL_TYPE = 'yolov8m.pt'
CONF_THRESH = 0.3
TRACKER_CONFIG = 'botsort.yaml'  # Ensure this file exists in the same directory or provide the full path

# --- Input Source ---
# For integrated camera on Jetson Nano, usually device 0.
# You might need to adjust this based on your setup (e.g., 1 for a USB camera).
CAMERA_SOURCE = 0

# --- Output Paths ---
OUTPUT_DIR_NAME = 'result_tracking_output'
OUTPUT_VIDEO_BASENAME = 'tracked_video.mp4'
OUTPUT_VIDEO_PATH = os.path.join(os.getcwd(), OUTPUT_DIR_NAME, OUTPUT_VIDEO_BASENAME)

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

# --- Functions ---

def track_ball_from_camera(camera_src, model_type, conf_thresh, tracker_config, output_path):
    """
    Tracks a sports ball in a live video stream from a camera.
    Saves the tracked video to the specified output path.
    """
    try:
        # Load the YOLO model
        model = YOLO(model_type)
        print(f"YOLO model '{model_type}' loaded.")

        # Open the video capture
        cap = cv2.VideoCapture(camera_src)
        if not cap.isOpened():
            raise IOError(f"Cannot open camera {camera_src}")
        print(f"Camera {camera_src} opened successfully.")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Assume 30 FPS if the camera doesn't provide it

        # Define the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Output will be saved to: {output_path}")

        frame_count = 0
        detected_frames = 0
        all_track_ids = []

        print("Starting real-time tracking...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            frame_count += 1

            # Perform tracking
            results = model.track(frame, classes=32, conf=conf_thresh, tracker=tracker_config, persist=True, verbose=False)

            # Process results and write to output video
            annotated_frame = results[0].plot() if results and len(results) > 0 else frame
            out.write(annotated_frame)

            # Collect statistics
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                detected_frames += 1
                current_ids = results[0].boxes.id.int().cpu().tolist()
                all_track_ids.extend(current_ids)

            # Display the processed frame (optional, can reduce performance)
            cv2.imshow('Tracked Video', annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Tracking finished and video saved.")

        # --- Analytics (after processing) ---
        print("\n--- Statistics ---")
        print(f"- Total frames processed: {frame_count}")
        if frame_count > 0:
            print(f"- Frames with ball detected: {detected_frames} ({detected_frames/frame_count:.1%})")
        else:
            print("- No frames processed.")

        if all_track_ids:
            unique_track_ids = set(all_track_ids)
            print(f"- Unique track IDs found: {len(unique_track_ids)}")
            max_sequence = 0
            if unique_track_ids:
                for track_id in unique_track_ids:
                    current_max = 0
                    longest_for_id = 0
                    for idx in all_track_ids:
                        if idx == track_id:
                            current_max += 1
                        else:
                            longest_for_id = max(longest_for_id, current_max)
                            current_max = 0
                    longest_for_id = max(longest_for_id, current_max)
                    max_sequence = max(max_sequence, longest_for_id)
            print(f"- Longest continuous track sequence for any ball instance: {max_sequence} frames")
        else:
            print("- No track IDs found for analytics.")

    except FileNotFoundError:
        print(f"Error: Model file '{model_type}' or tracker config '{tracker_config}' not found.")
    except IOError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

# --- Main Execution for Jetson Nano ---

print("--- Ball Tracking Script Start (Jetson Nano) ---")

track_ball_from_camera(CAMERA_SOURCE, MODEL_TYPE, CONF_THRESH, TRACKER_CONFIG, OUTPUT_VIDEO_PATH)

print("\n--- Ball Tracking Script End (Jetson Nano) ---")
