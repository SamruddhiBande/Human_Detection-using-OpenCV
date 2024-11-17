import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse

# Initialize YOLO model paths
person_model_path = "C:\\Human_detection\\yolo11n.pt"  # Update path as necessary for person detection
gender_model_path = "C:\\Human_detection\\Gender_detection_model.pt"  # Update for gender detection

# Load YOLO models
person_model = YOLO(person_model_path)  # YOLO model for person detection
gender_model = YOLO(gender_model_path)  # YOLO model for gender classification

# Initialize Deep SORT tracker
tracker = DeepSort(
    max_age=10,
    nms_max_overlap=0.6,
    max_cosine_distance=0.3
)

def detect_and_track(frame):
    """
    Perform detection and tracking for both person and gender classification.
    """
    # Person detection
    person_results = person_model(frame, iou=0.3)  # YOLO inference with IoU threshold
    bbs = []  # Bounding boxes for Deep SORT
    person_count = 0
    male_count = 0
    female_count = 0

    for result in person_results:
        for box in result.boxes:
            # Check if the detection is of class 'person' (COCO class ID 0)
            if int(box.cls[0]) == 0:
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = float(box.conf[0])

                # Crop the detected person for gender classification
                cropped_person = frame[y1:y2, x1:x2]

                # Gender detection
                gender_results = gender_model(cropped_person)
                for gender_result in gender_results:
                    for gender_box in gender_result.boxes:
                        class_id = int(gender_box.cls[0])  # Class ID for gender
                        gender_confidence = float(gender_box.conf[0])
                        if class_id == 0:  # Assuming 0 is 'Male'
                            gender = "Male"
                            male_count += 1
                            color = (255, 0, 0)  # Blue for males
                        elif class_id == 1:  # Assuming 1 is 'Female'
                            gender = "Female"
                            female_count += 1
                            color = (0, 0, 255)  # Red for females
                        else:
                            continue

                        # Draw gender label
                        cv2.putText(
                            frame,
                            f'{gender}: {gender_confidence:.2f}',
                            (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                # Append bounding box with normalized confidence for tracker
                bbs.append(([x1, y1, x2 - x1, y2 - y1], confidence, 'person'))

    # Update tracker with detections
    tracks = tracker.update_tracks(bbs, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())  # Convert to bounding box coordinates

        # Draw bounding box and track ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f'ID {track_id}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Display counts on the frame
    cv2.putText(frame, f'Total Persons: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Males: {male_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Females: {female_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Detection and Tracking", frame)

    return frame


def detect_by_camera():
    """
    Detect people and genders using the webcam.
    """
    video = cv2.VideoCapture(0)
    print('Detecting and tracking people...')
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = detect_and_track(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def detect_by_video(video_path, output_path=None):
    """
    Detect people and genders in a video file and optionally save the output.
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video file at {video_path}")
        return

    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Define codec and initialize VideoWriter for saving output
    out = None
    if output_path:
        # Using 'avc1' codec for better compatibility (H.264)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Saving output to: {output_path}")

    print("Detecting and tracking people in the video...")
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        processed_frame = detect_and_track(frame)  # Process each frame
        if out:
            out.write(processed_frame)  # Write the processed frame to output video
        cv2.imshow("Video", processed_frame)  # Show the processed frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    if out:
        out.release()
    cv2.destroyAllWindows()




def detect_by_image(image_path, output_path=None):
    """
    Detect people and genders in a single image and optionally save the output.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot open image {image_path}")
        return

    print("Detecting and tracking people in the image...")
    processed_image = detect_and_track(image)  # Process the image

    if output_path:
        cv2.imwrite(output_path, processed_image)  # Save the processed image
        print(f"Output image saved at: {output_path}")

    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def human_detector(args):
    """
    Main detection function.
    """
    video_path = args['video']
    image_path = args['image']
    camera = args['camera']
    output_path = args['output']

    if camera:
        print('[INFO] Opening Web Camera...')
        detect_by_camera()
    elif video_path:
        print('[INFO] Opening Video...')
        detect_by_video(video_path, output_path)  # Pass output path for saving video
    elif image_path:
        print('[INFO] Opening Image...')
        detect_by_image(image_path, output_path)  # Pass output path for saving image

def args_parser():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, help="path to video file")
    parser.add_argument("-i", "--image", type=str, help="path to image file")
    parser.add_argument("-c", "--camera", action='store_true', help="set to true to use the camera")
    parser.add_argument("-o", "--output", type=str, help="path to save the output video or image")
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    args = args_parser()
    human_detector(args)
