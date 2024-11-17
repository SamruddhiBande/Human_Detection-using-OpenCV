import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse
from inference_sdk import InferenceHTTPClient  # For Roboflow integration

# Initialize YOLOv8 model
model = YOLO("C:\\Human_detection\\yolo11n.pt")  # Update path as necessary

# Initialize Deep SORT tracker
tracker = DeepSort(
    max_age=10,  # Adjust based on testing
    nms_max_overlap=0.6,
    max_cosine_distance=0.3
)

# Initialize Roboflow client for gender classification
CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="Hd00rZ2bDqsAVVolGGYp"  # Replace with your actual Roboflow API key
)


def detect_and_track(frame):
    # Run YOLO inference with lower IoU threshold
    
    results = model(frame, iou=0.3)

    bbs = []  # Bounding boxes for Deep SORT
    person_count = 0

    for result in results:
        for box in result.boxes:
            # Check if detection is 'person' class (ID 0 in COCO dataset)
            if int(box.cls[0]) == 0:
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Append bounding box with normalized confidence
                bbs.append(([x1, y1, x2 - x1, y2 - y1], confidence, 'person'))

                # Crop the detected person from the frame
                person_image = frame[y1:y2, x1:x2]

                # Perform gender classification with Roboflow
                try:
                    # Send the cropped image to the gender classification model
                    result = CLIENT.infer(person_image, model_id="gender-8vbxd/1")  # Replace with your model ID
                    gender = result['predictions'][0]['class']  # 'male' or 'female'

                    # Display the gender label on the frame
                    cv2.putText(
                        frame,
                        f'Gender: {gender}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

                except Exception as e:
                    print(f"Error in Roboflow inference: {e}")

    # Update tracker with detections
    tracks = tracker.update_tracks(bbs, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Draw bounding box and track ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f'ID {track_id}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    # Display total person count
    cv2.putText(frame, f'Total Persons: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("Detection and Tracking", frame)

    return frame


# Modified detect functions to use detect_and_track for both detection and tracking
def detectByCamera(writer):
    video = cv2.VideoCapture(0)
    print('Detecting and tracking people...')

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame = detect_and_track(frame)

        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def detectByPathVideo(video_path, writer):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Cannot open video file at {video_path}")
        return

    print("Detecting and tracking people...")
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("Error: Unable to read frame from video.")
            break

        frame = detect_and_track(frame)

        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def detectByPathImage(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot open image {image_path}")
        return

    result_image = detect_and_track(image)

    if output_path:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def humanDetector(args):
    video_path = args['video']
    image_path = args['image']
    camera = args['camera']
    output_path = args['output']

    writer = None

    if camera:
        print('[INFO] Opening Web Camera...')
        detectByCamera(writer)
    elif video_path:
        print('[INFO] Opening Video...')
        detectByPathVideo(video_path, writer)
    elif image_path:
        print('[INFO] Opening Image...')
        detectByPathImage(image_path, output_path)


# Main function and argument parsing remain the same
def argsParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, help="path to video file")
    parser.add_argument("-i", "--image", type=str, help="path to image file")
    parser.add_argument("-c", "--camera", action='store_true', help="set to true to use the camera")
    parser.add_argument("-o", "--output", type=str, help="path to save the output video or image")
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":
    args = argsParser()
    humanDetector(args)
