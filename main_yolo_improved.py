import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse

# Initialize YOLOv8 model
model = YOLO("C:\\Human_detection\\yolo11n.pt")  # Update path as necessary

# Initialize Deep SORT tracker
tracker = DeepSort(
    max_age=10,  # Adjust based on testing
    nms_max_overlap=0.6,
    max_cosine_distance=0.3
)


def detect_and_track(frame):
    # Run YOLO inference on the frame
    results = model(frame)  # Adjust confidence as needed


    # Extract bounding boxes for 'person' class
    bbs = []  # List to store bounding box detections for Deep SORT
    person_count = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Only consider 'person' class (ID 0 in COCO dataset)
            if box.cls == 0:
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0]  # Confidence score

                # Append bounding box in format for Deep SORT
                bbs.append(([x1, y1, x2 - x1, y2 - y1], confidence, 'person'))

    # Update tracker with detections
    tracks = tracker.update_tracks(bbs, frame=frame)

    # Draw tracked bounding boxes with track IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())  # Bounding box coordinates from tracker

        # Draw bounding box and label with track ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id} ({confidence:.2f})', 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display total person count on the frame
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
