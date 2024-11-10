
# Human Detection Project

This project is a human detection system that identifies people in images, videos, and live camera feeds using OpenCV's HOG (Histogram of Oriented Gradients) descriptor and a pre-trained SVM model. It also utilizes Non-Maximum Suppression (NMS) to reduce overlapping bounding boxes, providing a cleaner and more accurate detection output.

## Features

- Detects humans in images, video files, and live webcam feeds.
- Draws bounding boxes around detected people and labels each one.
- Displays the total count of detected persons.
- Saves the output to an image or video file if specified.
- Option to use the camera or load media files for detection.

## How It Works

1. **Human Detection Model**: Uses OpenCV’s HOG (Histogram of Oriented Gradients) descriptor with a default pre-trained SVM model to detect people in frames.
2. **Non-Maximum Suppression (NMS)**: Reduces overlapping bounding boxes to ensure each person is detected only once.
3. **Output Display**: Draws bounding boxes around detected humans and shows the count of persons on the display or saved media file.

## Dependencies

- **Python 3.6+**
- **OpenCV**: `cv2`
- **NumPy**: `numpy`
- **imutils**: For resizing images and frames
- **argparse**: For parsing command-line arguments

You can install the dependencies using:
```bash
pip install opencv-python imutils numpy
```

## Project Structure

```
Human-Detection-Project/
├── main.py                   # Main script for human detection
├── people.jpg                # Example input image
├── people_on_road.mp4        # Example input video
├── output_image.jpg          # Sample output image with detection
├── output_video.mp4          # Sample output video with detection
├── people_edited.jpg         # Edited image example
└── README.md                 # Project documentation
```

## Usage

### Command-line Arguments

- `--video` (-v): Path to the video file for detection.
- `--image` (-i): Path to the image file for detection.
- `--camera` (-c): Enable live camera feed for detection.
- `--output` (-o): Path to save the output video or image file.

### Running the Program

#### Image Detection

```bash
python main.py -i path/to/image.jpg -o path/to/output_image.jpg
```

#### Video Detection

```bash
python main.py -v path/to/video.mp4 -o path/to/output_video.mp4
```

#### Live Camera Feed Detection

```bash
python main.py -c
```

### Quit Detection

Press `q` to stop the video feed in both camera and video modes.

## Functionality Overview

- `apply_nms`: Applies Non-Maximum Suppression (NMS) to bounding boxes to reduce overlapping boxes.
- `detect`: Detects humans in a frame, applies NMS, and draws bounding boxes and labels.
- `humanDetector`: Main function that selects the detection mode (image, video, or camera) based on user input.
- `detectByCamera`: Captures video from a live camera and performs human detection.
- `detectByPathVideo`: Reads a video file and performs human detection frame by frame.
- `detectByPathImage`: Loads an image and performs human detection.

## Example Output

### Image Detection

Sample output images with detection bounding boxes will be saved in the specified output path.

### Video Detection

A sample output video file (`output_video.mp4`) is provided in the repository, showcasing human detection in a recorded video.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
