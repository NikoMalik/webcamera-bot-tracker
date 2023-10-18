import cv2
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import pickle
import numpy as np
import concurrent.futures

# Define the URL for your IP camera
camera_url = "your_url"

# Set your camera username and password here
username = "your_username"
password = "your_password"

# Create a VideoCapture object for the camera feed
camera_credentials = f"{username}:{password}@"
video = cv2.VideoCapture(f"{camera_url}/{camera_credentials}video")
video.set(cv2.CAP_PROP_FPS, 30)

# Load face recognition data
with open("your_pickle", "rb") as file:
    data = pickle.load(file)

def draw_face_rectangles_pil(image, face_locations, names):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    font = ImageFont.truetype("arial.ttf", 24)
    text_color = (255, 255, 255)  # White color
    outline_color = (0, 0, 0)  # Black outline color

    for (top, right, bottom, left), name in zip(face_locations, names):
        if name == "Unknown":
            rect_color = (255, 0, 0)  # Red rectangle for unknown
        else:
            rect_color = (0, 255, 0)  # Green rectangle for recognized names

        thickness = 5
        draw.rectangle([left, top, right, bottom], outline=rect_color, width=thickness)
        
        # Draw the name above the rectangle with a thicker black outline
        text_position = (left, bottom + 40)
        draw.text(text_position, name, fill=outline_color, font=font)
        text_position = (left, bottom + 40)
        draw.text(text_position, name, fill=text_color, font=font)

    return np.array(pil_image)

def detect_faces_in_frame(frame):
    locations = face_recognition.face_locations(frame, model="hog")  # Use HOG model for faster processing
    encodings = face_recognition.face_encodings(frame, locations)

    recognized_names = []
    
    for face_encoding in encodings:
        results = face_recognition.compare_faces(data["encodings"], face_encoding)

        if any(results):
            index = results.index(True)
            recognized_names.append(data["name"])  # Append the recognized name

    unknown_names = ["Unknown"] * (len(locations) - len(recognized_names))

    names = recognized_names + unknown_names

    frame_with_annotations = draw_face_rectangles_pil(frame, locations, names)
    return frame_with_annotations

def process_frames():
    while True:
        ret, frame = video.read()
        if not ret:
            print("Cannot receive frame (stream end?). Exiting ...")
            break

        frame_with_faces = detect_faces_in_frame(frame)
        cv2.imshow("Face Detection in Camera Feed", frame_with_faces)

        k = cv2.waitKey(20)
        if k == ord('q'):
            print("Q pressed, closing the app")
            break

# Using concurrent.futures to process frames asynchronously
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    future = executor.submit(process_frames)

# Release resources when done
video.release()
cv2.destroyAllWindows()
