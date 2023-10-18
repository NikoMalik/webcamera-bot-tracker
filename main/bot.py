import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pickle
import os

# Initialize your bot token
bot = Bot(token="Your_token")
dp = Dispatcher(bot)

# Initialize the bot within the dispatcher
logging_middleware = LoggingMiddleware()
dp.setup_middleware(logging_middleware)

# Load face recognition data
with open("your_pickle", "rb") as file:
    data = pickle.load(file)
    
    

# Initialize video capture from the camera
camera_url = "your_url"   #Replace with your camera URL
username = "your_name"  # Replace with your camera username
password = "your_password"  # Replace with your camera password

camera_credentials = f"{username}:{password}@"
video = cv2.VideoCapture(f"{camera_url}/{camera_credentials}video")

# Set camera properties
video.set(cv2.CAP_PROP_FPS, 30)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Create a directory to store screenshots
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

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

async def send_unknown_character_notification(user_id, file_path):
    caption = "Unknown character detected."
    with open(file_path, "rb") as photo:
        await bot.send_photo(user_id, photo, caption=caption)
        os.remove(file_path)  # Remove the temporary screenshot file

async def detect_faces_and_notify(user_id):
    frame_number = 0
    screenshot_interval = 5
    while True:
        ret, frame = video.read()
        frame_number += 1
        if not ret:
            break

        locations = face_recognition.face_locations(frame, model="hog")
        encodings = face_recognition.face_encodings(frame, locations)

        recognized_names = []

        for face_encoding in encodings:
            results = face_recognition.compare_faces(data["encodings"], face_encoding)

            if any(results):
                index = results.index(True)
                recognized_names.append(data["names"][index])

        unknown_names = ["Unknown"] * (len(locations) - len(recognized_names))
        names = recognized_names + unknown_names

        frame_with_annotations = draw_face_rectangles_pil(frame, locations, names)
        cv2.imshow("Face Detection in Video", frame_with_annotations)

        if "Unknown" in names:
            screenshot_filename = f"screenshots/screenshot_{frame_number}.png"
            cv2.imwrite(screenshot_filename, frame)  # Save the frame as a screenshot
            await send_unknown_character_notification(user_id, screenshot_filename)

        k = cv2.waitKey(20)
        if k == ord("q"):
            print("Q pressed, closing the app")
            break

def main():
    user_id = 111111  # Replace with the user ID to send notification
    asyncio.get_event_loop().run_until_complete(detect_faces_and_notify(user_id))

if __name__ == '__main__':
    main()
