# CAMERA-BOT-TRACKER

This project provides a guide on how to train a face recognition model and create face encodings using a dataset of images. The generated encodings can be used for facial recognition in various applications.

## Table of Contents

- [CAMERA-BOT-TRACKER](#CAMERA-BOT-TRACKER)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Dataset Preparation](#dataset-preparation)
    - [Training the Model](#training-the-model)
  - [Usage](#usage)
  - [File Structure](#file-structure)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Description

The purpose of this project is to help with free open source software for your security camera with the possibility of upgrading and customizing it to suit your needs. This model can also be used to recognize faces in images or videos.


## Features

- Training a face recognition model.
- Generate face encodings from a set of images.
- Use the trained model to recognize faces.
- Use a bot to receive notifications of unknown faces
- Finding a familiar face in youtube videos


## Getting Started

Follow the steps below to get started 

### Prerequisites

Ensure you have the following dependencies installed:

- Python (3.10x recommended)
- OpenCV
- face_recognition library
- numpy
- Pillow
- pytube #optional
- aiogram


## Dataset Preparation


Organize your dataset: Create a directory containing images of the faces you want to recognize.

Run the train.py script, providing the name as a parameter. For example:
python train.py "your_name_for_pickle"

The script will process the dataset, extract faces, and create encodings for each face.


## Training the Model


After running the train.py script, you'll have a {name}_encodings.pickle file that contains the face encodings for the dataset. These encodings can be used for facial recognition tasks.

## Usage
You can use the trained model and face encodings for facial recognition in your applications. The face_recognition library can be utilized for real-time recognition or batch processing of images or video frames.

## Example usage:
with open("your_name_for_pickle_encodings.pickle", "rb") as file:
    data = pickle.load(file)

Load an image
image = face_recognition.load_image_file("test_image.jpg")

Get the face encodings of the test image
face_encodings = face_recognition.face_encodings(image)

Compare face encodings for recognition
results = face_recognition.compare_faces(data["encodings"], face_encodings[0])

Process results for facial recognition
if any(results):
    index = results.index(True)
    recognized_name = data["name"]
    print(f"Recognized as {recognized_name}")
else:
    print("Unknown face")

## File Structure
train.py: The script to train the face recognition model and generate encodings.
{name}_encodings.pickle: The output file containing face encodings for the given name.
Other project files as required.


## Acknowledgments:



[face_recognition library](https://github.com/ageitgey/face_recognition)


## Contributing
Feel free to contribute to this project by creating issues, suggesting improvements, or submitting pull requests.


## License
This project is licensed under the MIT License See the LICENSE file for details.

