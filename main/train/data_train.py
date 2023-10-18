# Import necessary libraries
import os
import pickle
import face_recognition
import cv2

# Define a function to check if a file is an image
def is_image_file(filename):
    # Check if the file has a common image file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    file_extension = os.path.splitext(filename)[1].lower()
    return file_extension in valid_extensions

# Define the main function to train and create encodings
def train(name):
    # Specify the directory containing image files
    dataset_dir = "your_folder"

    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] The directory '{dataset_dir}' does not exist.")
        return

    # Initialize a list to store known face encodings
    known_encodings = []

    # Loop through the files in the dataset directory
    for i, image in enumerate(os.listdir(dataset_dir)):
        # Check if the file is an image, if not, skip it
        if not is_image_file(image):
            continue  # Skip non-image files

        # Print a message indicating the image being processed
        print(f"[+] processing img {i + 1}/{len(os.listdir(dataset_dir))}")
        
        # Create the full path to the image file
        image_path = os.path.join(dataset_dir, image)

        try:
            # Load and encode the face in the image
            face_img = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(face_img)

            if face_encodings:
                # Add the face encoding to the list of known encodings
                face_enc = face_encodings[0]
                known_encodings.append(face_enc)
            else:
                print(f"No face found in {image}")

        except Exception as e:
            # Handle errors during image processing
            print(f"Error processing {image}: {e}")

    # Check if there are valid face encodings
    if known_encodings:
        # Create a dictionary with name and encodings data
        data = {
            "name": name,
            "encodings": known_encodings
        }
        
        # Save the data to a pickle file
        with open(f"{name}_encodings.pickle", "wb") as file:
            pickle.dump(data, file)
        print(f"[INFO] File {name}_encodings.pickle successfully created")
    else:
        print("No valid face encodings found in the dataset.")
        
        
        

        

# Define the main entry point of the script
def main():
    # Call the train function with a name parameter
    train("your_name_for_pickle")
    
# Run the main function when the script is executed directly
if __name__ == '__main__':
    main()
