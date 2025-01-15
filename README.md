# facial_recognition

Creating an attendance system using facial recognition with OpenCV involves several steps, including setting up your environment, capturing faces, recognizing them, and marking attendance. Below is a step-by-step guide to help you build this system.

1. Set Up Your Environment
Before you begin, make sure you have the required tools and libraries installed.

Install Python: Download and install the latest version of Python (Python 3.x).
Install Required Libraries: You will need several Python libraries like OpenCV, face recognition, numpy, and others.
bash
Copy code
pip install opencv-python opencv-python-headless numpy face_recognition
2. Import Required Libraries
In your Python script, start by importing the necessary libraries:

python
Copy code
import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import datetime
3. Set Up Face Recognition for Known Faces
To recognize faces, you need to have images of the people whose attendance you want to track. These images should be stored in a specific folder.

Create a Folder: Create a folder (e.g., images) where you'll store the face images of people.
Load Known Faces: Write code to load the images, convert them to grayscale, and extract facial features for recognition.
python
Copy code
def load_known_faces():
    known_faces = []
    known_names = []
    image_folder = 'images'
    
    # Loop through each image in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for face_recognition
        img_encoding = face_recognition.face_encodings(img_rgb)[0]  # Encode the face
        
        # Store the encoding and the name (image name without extension)
        known_faces.append(img_encoding)
        known_names.append(os.path.splitext(image_name)[0])
    
    return known_faces, known_names
4. Face Detection & Recognition
You will use the webcam to capture faces in real-time and compare them against the known faces in the database.

python
Copy code
def recognize_faces(known_faces, known_names):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces and their encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
            
            # If there is a match, get the name of the person
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
            
            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Mark attendance
            mark_attendance(name)
        
        # Display the resulting frame
        cv2.imshow('Attendance System', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
5. Mark Attendance
You need to store attendance data. A simple way is to save it to a CSV file with timestamps.

python
Copy code
def mark_attendance(name):
    with open("attendance.csv", "r+") as file:
        data = file.readlines()
        names = []
        for line in data:
            entry = line.split(',')
            names.append(entry[0])
        
        if name not in names:
            time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            file.writelines(f'{name},{time_now}\n')
            print(f"Attendance Marked for {name} at {time_now}")
6. Putting It All Together
Now you need to integrate all the parts into a single program. The basic flow will be:

Load known faces and names.
Continuously capture frames from the webcam.
Recognize faces and mark attendance.
python
Copy code
def main():
    known_faces, known_names = load_known_faces()
    recognize_faces(known_faces, known_names)

if __name__ == "__main__":
    main()
7. Testing
Make sure to place clear images of the people whose attendance you want to track in the images folder.
When you run the script, it will open the webcam, and if it detects a face, it will try to match it with the known faces.
When a match is found, it will mark the attendance in a CSV file with the name and timestamp.
Additional Enhancements:
Add a GUI: You can build a simple graphical user interface using Tkinter to show attendance records or manage settings.
Handle Multiple Faces: You can also make improvements to handle multiple faces in the frame.
Improve Accuracy: Fine-tune the face recognition model or try using deep learning models to improve accuracy.
Example Output (CSV):
Copy code
JohnDoe,2025-01-15 10:00:00
JaneDoe,2025-01-15 10:05:00
Conclusion:
This is a basic facial recognition-based attendance system. For real-world usage, you might need to handle edge cases like lighting issues, multiple people in the frame, or improve the performance for large-scale systems.



