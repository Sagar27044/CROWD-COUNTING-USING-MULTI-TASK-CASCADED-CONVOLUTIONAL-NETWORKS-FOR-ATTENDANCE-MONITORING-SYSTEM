import cv2
import face_recognition
import os

# Load known face images from "database" folder
known_faces_dir = "face_database"
known_face_encodings = []
known_face_names = []

# Step 1: Load and encode faces from the folder
for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(filename)[0])  # Use filename (without extension) as name

# Step 2: Start webcam for real-time recognition
video_capture = cv2.VideoCapture(0)

print("🔍 Press 'q' to quit")

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Faster processing
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Scale back up face locations since frame was resized
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and cleanup
video_capture.release()
cv2.destroyAllWindows()
