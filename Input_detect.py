import cv2
import torch
from facenet_pytorch import MTCNN
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image

# Initialize MTCNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# Open file dialog to select an image
root = tk.Tk()
root.withdraw()  # Hide root window
image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

# Check if user selected an image
if not image_path:
    print("No image selected. Exiting...")
else:
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for MTCNN

    # Convert image to PIL format
    image_pil = Image.fromarray(image_rgb)

    # Detect faces
    boxes, _ = mtcnn.detect(image_pil)

    # Ensure boxes are detected
    if boxes is not None:
        face_count = len(boxes)

        # Draw bounding boxes around faces
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)  # Convert to integer values
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

        # Display the face count
        text = f"Face Count: {face_count}"
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        face_count = 0
        print("No faces detected.")

    # Show the image with detected faces
    cv2.imshow("Improved Face Counting", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
