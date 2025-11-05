import cv2
import streamlit as st
import os
from datetime import datetime

# Load the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(r'C:\Users\pc\Desktop\GMC DS\Cours\Facial recognition\haarcascade_frontalface_default.xml')


# Function for face detection
def detect_faces(scale_factor, min_neighbors, color, save_images):
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Convert color from hex to BGR
    color = tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    color = (color[2], color[1], color[0])  # Convert RGB to BGR for OpenCV

    st.write("**Press 'Q' in the webcam window to quit.**")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Unable to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Show the frame in a new window
        cv2.imshow('Face Detection - Press Q to Quit', frame)

        # Save images if user chose to
        if save_images and len(faces) > 0:
            folder = "detected_faces"
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(filename, frame)

        # Quit when user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Streamlit app
def app():
    st.title("Face Detection using Viola-Jones Algorithm, By Djalal eddine")
    st.write("""
    Welcome to the **Face Detection App**!  
    This app uses the **Viola-Jones algorithm (Haar Cascade)** to detect faces in real-time from your webcam.  

    **How to use:**
    1. Allow webcam access when prompted.  
    2. Adjust the detection parameters below if needed.  
    3. Choose your preferred rectangle color.  
    4. Click **Start Detection** to begin.  
    5. Press **Q** in the webcam window to stop detection.
    """)

    # --- Parameters section ---
    st.subheader("⚙️ Detection Parameters")
    scale_factor = st.slider("Scale Factor", 1.05, 1.5, 1.3, 0.01,
                             help="Controls how much the image size is reduced at each image scale.")
    min_neighbors = st.slider("Min Neighbors", 3, 10, 5,
                              help="Specifies how many neighbors each candidate rectangle should have to retain it.")

    color = st.color_picker("Select rectangle color", "#00FF00")

    save_images = st.checkbox("Save images with detected faces", value=False)

    if st.button("🚀 Start Detection"):
        detect_faces(scale_factor, min_neighbors, color, save_images)


if __name__ == "__main__":
    app()
