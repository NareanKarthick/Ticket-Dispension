from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
# Initialize the camera
camera = cv2.VideoCapture(0)

# Load the pre-trained CNN model
model = load_model('model.h5')

# Global variable to store the path of the captured image
captured_image_path = None

def generate_frames():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    global captured_image_path
    global capture_val
    capture_val = 0
    success, frame = camera.read()
    frame = cv2.flip(frame, 1)
    if success:
        captured_image_path = 'static/captured_frame.jpg'
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y+h, x:x+w]
        try:
            cv2.imwrite(captured_image_path, face)
            print("Captured")
        except:
            print("Error")
        return captured_image_path, 200
    else:
        return "Failed to Capture Frame", 500

@app.route('/classify', methods=['POST'])
def classify():
    global captured_image_path
    if captured_image_path and os.path.exists(captured_image_path):
        # Preprocess the image
        image = cv2.imread(captured_image_path)
        image = cv2.resize(image, (224, 224))  # Resize to the size your model expects
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize the image

        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Map the predicted class index to a label
        class_labels = ['0-3 : Free ticket', '3-13 : Half ticket', '13 and above : Full ticket']  # Replace with your actual class labels
        result = class_labels[predicted_class]

        return jsonify({'result': result}), 200
    else:
        return jsonify({'error': 'No image captured'}), 404

if __name__ == "__main__":
    app.run(debug=True)
