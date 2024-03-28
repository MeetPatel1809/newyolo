import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, Response, send_from_directory
from werkzeug.utils import secure_filename
import os
import time
from ultralytics import YOLO

app = Flask(__name__)

# Error handler for internal server errors
@app.errorhandler(500)
def internal_server_error(e):
    return "Internal Server Error: " + str(e)

# Error handler for file not found errors
@app.errorhandler(404)
def page_not_found(e):
    return "Page not found: " + str(e)

@app.route("/")
def hello_world():
    return render_template('mk.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    try:
        if request.method == "POST":
            if 'file' in request.files:
                f = request.files['file']
                basepath = os.path.dirname(__file__)
                filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
                print("upload folder is ", filepath)
                f.save(filepath)
                global imgpath
                imgpath = f.filename
                print("printing imgpath:", imgpath)

                file_extension = f.filename.rsplit('.', 1)[1].lower()

                if file_extension == 'jpg':
                    img = cv2.imread(filepath)
                    frame = cv2.imencode('.jpg', img)[1].tobytes()

                    image = Image.open(io.BytesIO(frame))
                    yolo = YOLO('best.pt')
                    detections = yolo.predict(image, save=True)
                    return display(f.filename)

                elif file_extension == 'mp4':
                    video_path = filepath
                    cap = cv2.VideoCapture(video_path)

                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

                    model = YOLO('best.pt')

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        results = model(frame, save=True)
                        print(results)
                        cv2.waitKey(1)
                        res_plotted = results[0].plot()
                        cv2.imshow("result", res_plotted)

                        out.write(res_plotted)
                        if cv2.waitKey(1) == ord('q'):
                            break

                    return video_feed()

        return render_template('mk.html')

    except Exception as e:
        return "An error occurred: " + str(e)

@app.route('/<path:filename>')
def display(filename):
    try:
        folder_path = 'runs/detect'
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
        directory = folder_path + '/' + latest_subfolder
        files = os.listdir(directory)
        latest_file = files[0]
        filename = os.path.join(folder_path, latest_subfolder, latest_file)
        file_extension = filename.rsplit('.', 1)[1].lower()

        environ = request.environ
        if file_extension == 'jpg':
            return send_from_directory(directory, latest_file, environ)
        else:
            return "Invalid file format"

    except Exception as e:
        return "An error occurred: " + str(e)

def get_frame():
    try:
        folder_path = os.getcwd()
        mp4_files = 'output.mp4'
        video = cv2.VideoCapture(mp4_files)
        while True:
            success, image = video.read()
            if not success:
                break
            ret, jpeg = cv2.imencode('.jpg', image)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.1)

    except Exception as e:
        return "An error occurred: " + str(e)

@app.route("/video_feed")
def video_feed():
    try:
        return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

    except Exception as e:
        return "An error occurred: " + str(e)

if __name__ == "_main_":
    try:
        parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
        parser.add_argument("--port", default=5000, type=int, help="port number")
        args = parser.parse_args()

        app.run(host='0.0.0.0', port=args.port)

    except Exception as e:
        print("An error occurred: " + str(e))