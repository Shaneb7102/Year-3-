from flask import Flask, render_template, jsonify, request
from mfrc522 import SimpleMFRC522
from threading import Thread
from time import sleep
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import shutil
from imutils.video import VideoStream
from imutils.video import FPS
import time
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)

app = Flask(__name__)
reader = SimpleMFRC522()
latest_rfid = None
rfid_scanning_enabled = False

@app.route("/")
def index():
    return render_template("website.html")

def facial_unlock():
    from imutils.video import VideoStream
    from imutils.video import FPS
    import face_recognition
    import imutils
    import pickle
    import time
    import cv2
    import RPi.GPIO as GPIO
    
    relay = 18;
    GPIO.setwarnings(False)
    GPIO.setup(relay, GPIO.OUT)
    GPIO.output(relay , 1)

    #Initialize 'currentname' to trigger only when a new person is identified.
    currentname = "unknown"
    #Determine faces from encodings.pickle file model created from train_model.py
    encodingsP = "encodings.pickle"

    # load the known faces and embeddings along with OpenCV's Haar
    # cascade for face detection
    
    #Algorithm for deetcting objects in images regardless of scale in image and location.
    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open(encodingsP, "rb").read())

    # initialize the video stream and allow the camera sensor to warm up
    # Set the ser to the followng
    #vs = VideoStream(src=2,framerate=10).start()
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)

    # start the FPS counter
    fps = FPS().start()

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        # Detect the fce boxes
        #This line of code uses the face_recognition library to detect face locations in a given image frame. The face_locations() function takes an image (in this case, frame) as input and returns a list of bounding boxes for all detected faces in the image.
        boxes = face_recognition.face_locations(frame)
        
        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                encoding)
            name = "Unknown" #if face is not recognized, then print Unknown

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

                #If someone in your dataset is identified, print their name on the screen
                if currentname != name:
                    currentname = name
                    GPIO.output(relay , 0)
                    print(currentname)

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image - color is in BGR
            cv2.rectangle(frame, (left, top), (right, bottom),
                (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                .8, (0, 255, 255), 2)

        # display the image to our screen
        cv2.imshow("Facial Recognition is Running", frame)
        key = cv2.waitKey(1) & 0xFF

        # quit when 'q' key is pressed
        if key == ord("q"):
            GPIO.output(relay , 1)        
            break
        

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup

    cv2.destroyAllWindows()
    vs.stop()
    
@app.route('/facial_unlock', methods=['GET'])
def trigger_facial_unlock():
    thread = Thread(target=facial_unlock)
    thread.start()
    return jsonify({"status": "Facial unlocking started"})


@app.route("/get_rfid", methods=["GET"])
def get_rfid():
    global latest_rfid
    if latest_rfid:
        response = jsonify({"id": latest_rfid[0], "text": latest_rfid[1]})
        latest_rfid = None
        return response
    else:
        return jsonify({"id": None, "text": None})

@app.route("/toggle_rfid", methods=["POST"])
def toggle_rfid():
    global rfid_scanning_enabled
    rfid_scanning_enabled = not rfid_scanning_enabled
    return jsonify({"enabled": rfid_scanning_enabled})

@app.route("/train_model", methods=["POST"])
def train_model():
        #! /usr/bin/python

    # our images are located in the dataset folder
    print("[INFO] start processing faces...")
    imagePaths = list(paths.list_images("dataset"))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
            len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,
            model="hog")

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open("encodings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()
    
    return jsonify({"status": "Model training complete"}), 200

    
def get_folders():
    dataset_path = "dataset"
    folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    return jsonify({"folders": folders})

@app.route("/get_folder_names", methods=["GET"])
def get_folder_names():
    folder_path = "dataset"
    folder_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return jsonify({"folders": folder_names})


@app.route("/remove_folder", methods=["POST"])
def remove_folder():
    data = request.get_json()
    folder_name = data.get("folder_name")
    folder_path = os.path.join("dataset", folder_name)

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error"})


    
@app.route("/capture_face", methods=["POST"])
def capture_face():
    import cv2
    data = request.get_json()
    name = data.get("name", "Unknown")

    # Create the directory if it does not exist
    directory = "dataset/" + name
    if not os.path.exists(directory):
        os.makedirs(directory)
  

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("press space to take a photo", 500, 300)

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("press space to take a photo", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "dataset/" + name + "/image{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

    return jsonify({"status": "success"})


def read_rfid_loop():
    global latest_rfid, rfid_scanning_enabled
    while True:
        if rfid_scanning_enabled:
            try:
                id, text = reader.read()
                print(f"RFID scanned: {id} - {text}")
                latest_rfid = (id, text)
                sleep(1)
            except KeyboardInterrupt:
                GPIO.cleanup()
                break
        else:
            sleep(1)
            


if __name__ == "__main__":
    Thread(target=read_rfid_loop).start()
    app.run(host="0.0.0.0", port=5000)


