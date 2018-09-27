import face_recognition
import cv2
import boto3
# import awscam
import threading
import os
import json
import numpy as np
from keras.preprocessing import image
import random
import string
from websocket import create_connection

#------- Classes and functions


class people():
    face_found = False
    val = {
        'name': 'Unknown',
        'mood': 'Unknown',
        'imageName': 'none',
        'eventName': 'profile',
        'reactions': {
            'happy': '0',
            'sad': '0',
            'angry': '0',
            'calm': '0',
            'disgusted': '0',
            'confused': '0',
            'surprised': '0'
        }
    }


class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(
            self, group, target, name, args, kwargs, Verbose)
        self._return = None

    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)

    def join(self):
        threading.Thread.join(self)
        return self._return


def getAllEncodingsFromS3():
    list = s3Client.list_objects(Bucket='face-rec-final-enc')['Contents']
    for key in list:
        s3Client.download_file('face-rec-final-enc',
                               key['Key'], 'data/'+key['Key'])

# function that takes a frame and create make an emotion recognition locally


def localEmotionRecognition(img):

    img = cv2.resize(img, (740, 560))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
         # crop detected face
        detected_face = img[int(y):int(y + h), int(x):int(x + w)]
        detected_face = cv2.cvtColor(
            detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
        detected_face = cv2.resize(
            detected_face, (48, 48))  # resize to 48x48

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        # pixels are in scale of [0, 255]. normalize all pixels in scale of
        # [0, 1]
        img_pixels /= 255
        predictions = model.predict(img_pixels)
        # connect face and expressions
        cv2.line(img, (int((x + x + w) / 2), y + 15),
                 (x + w, y - 20), (255, 255, 255), 1)
        cv2.line(img, (x + w, y - 20),
                 (x + w + 10, y - 20), (255, 255, 255), 1)
        all_emotions_numbers = []
        for i in range(len(predictions[0])):
            people.val['reactions'][emotions[i]] = round(
                predictions[0][i]*100, 2)
            all_emotions_numbers.append(round(predictions[0][i]*100, 2))
        if max(all_emotions_numbers) > 80:
            people.val['mood'] = emotions[all_emotions_numbers.index(
                max(all_emotions_numbers))]
        else:
            people.val['mood'] = 'Unknown'
    return 'done!'

# picks a randomword as an ID


def randomword():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(10))
# uploads a file to S3 bucket


def uploadToS3(passed_small_frame):
    temp_image_word = randomword()
    people.val['imageName'] = temp_image_word+'.jpg'
    people.val['name'] = temp_image_word
    print(people.val['imageName'])
    cv2.imwrite(filename='Faces/' +
                people.val['imageName'], img=passed_small_frame)
    temp_image = face_recognition.load_image_file(
        'Faces/'+people.val['imageName'])
    temp_encoding = face_recognition.face_encodings(temp_image)[0]
    np.save('ImagesEncodings/'+temp_image_word+'.npy', temp_encoding)
    known_people_encodings.append(temp_encoding)
    known_people_name.append(temp_image_word)
    s3.Bucket(picturesBucket).upload_file("Faces/" +
                                          people.val['imageName'], people.val['imageName'], ExtraArgs={'ACL': 'public-read'})
    s3.Bucket(encodingsBucket).upload_file("ImagesEncodings/" +
                                           temp_image_word+'.npy', temp_image_word+'.npy', ExtraArgs={'ACL': 'public-read'})


# opencv initialization
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# -----------------------------
# face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open(
    "facial_expression_model_structure.json", "r").read())
model.load_weights(
    'facial_expression_model_weights.h5')  # load weights
# -----------------------------

emotions = ('angry', 'disgusted', 'confused',
            'happy', 'sad', 'surprised', 'calm')
# --------------------------- End of dependencies

# aws rekognition client declaration
client = boto3.client('rekognition')
# aws s3 bucket client declaration
s3Client = boto3.client('s3')
# bucket instance
s3 = boto3.resource('s3')
picturesBucket = "face-rec-final"
encodingsBucket = "face-rec-final-enc"

# ---------------- Code starts here
# Open a file
known_people_name = []
known_people_encodings = []
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
temp_person = 'unknown'
temp_emotion = 'unknown'

# get all the encodings from the bucket

getAllEncodingsFromS3()
path = "data/"
dirs = os.listdir(path)
for file in dirs:
    fname, fext = file.split('.')
    known_people_name.append(fname)
    temp_encoding = np.load('data/' + file)
    known_people_encodings.append(temp_encoding)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
ws = create_connection("ws://gitex.ahla.io:5555")
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # ret, frame = awscam.getLastFrame()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(
            known_people_encodings, face_encoding, tolerance=0.5)
        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            emotionDetect = ThreadWithReturnValue(target=localEmotionRecognition, args=(frame,))
            emotionDetect.start()
            localEmotionRecognition(frame)
            first_match_index = matches.index(True)
            name = known_people_name[first_match_index]
            people.val['name'] = name
            people.val['imageName'] = 'none'
            people.face_found = True
    if people.face_found == False and not face_encodings == []:
        localEmotionRecognition(frame)
        print('unknown person detected!')
        uploadImage = ThreadWithReturnValue(target=uploadToS3, args=(frame,))
        uploadImage.start()
    elif face_encodings == []:
        people.val = {
            'name': 'Unknown',
            'mood': 'Unknown',
            'imageName': 'none',
            'eventName': 'profile',
            'reactions': {
                'happy': '0',
                'sad': '0',
                'angry': '0',
                'calm': '0',
                'disgusted': '0',
                'confused': '0',
                'surprised': '0'
            }
        }

    process_this_frame = not process_this_frame
    if people.face_found == True:
        emotionDetect.join()

    if temp_person != people.val['name'] or temp_emotion != people.val['mood']:
        ws.send(json.dumps(people.val))
        temp_person = people.val['name']
        temp_emotion = people.val['mood']
        print('New person or emotion detected!')
        print(people.val)
        people.face_found = False

# Release handle to the webcam
ws.close()
video_capture.release()
cv2.destroyAllWindows()
