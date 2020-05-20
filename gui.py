import argparse
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
from numpy import ndarray, argmin
from math import floor, ceil

import cv2
import face_recognition

def findFace(frame: ndarray, faceCascade) -> [tuple, None]:
    # make frame gray
    frameGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # equalize histogram
    frameGray = cv2.equalizeHist(frameGray)

    # -- Detect faces
    faces = faceCascade.detectMultiScale(frameGray)
    if isinstance(faces, ndarray):
        if faces.size == 4:
            x0 = faces[0][0]
            width = faces[0][2]
            y0 = faces[0][1]
            height = faces[0][3]
            return x0, y0, width, height
    return None

def findFaceAndEyes(frame: ndarray, faceCascade, eyesCascade) -> [tuple, None]:
    """
    Find face (if there is only one face) and eyes on found face in picture.

    :param frame: frame where face and eyes should be found
    :param faceCascade: Haar cascade for face detection
    :param eyesCascade: Haar cascade for eyes detection
    :return: tuple with 3 elements: 1st - tuple (x0, y0, w, h) for face, 2nd and 3rd - same tuples for eyes.
             None if there is no faces or more than one face, if can not detect 2 eyes on found face
    """
    # make frame gray
    frameGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # equalize histogram
    frameGray = cv2.equalizeHist(frameGray)

    # -- Detect faces
    faces = faceCascade.detectMultiScale(frameGray)
    if isinstance(faces, ndarray):
        if faces.size == 4:
            x0 = faces[0][0]
            width = faces[0][2]
            y0 = faces[0][1]
            height = faces[0][3]
            eyes = eyesCascade.detectMultiScale(frameGray[y0: y0 + height, x0: x0 + width])
            if isinstance(eyes, ndarray):
                if eyes.size == 8:
                    return ((x0, y0, width, height),
                            (eyes[0][0], eyes[0][1], eyes[0][2], eyes[0][3]),
                            (eyes[1][0], eyes[1][1], eyes[1][2], eyes[1][3]))
    return None


def cropFace(frame: ndarray, face: tuple, width: int = 100, height: int = 100) -> ndarray:
    """
    Crop face in frame

    :param frame: frame where face should be
    :param face: tuple with 3 elements: 1st - tuple (x0, y0, w, h) for face, 2nd and 3rd - same tuples for eyes.
    :param width: width of output image
    :param height: height of output image
    :return: cropped frame with only face in it
    """
    borderCoefficient = 0.2
    y0 = floor(face[1] - face[3] * borderCoefficient if face[1] - face[3] * borderCoefficient >= 0 else 0)
    y1 = ceil(face[1] + face[3] * (1 + borderCoefficient)
              if face[1] + face[3] * (1 + borderCoefficient) <= frame.shape[0] else frame.shape[0])
    x0 = floor(face[0] - face[2] * borderCoefficient if face[0] - face[2] * borderCoefficient >= 0 else 0)
    x1 = ceil(face[0] + face[2] * (1 + borderCoefficient)
              if face[0] + face[2] * (1 + borderCoefficient) <= frame.shape[1] else frame.shape[1])

    frame = frame[y0:y1,
            x0:x1]
    # frame = cv.circle(frame, (face[1][0], face[1][1]), 1, (255, 0, 0), 4)
    # frame = cv.circle(frame, (face[2][0], face[2][1]), 1, (255, 0, 0), 4)
    frame = cv2.resize(frame, (width, height))
    return frame


class App:
    def __init__(self, window, window_title, faceCascade, eyesCascade, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        self.face_cascade = faceCascade
        self.eyes_cascade = eyesCascade
        self.knownFaces = []
        self.knownNames = []

        self.facePhoto = None
        self.faceFrame = None
        self.camPhoto = None

        self.isRemembering = False

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        # self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        # self.canvas.pack()

        self.camCanvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height, bg='white')
        self.camCanvas.grid(column=0, row=0)
        self.faceCanvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height, bg='white')
        self.faceCanvas.grid(column=1, row=0)

        self.recognizedFaceStatus = tk.StringVar()
        self.recognizedFaceStatusLabel = tk.Label(window)
        self.recognizedFaceStatusLabel['textvariable'] = self.recognizedFaceStatus
        self.recognizedFaceStatusLabel.grid(column=1, row=1)

        self.rememberFaceStatus = tk.StringVar()
        self.rememberFaceStatusLabel = tk.Label(window)
        self.rememberFaceStatusLabel['textvariable'] = self.rememberFaceStatus
        self.rememberFaceStatusLabel.grid(column=1, row=2)
        self.rememberButton = tk.Button(window, text="Remember face", width=50, command=self.rememberFace)
        self.rememberButton.grid(column=1, row=3)

        self.window.columnconfigure(0, weight=1, minsize=self.vid.width)
        self.window.columnconfigure(1, weight=1, minsize=self.vid.width)
        self.window.rowconfigure(0, weight=1, minsize=self.vid.height)
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        self.window.rowconfigure(1, weight=1)
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        self.window.rowconfigure(2, weight=1)
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        self.window.rowconfigure(3, weight=1)

        self.window.update()
        self.window.minsize(self.window.winfo_width(), self.window.winfo_height())

        # Button that lets the user take a snapshot
        # self.btn_snapshot = tk.Button(window, text="Snapshot", width=50, command=self.snapshot)
        # self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def rememberFace(self):
        if self.isRemembering is False:
            self.isRemembering = True
            if self.faceFrame is not None:
                encodedFace = face_recognition.face_encodings(self.faceFrame)
                if len(encodedFace) > 0:
                    encodedFace = encodedFace[0]
                    if not (True in face_recognition.compare_faces(self.knownFaces, encodedFace)):
                        face = face_recognition.face_encodings(self.faceFrame)
                        name = simpledialog.askstring(title="Input name", prompt="Input name")
                        if name is None:
                            self.rememberFaceStatus.set('You should provide name to remember face')
                        elif name == '':
                            self.rememberFaceStatus.set('You should provide name that is not empty')
                        else:
                            self.knownNames.append(name)
                            self.knownFaces.append(face[0])
                            self.rememberFaceStatus.set('Done!')
                    else:
                        self.rememberFaceStatus.set('This face is already remembered.')
                else:
                    self.rememberFaceStatus.set('Bad quality of image. Try again.')
            else:
                self.rememberFaceStatus.set('There is no any face.')
            self.isRemembering = False

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            faceCoordinates = findFace(frame, self.face_cascade)
            # faceCoordinates = findFaceAndEyes(frame, self.face_cascade, self.eyes_cascade)
            # faceCoordinates = faceCoordinates[0] if faceCoordinates is not None else None
            if faceCoordinates is not None:
                self.faceFrame = cropFace(frame, faceCoordinates, int(self.vid.width), int(self.vid.height))
                self.facePhoto = ImageTk.PhotoImage(image=Image.fromarray(self.faceFrame))
                self.faceCanvas.create_image(0, 0, image=self.facePhoto, anchor=tk.NW)

                faceEncodings = face_recognition.face_encodings(self.faceFrame)
                name = "Unknown"
                if (len(faceEncodings) == 1) and (len(self.knownFaces) > 0):
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.knownFaces, faceEncodings[0])

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(self.knownFaces, faceEncodings[0])
                    if isinstance(face_distances, ndarray):
                        best_match_index = int(argmin(face_distances))
                        if matches[best_match_index]:
                            name = self.knownNames[best_match_index]

                self.recognizedFaceStatus.set(name)

            self.camPhoto = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.camCanvas.create_image(0, 0, image=self.camPhoto, anchor=tk.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return None, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    # parse args to find cascade classifier
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--face_cascade', help='Path to face cascade.',
                        default='data/haarcascades/haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.',
                        default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    args = parser.parse_args()

    face_cascade_name = args.face_cascade
    eyes_cascade_name = args.eyes_cascade

    # init classifier class
    face_cascade = cv2.CascadeClassifier()
    eyes_cascade = cv2.CascadeClassifier()

    # -- 1. Load the cascades
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        raise ValueError('Can not load face cascade. Did you pass right argument?')
    if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
        raise ValueError('Can not load eyes cascade. Did you pass right argument?')
    # Create a window and pass it to the Application object
    App(tk.Tk(), "Tkinter and OpenCV", face_cascade, eyes_cascade)
