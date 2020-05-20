import argparse
from math import floor, ceil
import face_recognition
import time
import cv2
from numpy import ndarray
from os import walk


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
    y0 = floor(face[0][1] - face[0][3] * borderCoefficient if face[0][1] - face[0][3] * borderCoefficient >= 0 else 0)
    y1 = ceil(face[0][1] + face[0][3] * (1 + borderCoefficient)
              if face[0][1] + face[0][3] * (1 + borderCoefficient) <= frame.shape[0] else frame.shape[0])
    x0 = floor(face[0][0] - face[0][2] * borderCoefficient if face[0][0] - face[0][2] * borderCoefficient >= 0 else 0)
    x1 = ceil(face[0][0] + face[0][2] * (1 + borderCoefficient)
              if face[0][0] + face[0][2] * (1 + borderCoefficient) <= frame.shape[1] else frame.shape[1])

    frame = frame[y0:y1,
            x0:x1]
    # frame = cv.circle(frame, (face[1][0], face[1][1]), 1, (255, 0, 0), 4)
    # frame = cv.circle(frame, (face[2][0], face[2][1]), 1, (255, 0, 0), 4)
    frame = cv2.resize(frame, (width, height))
    return frame


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

    result = [[0, 0], [0, 0]]
    result2 = [[0, 0], [0, 0]]
    ocv_time = 0
    ocv_improved_time = 0
    frgn_time = 0

    for dirpath, dnames, fnames in walk("./data/faces/"):
        for f in fnames:
            frame = cv2.imread(dirpath + '/' + f)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ocv_yes = 0
            frgn_yes = 0
            ocv_improved_yes = 0

            # Only process every other frame of video to save time
            # Find all the faces and face encodings in the current frame of video
            start_time = time.time()
            face_locations_frgn = face_recognition.face_locations(rgb_frame)
            frgn_time += time.time() - start_time

            start_time = time.time()
            face_locations_ocv = findFace(frame, face_cascade)
            ocv_time += time.time() - start_time

            start_time = time.time()
            face_locations_ocv_improved = findFaceAndEyes(frame, face_cascade, eyes_cascade)
            ocv_improved_time += time.time() - start_time

            if face_locations_ocv is not None:
                ocv_yes = 1
                frame = cv2.circle(frame, (face_locations_ocv[0], face_locations_ocv[1]), 10, (255, 0, 0), 4)

            if face_locations_ocv_improved is not None:
                ocv_improved_yes = 1

            if len(face_locations_frgn) > 0:
                frgn_yes = 1
                frame = cv2.rectangle(frame, (face_locations_frgn[0][3], face_locations_frgn[0][0]),
                                      (face_locations_frgn[0][1], face_locations_frgn[0][2]), (0, 0, 255), 2)

            result[ocv_yes][frgn_yes] += 1
            result2[ocv_improved_yes][frgn_yes] += 1

    print(result)
    print(result2)
    print(ocv_time)
    print(ocv_improved_time)
    print(frgn_time)

    # [[20, 1066], [37, 12110]]
    # [[51, 9472], [6, 3704]]
    # 288.47836446762085
    # 363.0103597640991
    # 544.549652338028
