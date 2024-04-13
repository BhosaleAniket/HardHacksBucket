"""
Sleep Detection Script
Author: Aniket Bhosale
Team: Bucket
Date: 04/13/2024
"""

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist


SLEEPINESS_THRESHOLD = 20 # If the person is drowsy for 20 frames, then the alarm will be triggered
CLOSED_EYES_THRESHOLD = 0.25 # If the eye aspect ratio is less than 0.25, then the eye is closed
(leftStart, leftEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"] # Get the indices of the left eye
(rightStart, rightEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"] # Get the indices of the right eye

def eye_open_ratio(eye):
    """
    Calculates the eye aspect ratio (EAR) according to the formula provided in the paper.
    EAR = ||p2 - p6|| + ||p3 - p5|| / 2 * ||p1 - p4||
    :param eye: np array indicating the coordinates of the eye
    :return: float indicating the EAR
    """

    diff_1 = dist.euclidean(eye[1], eye[5])
    diff_2 = dist.euclidean(eye[2], eye[4])
    denominator = dist.euclidean(eye[0], eye[3])
    return (diff_1 + diff_2) / (2.0 * denominator)


face_detector = dlib.get_frontal_face_detector() # Use the dlib face detector to detect faces
feature_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # Use the dlib shape predictor to find facial features
video_capture = cv2.VideoCapture(0) # Capture the video from the camera
frames_asleep = 0 # Counter to keep track of the number of frames the person is asleep

while True:
    _,image = video_capture.read()
    image = cv2.resize(image, (450,450)) # Resize the image to 450x450 for faster processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
    faces = face_detector(gray_image, 0) # Detect faces in the image
    for face in faces:
        features = feature_predictor(gray_image, face)
        features = face_utils.shape_to_np(features)
        left_eye = features[leftStart:leftEnd]
        right_eye = features[rightStart:rightEnd]
        left_ratio = eye_open_ratio(left_eye)
        right_ratio = eye_open_ratio(right_eye)
        avg_ratio = (left_ratio + right_ratio) / 2

        if avg_ratio < CLOSED_EYES_THRESHOLD:
            frames_asleep += 1
            if frames_asleep >= SLEEPINESS_THRESHOLD:
                print('ALERT! The person is asleep!')

        else:
            frames_asleep = 0

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

video_capture.release()
