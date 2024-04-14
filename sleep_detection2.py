"""
Sleep Detection Script
Author: Aniket Bhosale
Team: Bucket
Date: 04/13/2024
"""

import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

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

def process_frame(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (450, 450))  # Resize the image to 450x450 for faster processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    face_detector = dlib.get_frontal_face_detector() # Use the dlib face detector to detect faces
    feature_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # Use the dlib shape predictor to find facial features

    faces = face_detector(gray_image, 0) # Detect faces in the image
    for face in faces:
        features = feature_predictor(gray_image, face)
        features = face_utils.shape_to_np(features)
        left_eye = features[leftStart:leftEnd]
        right_eye = features[rightStart:rightEnd]
        left_ratio = eye_open_ratio(left_eye)
        right_ratio = eye_open_ratio(right_eye)
        avg_ratio = (left_ratio + right_ratio) / 2

        return avg_ratio

