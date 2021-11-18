import cv2.cv2 as cv2
import mediapipe as mp


class HandDetectorWrapper:
    def __init__(self, static_image_mode=False, maximum_number_hands=2, detection_confidence=.5,
                 tracking_confidence=.5):
        # Image mode: photo or video mode
        self.__static_image_mode = static_image_mode

        # Maximum number of hands for detection
        self.__maximum_number_hands = maximum_number_hands

        # Minimum confidence for detection and tracking
        self.__detection_confidence = detection_confidence
        self.__tracking_confidence = tracking_confidence

        # Initialize the Hands detector
        self.__hands_detector = mp.solutions.hands.Hands(
            model_complexity=0,
            static_image_mode=self.__static_image_mode,
            max_num_hands=self.__maximum_number_hands,
            min_detection_confidence=self.__detection_confidence,
            min_tracking_confidence=self.__tracking_confidence
        )

        # Mediapipe drawing utility
        self.__draw = mp.solutions.drawing_utils

        # Detection results
        self.__results_detection = None

        # Landmarks

    def __set_results_detection(self, results):
        self.__results_detection = results

    def __set_landmarks(self, landmarks):
        self.__landmarks = landmarks

    def detect(self, frame, x1, y1, x2, y2):
        results = self.__hands_detector.process(frame[y1:y2, x1:x2])
        self.__set_results_detection(results)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.__draw.draw_landmarks(
                    frame[y1:y2, x1:x2],
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

        return frame

    def get_landmarks(self, frame):
        landmarks = {}

        if self.__results_detection.multi_hand_landmarks:
            # Only for one hand detection
            hand_instance = self.__results_detection.multi_hand_landmarks[0]
            for landmark_id, landmark in enumerate(hand_instance.landmark):
                height, width = frame.shape[:2]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmarks[str(landmark_id)] = (x, y)

        self.__set_landmarks(landmarks)
        return len(self.__landmarks) != 0

    def detect_hand_digit_from_landmarks(self):
        # Simple 4 classes detector using landmark points

        # Extract the y coordinate for each finger of interest
        _, class_point_index_y = self.__landmarks['8']
        _, class_point_middle_y = self.__landmarks['12']
        _, class_point_ring_y = self.__landmarks['16']

        interest_landmarks_points = {'big': self.__landmarks['4'],
                                     'index': self.__landmarks['8'],
                                     'middle': self.__landmarks['12'],
                                     'ring': self.__landmarks['16'],
                                     'pinky': self.__landmarks['20']}

        # For class 1: index
        class_one = True
        del interest_landmarks_points['index']
        for point in interest_landmarks_points.values():
            _, p_y = point
            if class_point_index_y > p_y:
                class_one = False
                break

        if class_one:
            return 1

        # For class 2: index and middle
        class_two = True
        del interest_landmarks_points['middle']
        for point in interest_landmarks_points.values():
            _, p_y = point
            if class_point_index_y > p_y or class_point_middle_y > p_y:
                class_two = False
                break

        if class_two:
            return 2

        # For class 3: index, middle and ring
        class_three = True
        del interest_landmarks_points['ring']
        for point in interest_landmarks_points.values():
            _, p_y = point
            if class_point_index_y > p_y or class_point_middle_y > p_y or class_point_ring_y > p_y:
                class_three = False
                break

        if class_three:
            return 3

        return 0
