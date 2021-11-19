import cv2.cv2 as cv2

from hand_detector import HandDetectorWrapper
from tensorflow import keras


class VideoCaptureException(Exception):
    def __init__(self, message):
        self.__message = message
        super.__init__(self.__message)

    def __str__(self):
        return self.__message


class HandTrackingApp:
    def __init__(self):

        # Initialize the mediapipe hand detector
        self.__hand_detector = HandDetectorWrapper(
            static_image_mode=False,
            maximum_number_hands=2,
            detection_confidence=.5,
            tracking_confidence=.5
        )
        self.__landmarks = []

        # Initialize the CNN hand classifier
        # self.__load_classifier()

        # Initialize the camera
        self.__init_webcam()
        cv2.namedWindow('Hand Tracking Capture')
        cv2.setMouseCallback('Hand Tracking Capture', self.__click)

        # Detection button
        self.__pressed = False
        self.__button_x1 = (25, 25)
        self.__button_x2 = (50, 50)
        self.__button_color = (255, 0, 0)
        self.__text_button_origin = (60, 46)
        self.__text_button_color = (255, 255, 255)

        # MediaPipe detection button
        self.__button_detect_option_media_x1 = (25, 75)
        self.__button_detect_option_media_x2 = (50, 100)
        self.__text_button_media_origin = (60, 96)
        self.__media_pressed = False

        # CNN Model detection button
        self.__button_detect_option_model_x1 = (25, 125)
        self.__button_detect_option_model_x2 = (50, 150)
        self.__text_button_model_origin = (60, 146)
        self.__cnn_pressed = False

        # Detection option
        self.__media_option = False
        self.__cnn_option = False

        # Effect options
        self.__option = 0

    def __set_frame(self, frame):
        self.__frame = frame

    def __load_classifier(self):
        self.__hand_digit_classifier = keras.models.load_model('./model/model.h5')
        print('CNN Hand Classifier loaded.')

    def __click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the click was made inside the detection rectangle
            if self.__button_x1[0] < x < self.__button_x2[0] and \
                    self.__button_x1[1] < y < self.__button_x2[1]:
                self.__pressed = not self.__pressed

            # Check if the click was made inside the mediapipe rectangle
            if self.__button_detect_option_media_x1[0] < x < self.__button_detect_option_media_x2[0] and \
                    self.__button_detect_option_media_x1[1] < y < self.__button_detect_option_media_x2[1]:
                self.__media_pressed = True
                self.__cnn_pressed = False

            # Check if the click was made inside the cnn rectangle
            if self.__button_detect_option_model_x1[0] < x < self.__button_detect_option_model_x2[0] and \
                    self.__button_detect_option_model_x1[1] < y < self.__button_detect_option_model_x2[1]:
                self.__cnn_pressed = True
                self.__media_pressed = False

    def __init_webcam(self):
        self.__capture = cv2.VideoCapture(0)

        if not self.__capture.isOpened():
            raise VideoCaptureException("Error when accessing the webcam.")

    def __set_bounding_box(self):
        # Set the dimensions and positions of the bounding box
        self.__height, self.__width = self.__frame.shape[:2]

        self.__x1 = self.__width // 2
        self.__y1 = 0
        self.__x2 = self.__x1 + 400
        self.__y2 = self.__y1 + 400

        # Draw the bounding box
        cv2.rectangle(self.__frame, (self.__x1, self.__y1), (self.__x2, self.__y2), (255, 0, 0), 2)

    def __create_rectangle(self, x1, x2, filled):
        if filled:
            thickness = -1
        else:
            thickness = 1
        cv2.rectangle(self.__frame, pt1=x1, pt2=x2, color=self.__button_color, thickness=thickness)

    def __put_text(self, text, origin):
        cv2.putText(
            self.__frame,
            text=text,
            org=origin,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=.75,
            color=self.__text_button_color,
            thickness=1)

    def __set_button_interface(self):
        # Set the button and the text for starting the detection
        self.__create_rectangle(self.__button_x1, self.__button_x2, self.__pressed)

        # Set the text on the detection button
        self.__put_text('Enable detection', self.__text_button_origin)

        # Set the button for the mediapipe detection method
        self.__create_rectangle(self.__button_detect_option_media_x1, self.__button_detect_option_media_x2, self.__media_pressed)

        # Set the text for the mediapipe detection method
        self.__put_text('MediaPipe Detection', self.__text_button_media_origin)

        # Set button for the model detection method
        self.__create_rectangle(self.__button_detect_option_model_x1, self.__button_detect_option_model_x2, self.__cnn_pressed)

        # Set the text for the model detection method
        self.__put_text('CNN Detection', self.__text_button_model_origin)

    def __apply_filter(self):
        # Option 1: Convert to GRAYSCALE
        if self.__option == 1:
            self.__frame = cv2.cvtColor(self.__frame, cv2.COLOR_BGR2RGB)

        # Option 2: Convert to HSV
        if self.__option == 2:
            self.__frame = cv2.cvtColor(self.__frame, cv2.COLOR_BGR2GRAY)

        # Option 3: Convert to RGB
        if self.__option == 3:
            self.__frame = cv2.cvtColor(self.__frame, cv2.COLOR_BGR2HSV)

    def __run_video_capture(self):

        done = False

        while not done:
            # Start the capture
            success, frame = self.__capture.read()
            if not success:
                raise VideoCaptureException("Error, empty camera frame")

            self.__set_frame(frame)

            # Prepare the frame image
            self.__frame = cv2.resize(self.__frame, (0, 0), fx=1.25, fy=1.25)
            self.__frame = cv2.flip(self.__frame, 1)

            # Set the buttons interface
            self.__set_button_interface()

            # Set the detection box
            if self.__pressed is True:
                self.__set_bounding_box()

                if self.__media_pressed is True:
                    self.__frame = self.__hand_detector.detect(self.__frame, self.__x1, self.__y1, self.__x2, self.__y2)

                    # Get the landmarks for the hand detected
                    landmarks_detected = self.__hand_detector.get_landmarks(self.__frame)
                    if landmarks_detected:
                        digit_class = self.__hand_detector.detect_hand_digit_from_landmarks()

                        if digit_class != 0:
                            self.__option = digit_class
                            self.__apply_filter()

            cv2.imshow('Hand Tracking Capture', self.__frame)

            ch = cv2.waitKey(1)
            if ch & 0xFF == ord('q'):
                done = True

        self.__capture.release()
        cv2.destroyAllWindows()

    def run(self):
        self.__run_video_capture()
