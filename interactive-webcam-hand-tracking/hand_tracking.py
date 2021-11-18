import cv2.cv2 as cv2

from hand_detector import HandDetectorWrapper


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

        # Initialize the camera
        self.__init_webcam()
        cv2.namedWindow('Hand Tracking Capture')
        cv2.setMouseCallback('Hand Tracking Capture', self.__click)

        # OpenCV interface
        self.__pressed = False
        self.__button_x1 = (25, 25)
        self.__button_x2 = (275, 50)
        self.__button_color = (255, 0, 0)
        self.__text_button_origin = (30, 46)
        self.__text_button_color = (255, 255, 255)

        # Effect options
        self.__option = 0

    def __click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the click was made inside the rectangle
            if 25 < x < 275 and 25 < y < 50:
                self.__pressed = not self.__pressed

    def __init_webcam(self):
        self.__capture = cv2.VideoCapture(0)

        if not self.__capture.isOpened():
            raise VideoCaptureException("Error when accessing the webcam.")

    def __set_bounding_box(self, frame):
        # Set the dimensions and positions of the bounding box
        self.__height, self.__width = frame.shape[:2]

        self.__x1 = self.__width // 2
        self.__y1 = 0
        self.__x2 = self.__x1 + 400
        self.__y2 = self.__y1 + 400

        # Draw the bounding box
        cv2.rectangle(frame, (self.__x1, self.__y1), (self.__x2, self.__y2), (255, 0, 0), 2)

    def __apply_filter(self, frame):
        # Option 1: Convert to GRAYSCALE
        if self.__option == 1:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Option 2: Convert to HSV
        if self.__option == 2:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Option 3: Convert to RGB
        if self.__option == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        return frame

    def __run_video_capture(self):

        done = False

        while not done:
            # Start the capture
            success, frame = self.__capture.read()
            if not success:
                raise VideoCaptureException("Error, empty camera frame")

            # Prepare the frame image
            frame = cv2.resize(frame, (0, 0), fx=1.25, fy=1.25)
            frame = cv2.flip(frame, 1)

            # Set the button and the text
            cv2.rectangle(
                frame,
                pt1=self.__button_x1,
                pt2=self.__button_x2,
                color=self.__button_color,
                thickness=2
            )
            # Set the text on the button
            cv2.putText(
                frame,
                text='Click for detection',
                org=self.__text_button_origin,
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1.5,
                color=self.__text_button_color,
                thickness=2)

            # Set the detection box
            if self.__pressed is True:
                self.__set_bounding_box(frame)

                # Detect, but only inside the bounding box
                frame = self.__hand_detector.detect(frame, self.__x1, self.__y1, self.__x2, self.__y2)

                # Get the landmarks for the hand detected
                landmarks_detected = self.__hand_detector.get_landmarks(frame)
                if landmarks_detected:
                    digit_class = self.__hand_detector.detect_hand_digit_from_landmarks()

                    if digit_class != 0:
                        self.__option = digit_class
                        frame = self.__apply_filter(frame)

            cv2.imshow('Hand Tracking Capture', frame)

            ch = cv2.waitKey(1)
            if ch & 0xFF == ord('q'):
                done = True

        self.__capture.release()
        cv2.destroyAllWindows()

    def run(self):
        self.__run_video_capture()
