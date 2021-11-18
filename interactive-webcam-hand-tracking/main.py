from hand_tracking import HandTrackingApp, VideoCaptureException

app = HandTrackingApp()

try:
    print("Starting app...")
    app.run()
except VideoCaptureException as error:
    print("Error: {}".format(str(error)))
