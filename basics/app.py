"""
Use Jetson CSI camera interface.

Application uses JetsonVideoStream API to capture video stream from CSI camera.


https://alwaysai.co/docs/application_development/configuration_and_packaging.html
"""
import time
import edgeiq


def main():
    """Run CSI camera."""
    try:
        with edgeiq.JetsonVideoStream(cam=0,
                                      rotation=edgeiq.FrameRotation.ROTATE_180,
                                      camera_mode=edgeiq.JetsonCameraMode.
                                      IMX219_1920x1080_30_2, display_width=640,
                                      display_height=480) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            # loop detection
            while True:
                frame = video_stream.read()

                # Generate text to display on streamer
                text = ["Jetson CSI Camera"]
                streamer.send_data(frame, text)
                if streamer.check_exit():
                    break

    finally:
        print("Program Ending")


if __name__ == "__main__":
    main()
