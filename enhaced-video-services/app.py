"""
Using enhanced Jetson enhanced csi package with object detector.

Use object detection to detect objects in the frame in realtime. The
types of objects detected can be changed by selecting different models.

To change the computer vision model, follow this guide:
https://alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html

To install app dependencies in the runtime container, list them in the
requirements.txt file.
"""
import edgeiq
import time
import enhanced_csi


def main():
    """Run csi video stream and object detector."""
    obj_detect = edgeiq.ObjectDetection(
                "alwaysai/ssd_mobilenet_v1_coco_2018_01_28")
    obj_detect.load(engine=edgeiq.Engine.DNN_CUDA)

    print("Loaded model:\n{}\n".format(obj_detect.model_id))
    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Labels:\n{}\n".format(obj_detect.labels))

    try:
        with enhanced_csi.JetsonVideoStream(cam=0,
                                            rotation=enhanced_csi.
                                            FrameRotation.ROTATE_180,
                                            camera_mode=enhanced_csi.
                                            JetsonCameraMode.
                                            IMX477_4032x3040_30_0,
                                            display_width=640,
                                            display_height=480) as video_stream,\
                edgeiq.Streamer() as streamer:
            time.sleep(2.0)
            video_stream.start_counting_fps()

            # loop detection
            while True:
                frame = enhanced_csi.read_camera(video_stream, True)
                results = obj_detect.detect_objects(frame, confidence_level=.4)
                frame = edgeiq.markup_image(
                        frame, results.predictions, colors=obj_detect.colors)

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                text.append("Objects:")

                for prediction in results.predictions:
                    text.append("{}: {:2.2f}%".format(
                        prediction.label, prediction.confidence * 100))

                video_stream.frames_displayed += 1

                streamer.send_data(frame, text)

                if streamer.check_exit():
                    break
            video_stream.release_fps_stats()

    finally:
        print("Program Ending")


if __name__ == "__main__":
    main()
