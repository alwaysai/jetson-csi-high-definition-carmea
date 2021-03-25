
"""
Advanced CSI video services for NVIDIA Jetson products.

A set of utility services that can be used for capturing and gathering
statistics on a csi video streams.  Services include real time fps statistics
on inbound frames from the camera and fps on displaying frames for the
application.  Other utilies include camera management function and data display
function
"""
import threading
from collections import deque
from enum import Enum
import cv2


class CameraFailedToStart(RuntimeError):
    """Error when camera initialization fails."""

    def __init__(self, cmd, backend, append=''):
        """Failure to start camera RuntimeError."""
        self._cmd = cmd
        self._backend = backend
        self._append = append

    def __str__(self):
        """Failure camera start up message."""
        return '''Failed to open video capture for {} using backend {}. {}
        Note: Check to make sure your camera is plugged in correctly.
        Common errors during installation of CSI cameras include plugging the
        strip in backwards and plugging the camera into the wrong port.
        '''.format(self._cmd, self._backend, self._append)


class CameraConnectionLost(RuntimeError):
    """
    Camera connection failure.

    Error for when the camera connection is not found after the connection has
    been established.
    """

    def __init__(self):
        """Error message for camera connection lost."""
        super().__init__(
            """Failed to read a frame from video capture.Connection to
            the camera has been lost."""
        )


class RepeatTimer(threading.Timer):
    """Timer to reset statistics at a specified interval."""

    def run(self):
        """Run RepeatTimer method."""
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class _BaseVideoStream:
    def __init__(self, cmd, backend):
        self._cmd = cmd
        self._backend = backend
        self.fps_timer = None
        self._running = False
        self.frames_read = 0
        self.frames_displayed = 0
        self.last_frames_read = 0
        self.last_frames_displayed = 0
        self._stream = None
        self._thread = None
        self._queue_depth = 2
        self._frame_queue = deque(maxlen=self._queue_depth)
        self._stopped = False

    def start(self):
        """
        Start reading frames from the video stream.

        :returns: `self`

        :raises: :class:`~edgeiq.edge_tools.CameraFailedToStart`
                         if video stream fails to open
        """
        # initialize the video stream and read the first frame
        # from the stream
        try:
            self._stream = cv2.VideoCapture(self._cmd, self._backend)
        except RuntimeError:
            raise CameraFailedToStart(self._cmd, self._backend)
        if self._stream is None or not self._stream.isOpened():
            raise CameraFailedToStart(self._cmd, self._backend,
                                      'Stream not open.')
        (grabbed, frame) = self._stream.read()  # Attempt to grab a frame
        if grabbed is False or frame is None:
            raise CameraFailedToStart(self._cmd, self._backend,
                                      'Failed to grab frame')
        self._update_failure = threading.Event()
        self._thread = threading.Thread(target=self._update, args=())
        self._thread.start()
        return self

    @property
    def fps(self):
        """
        Run FPS averaging of the video stream.

        :type: float

        :raises: `RuntimeError` if FPS cannot be queried
        """
        fps = self._stream.get(cv2.CAP_PROP_FPS)
        if fps == -1.0:
            raise RuntimeError('Failed to get camera FPS!')
        return fps

    def update_fps_stats(self):
        """Update fps stats."""
        self.last_frames_read = self.frames_read
        self.last_frames_displayed = self.frames_displayed
        self.frames_read = 0
        self.frames_displayed = 0

    def start_counting_fps(self):
        """Start fps counter to get camera input and dispaly stats."""
        self.fps_timer = RepeatTimer(1.0, self.update_fps_stats)
        self.fps_timer.start()

    def _update(self):
        """Read frames as they're available."""
        while True:
            # if the thread indicator variable is set, stop the thread
            if self._stopped:
                return
            # otherwise read the next frame from the stream
            (grabbed, frame) = self._stream.read()
            if grabbed is False or frame is None:
                self._update_failure.set()
                return
            self.frames_read += 1
            self._frame_queue.appendleft(frame)

    def read(self):
        """
        Return the most recent frame from the camera.

        This function blocks on waiting for a new frame.

        :returns: numpy array -- The frame that was read from the camera
        """
        while True:
            if self._update_failure.is_set():
                raise CameraConnectionLost()

            if len(self._frame_queue) > 0:
                break
        return self._frame_queue.pop()

    def stop(self):
        """Stop and clean up the camera connection."""
        self._stopped = True
        if self._thread:
            self._thread.join()
        self._stream.release()

    def release_fps_stats(self):
        self.fps_timer.cancel()
        self.fps_timer.join()

    def __enter__(self):
        return self.start()

    def __exit__(self, type, value, traceback):
        self.stop()


class FrameRotation(Enum):
    """Amount of rotation applied to each frame in degrees."""

    ROTATE_NONE = 0
    ROTATE_90 = 90
    ROTATE_180 = 180


class JetsonCameraMode(Enum):
    """
    Sensor mode for Jetson CSI camera.

    Sensor Mode applied to CSI camera which determines input width and height
    and framerate.  The first Number identifies the Sony Sensor Number (IMX219
    or IMX477).  The second numbers are the input width and height.  The third
    number is the framerate and fourth the number is the camera sensor mode.
    """

    IMX219_3264x2468_21_0 = 0
    IMX219_3264x1848_28_1 = 1
    IMX219_1920x1080_30_2 = 2
    IMX219_1640x1232_30_3 = 3
    IMX477_4032x3040_30_0 = 4
    IMX477_1920x1080_60_1 = 5
    IMX477_2560x1440_40_3 = 7


class JetsonVideoStream(_BaseVideoStream):
    """
    Capture video frames from a CSI ribbon camera on NVIDIA Jetson.

    `JetsonVideoStream` can be instantiated as a context manager::

        with edgeiq.JetsonVideoStream() as video_stream:
            ...

    To use `JetsonVideoStream` directly, use the
    :func:`~edgeiq.edge_tools.JetsonVideoStream.start()` and
    :func:`~edgeiq.edge_tools.JetsonVideoStream.stop()` functions::

        video_stream = edgeiq.JetsonVideoStream().start()
        ...
        video_stream.stop()

    Typical usage::

        with edgeiq.JetsonVideoStream() as video_stream:
            while True:
                frame = video_stream.read()

    :type cam: integer
    :param cam: The integer identifier of the camera.
    :type rotation: :class:`~FrameRotation`
    :param rotation: The rotation applied to each frame
    :type camera_mode: :class:`~JetsonCameraMode`
    :param camera_mode: The sensor mode for csi camera
    :type display_width: integer
    :param display_width: The output image width in pixels.
    :type display_height: integer
    :param display_height: The output image height in pixels.
    """

    def __init__(
            self, cam=0, rotation=FrameRotation.ROTATE_NONE,
            camera_mode=JetsonCameraMode.IMX219_1920x1080_30_2,
            display_width=640, display_height=480):
        """Initialize CSI camera."""
        self._sensor_id = cam
        self._rotation = rotation
        self._sensor_mode = camera_mode
        self._display_width = display_width
        self._display_height = display_height

        if self._rotation == FrameRotation.ROTATE_NONE:
            flip = 0
        elif self._rotation == FrameRotation.ROTATE_90:
            flip = 1
        elif self._rotation == FrameRotation.ROTATE_180:
            flip = 2
        else:
            raise ValueError(
                    'Invalid input for rotation: {}'.format(self._rotation))

        if self._sensor_mode == JetsonCameraMode.IMX219_3264x2468_21_0:
            self._mode = 0
            self._capture_width = 3264
            self._capture_height = 2468
            self._framerate = 21
        elif self._sensor_mode == JetsonCameraMode.IMX219_3264x1848_28_1:
            self._mode = 1
            self._capture_width = 3264
            self._capture_height = 1848
            self._framerate = 28
        elif self._sensor_mode == JetsonCameraMode.IMX219_1920x1080_30_2:
            self._mode = 2
            self._capture_width = 1920
            self._capture_height = 1080
            self._framerate = 30
        elif self._sensor_mode == JetsonCameraMode.IMX219_1640x1232_30_3:
            self._mode = 3
            self._capture_width = 1640
            self._capture_height = 1232
            self._framerate = 30
        elif self._sensor_mode == JetsonCameraMode.IMX477_4032x3040_30_0:
            self._mode = 0
            self._capture_width = 4032
            self._capture_height = 3040
            self._framerate = 30
        elif self._sensor_mode == JetsonCameraMode.IMX477_1920x1080_60_1:
            self._mode = 1
            self._capture_width = 1920
            self._capture_height = 1080
            self._framerate = 60
        elif self._sensor_mode == JetsonCameraMode.IMX477_2560x1440_40_3:
            self._mode = 3
            self._capture_width = 2560
            self._capture_height = 1440
            self._framerate = 40
        else:
            raise ValueError(
                    'Invalid input for camera_mode: {}'.format(
                      self._sensor_mode))

        cmd = (
                'nvarguscamerasrc sensor-id=%d sensor-mode=%d !'
                'video/x-raw(memory:NVMM), '
                'width=(int)%d, height=(int)%d, '
                'format=(string)NV12, framerate=(fraction)%d/1 ! '
                'nvvidconv flip-method=%d ! '
                'video/x-raw, width=(int)%d, height=(int)%d,format=(string)BGRx ! '
                'videoconvert ! '
                'video/x-raw, format=(string)BGR ! appsink '
                'wait-on-eos=false drop=true max-buffers=60' % (
                    self._sensor_id, self._mode,
                    self._capture_width, self._capture_height,
                    self._framerate, flip, self._display_width,
                    self._display_height))

        backend = cv2.CAP_GSTREAMER
        super(JetsonVideoStream, self).__init__(cmd=cmd, backend=backend)


def read_camera(camera_stream, monitor_fps):
    """
    Read camera video stream and monitor fps in real time.

    This function reads camera stream and adds fps information if monitor_fps
    variable is True.

    :type camera_stream: :class:`WebcamVideoStream` 'JetsonVideoStream'
                                'GStreamerVideoStream'
    :param camera_stream: The VideoStream to read from.
    :type monitor_fps: :boolean
    :param monitor_fps: True value enables fps statistics to be visiable on the
                        image

    :returns: image -- Numpy array of image in BGR format
    """
    camera_image = camera_stream.read()
    if monitor_fps:
        draw_label(camera_image, "Frames Displayed (PS): "+str(
                    camera_stream.last_frames_displayed), (10, 20))
        draw_label(camera_image, "Frames Read (PS): "+str(
                    camera_stream.last_frames_read), (10, 40))
    return camera_image


def draw_label(image, label_text, label_position):
    """
    Draw a label on a image.

    This function will place a label on image at a specified position.

    :type  image: numpy array of image in BGR format
    :param image: The image for label to be placed on.
    :type label_text: string
    :param label_text: Text string to be drawn on the image.
    :type label_position: tuples of two values i.e. (X coordinate value,
                          Y coordinate value).
    :param label_position: The coordinates of the bottom-left corner of
                           the text string in the image.

    :returns: image -- numpy array of image in BGR format with label on it
    """
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (255, 255, 255)
    # You can get the size of the string with cv2.getTextSize here
    cv2.putText(image, label_text, label_position, font_face, scale, color, 1,
                cv2.LINE_AA)
