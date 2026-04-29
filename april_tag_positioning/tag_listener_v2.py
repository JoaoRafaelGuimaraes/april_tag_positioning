import rclpy
from rclpy.node import Node

import numpy as np
from tf2_ros import Buffer, Time, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from april_tag_positioning.goal_direction_video import GoalDirectionWebVisualizer

try:
    import cv2
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
except ImportError as exc:  # pragma: no cover - depends on ROS env
    cv2 = None
    CvBridge = None
    Image = None
    _CAMERA_IMPORT_ERROR = exc
else:
    _CAMERA_IMPORT_ERROR = None


CHECK_FREQ = 30.0
LOG_PERIOD = 1.0
TAG_ORIGIN = np.array([0.0, 0.0, 0.0])
WEB_HOST = "0.0.0.0"
WEB_PORT = 8050
WEB_REFRESH_MS = 100
CAMERA_TOPIC = "/camera/camera/color/image_raw"
CAMERA_TOPIC_CHECK_PERIOD = 1.0
CAMERA_PREVIEW_MAX_WIDTH = 480
CAMERA_PREVIEW_JPEG_QUALITY = 65
CAMERA_PREVIEW_MAX_FPS = 15.0

GOAL_POINT = np.array([5.0, 0.0, 0.0])  # Ponto fixo no mapa para o drone se aproximar


class TagListener(Node):
    def __init__(self):
        super().__init__('tag_listener')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_log_time = self.get_clock().now()
        self.last_camera_preview_time = 0.0
        self.camera_bridge = CvBridge() if CvBridge is not None else None
        self.camera_subscription = None
        self.camera_topic_warned_missing = False
        self.camera_topic_logged_available = False
        self.camera_dependency_warned = False
        self.visualizer = GoalDirectionWebVisualizer(
            goal_point=GOAL_POINT,
            tag_origin=TAG_ORIGIN,
            host=WEB_HOST,
            port=WEB_PORT,
            refresh_ms=WEB_REFRESH_MS,
        )
        self.visualizer.start()

        self.get_logger().info("Iniciando a escuta das transformações...")
        self.get_logger().info(
            "Visualizador web em tempo real disponivel em "
            f"{self.visualizer.get_local_url()} . "
            "Se estiver em SSH, use tunel de porta para acessar no navegador local."
        )

        self.timer = self.create_timer(1.0 / CHECK_FREQ, self.run)
        self.camera_topic_timer = self.create_timer(
            CAMERA_TOPIC_CHECK_PERIOD,
            self._ensure_camera_subscription,
        )


    def run(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",        # target frame
                "camera_estimated",   # source frame
                Time()           # último transform disponível
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Transform not available yet: {e}")
            return

        drone_position = transform.transform.translation
        drone_position_array = np.array(
            [drone_position.x, drone_position.y, drone_position.z],
            dtype=float,
        )

        direction_vector = self.get_vector_to_goal(drone_position_array)
        self.visualizer.update(drone_position_array)

        now = self.get_clock().now()
        elapsed_since_log = (now - self.last_log_time).nanoseconds / 1e9
        if elapsed_since_log >= LOG_PERIOD:
            self.last_log_time = now
            self.get_logger().info(
                "Drone em map: "
                f"x={drone_position_array[0]:.3f}, "
                f"y={drone_position_array[1]:.3f}, "
                f"z={drone_position_array[2]:.3f} | "
                "vetor para goal: "
                f"dx={direction_vector[0]:.3f}, "
                f"dy={direction_vector[1]:.3f}, "
                f"dz={direction_vector[2]:.3f}"
            )

        return


    def get_vector_to_goal(self, position):
        return GOAL_POINT - np.asarray(position, dtype=float)

    def _ensure_camera_subscription(self):
        if self.camera_subscription is not None:
            return

        if _CAMERA_IMPORT_ERROR is not None:
            if not self.camera_dependency_warned:
                self.camera_dependency_warned = True
                self.get_logger().warn(
                    "Preview da camera desabilitado porque dependencias de "
                    f"imagem nao estao disponiveis: {_CAMERA_IMPORT_ERROR}"
                )
            return

        topic_names = dict(self.get_topic_names_and_types())
        if CAMERA_TOPIC not in topic_names:
            self.visualizer.set_camera_topic_available(False)
            if not self.camera_topic_warned_missing:
                self.camera_topic_warned_missing = True
                self.get_logger().warn(
                    f"Topico {CAMERA_TOPIC} nao encontrado. "
                    "A janela auxiliar da camera nao sera exibida."
                )
            return

        self.camera_subscription = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self._camera_callback,
            10,
        )
        self.visualizer.set_camera_topic_available(True)
        if not self.camera_topic_logged_available:
            self.camera_topic_logged_available = True
            self.get_logger().info(
                f"Topico de camera encontrado: {CAMERA_TOPIC}"
            )

    def _camera_callback(self, msg):
        now_seconds = self.get_clock().now().nanoseconds / 1e9
        if now_seconds - self.last_camera_preview_time < 1.0 / CAMERA_PREVIEW_MAX_FPS:
            return

        self.last_camera_preview_time = now_seconds

        try:
            frame = self.camera_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = self._resize_camera_preview(frame)
            success, encoded_frame = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), CAMERA_PREVIEW_JPEG_QUALITY],
            )
        except Exception as exc:  # pragma: no cover - depends on ROS runtime data
            self.get_logger().warn(
                f"Falha ao converter frame da camera para preview web: {exc}"
            )
            return

        if not success:
            self.get_logger().warn(
                "Falha ao codificar frame da camera para preview web."
            )
            return

        self.visualizer.update_camera_image(encoded_frame.tobytes())

    def _resize_camera_preview(self, frame):
        height, width = frame.shape[:2]
        if width <= CAMERA_PREVIEW_MAX_WIDTH:
            return frame

        scale = CAMERA_PREVIEW_MAX_WIDTH / float(width)
        resized_height = max(int(round(height * scale)), 1)
        return cv2.resize(
            frame,
            (CAMERA_PREVIEW_MAX_WIDTH, resized_height),
            interpolation=cv2.INTER_AREA,
        )


def main(args=None):
    rclpy.init(args=args)

    tag_list = TagListener()

    try:
        rclpy.spin(tag_list)
    except KeyboardInterrupt:
        pass

    tag_list.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
