import rclpy
from rclpy.node import Node

import numpy as np
from tf2_ros import Buffer, Time, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from april_tag_positioning.goal_direction_video import GoalDirectionWebVisualizer


CHECK_FREQ = 30.0
LOG_PERIOD = 1.0
TAG_ORIGIN = np.array([0.0, 0.0, 0.0])
WEB_HOST = "0.0.0.0"
WEB_PORT = 8050
WEB_REFRESH_MS = 100

GOAL_POINT = np.array([5.0, 0.0, 0.0])  # Ponto fixo no mapa para o drone se aproximar


class TagListener(Node):
    def __init__(self):
        super().__init__('tag_listener')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_log_time = self.get_clock().now()
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
