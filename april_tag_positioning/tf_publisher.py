import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, Time, TransformListener

from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import numpy as np

#Publica as transformadas do mundo -> 
CHECK_FREQ = 10.0
class Tf_publisher(Node):
    def __init__(self):
        super().__init__('tf_publisher')
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.dynamic_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publish_fixed_tags()

        self.timer = self.create_timer(1.0 / CHECK_FREQ, self.run)

    def make_transform(self, parent_frame, child_frame, x, y, z, roll, pitch, yaw):
        transform = TransformStamped()

        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame

        transform.transform.translation.x = float(x)
        transform.transform.translation.y = float(y)
        transform.transform.translation.z = float(z)

        q = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
        # scipy gives [x, y, z, w]

        transform.transform.rotation.x = float(q[0])
        transform.transform.rotation.y = float(q[1])
        transform.transform.rotation.z = float(q[2])
        transform.transform.rotation.w = float(q[3])

        return transform

    def publish_fixed_tags(self):
        tag_0 = self.make_transform( #Position of tag_0 in the map frame
            parent_frame="map",
            child_frame="tag_0_fixed",
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0
        )

        tag_1 = self.make_transform( #Position of tag_2 in the map frame    
            parent_frame="map",
            child_frame="tag_1_fixed",
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0
        )

        self.static_broadcaster.sendTransform([tag_0, tag_1])

        self.get_logger().info("Published fixed tag transforms:")
        self.get_logger().info("map -> tag_0_fixed at (0, 0, 0)")
        self.get_logger().info("map -> tag_1_fixed at (1, 0, 0)")


    def run(self): # map -> camera = map -> tag_fixed * tag_detected -> camera
        # For now not treating the case where multiple tags are detected
        try:
            T_tag_2_camera = self.tf_buffer.lookup_transform( #comes from apriltag_ros
                "camera",        # target frame
                "tag36h11:0",   # source frame
                Time()           # último transform disponível
            )

            T_map_2_tag_fixed = self.tf_buffer.lookup_transform(
                "map",        # target frame
                "tag_0_fixed",   # source frame
                Time()           # último transform disponível
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Transform not available yet: {e}")
            return
        
        # Convert to matrices
        T_map_2_tag_fixed = self.transform_to_matrix(T_map_2_tag_fixed)
        T_tag_2_camera = self.transform_to_matrix(T_tag_2_camera)

        # Invert camera -> tag to get tag -> camera
        T_camera_2_tag = np.linalg.inv(T_tag_2_camera)

        # Assuming detected tag frame and fixed tag frame have same orientation convention:
        # map -> camera = map -> tag_fixed * tag_detected -> camera
        T_map_camera = T_map_2_tag_fixed @ T_camera_2_tag

        map_to_camera = self.matrix_to_transform(
            T_map_camera,
            parent_frame="map",
            child_frame="camera_estimated"
        )

        self.dynamic_broadcaster.sendTransform(map_to_camera)

        t = map_to_camera.transform.translation

        self.get_logger().info(
            f"Using {'detected_tag_frame'}: "
            f"camera/drone in map: "
            f"x={t.x:.3f}, y={t.y:.3f}, z={t.z:.3f}"
        )

        
        return

    def transform_to_matrix(self, transform):
        t = transform.transform.translation
        q = transform.transform.rotation

        translation = np.array([t.x, t.y, t.z])
        rotation = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        matrix = np.eye(4)
        matrix[0:3, 0:3] = rotation
        matrix[0:3, 3] = translation

        return matrix
    
    def matrix_to_transform(self, matrix, parent_frame, child_frame):
        transform = TransformStamped()

        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame

        translation = matrix[0:3, 3]
        rotation_matrix = matrix[0:3, 0:3]
        q = R.from_matrix(rotation_matrix).as_quat()

        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])

        transform.transform.rotation.x = float(q[0])
        transform.transform.rotation.y = float(q[1])
        transform.transform.rotation.z = float(q[2])
        transform.transform.rotation.w = float(q[3])

        return transform


            


def main(args=None):
    rclpy.init(args=args)

    node = Tf_publisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()