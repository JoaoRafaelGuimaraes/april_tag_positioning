import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.spatial.transform import Rotation as R


class CameraMotionRecorder(Node):
    def __init__(self):
        super().__init__('camera_motion_recorder')

        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )

        self.positions = []
        self.start_time = None
        self.duration = 30.0  # seconds

        self.get_logger().info("Recording camera motion for 30 seconds...")

    def tf_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.start_time is None:
            self.start_time = now

        elapsed = now - self.start_time

        if elapsed > self.duration:
            self.get_logger().info("Finished recording.")
            rclpy.shutdown()
            return

        for transform in msg.transforms:
            if transform.header.frame_id == "camera" and transform.child_frame_id.startswith("tag"):
                t = transform.transform.translation
                q = transform.transform.rotation

                # Tag pose in camera frame
                tag_pos_cam = np.array([t.x, t.y, t.z])

                rot_cam_tag = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

                # Invert transform:
                # camera position in tag frame
                cam_pos_tag = -rot_cam_tag.T @ tag_pos_cam

                self.positions.append(cam_pos_tag)

                self.get_logger().info(
                    f"Camera relative to tag: "
                    f"x={cam_pos_tag[0]:.3f}, "
                    f"y={cam_pos_tag[1]:.3f}, "
                    f"z={cam_pos_tag[2]:.3f}"
                )


def create_video(positions, output_file="camera_motion_from_tag.mp4"):
    positions = np.array(positions)

    if len(positions) < 2:
        print("Not enough points to create video.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    margin = 0.2

    xmin, ymin, zmin = positions.min(axis=0) - margin
    xmax, ymax, zmax = positions.max(axis=0) + margin

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    ax.set_xlabel("X relative to tag [m]")
    ax.set_ylabel("Y relative to tag [m]")
    ax.set_zlabel("Z relative to tag [m]")

    ax.set_title("Camera motion relative to AprilTag")

    # Tag origin
    ax.scatter([0], [0], [0], s=80, marker='x', label="AprilTag origin")

    # Camera point and trajectory
    point, = ax.plot([], [], [], marker='o', markersize=6, label="Camera")
    trail, = ax.plot([], [], [], linewidth=1, label="Trajectory")

    ax.legend()

    def update(frame):
        current = positions[:frame + 1]

        point.set_data([positions[frame, 0]], [positions[frame, 1]])
        point.set_3d_properties([positions[frame, 2]])

        trail.set_data(current[:, 0], current[:, 1])
        trail.set_3d_properties(current[:, 2])

        return point, trail

    fps = 30
    total_frames = fps * 30

    frame_indices = np.linspace(0, len(positions) - 1, total_frames).astype(int)
    sampled_positions = positions[frame_indices]

    positions[:] = sampled_positions[:len(positions)] if len(sampled_positions) == len(positions) else positions

    def update_sampled(frame):
        idx = frame_indices[frame]
        current = positions[:idx + 1]

        point.set_data([positions[idx, 0]], [positions[idx, 1]])
        point.set_3d_properties([positions[idx, 2]])

        trail.set_data(current[:, 0], current[:, 1])
        trail.set_3d_properties(current[:, 2])

        return point, trail

    ani = FuncAnimation(
        fig,
        update_sampled,
        frames=total_frames,
        interval=1000 / fps,
        blit=False
    )

    writer = FFMpegWriter(fps=fps)
    ani.save(output_file, writer=writer)

    print(f"Saved video to: {output_file}")


def main(args=None):
    rclpy.init(args=args)

    recorder = CameraMotionRecorder()

    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass

    positions = recorder.positions

    recorder.destroy_node()

    print(f"Recorded {len(positions)} camera poses.")

    create_video(positions)


if __name__ == '__main__':
    main()