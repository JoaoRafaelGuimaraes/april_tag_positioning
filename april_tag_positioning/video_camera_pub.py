#!/usr/bin/env python3

import json
import math
import shutil
import subprocess

import cv2
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image


class VideoCameraPublisher(Node):
    FULL_FRAME_WIDTH_MM = 36.0
    FULL_FRAME_HEIGHT_MM = 24.0
    FULL_FRAME_DIAGONAL_MM = math.hypot(FULL_FRAME_WIDTH_MM, FULL_FRAME_HEIGHT_MM)

    def __init__(self):
        super().__init__("video_camera_publisher")

        self.declare_parameter("video_path", "IMG_5994.MOV")
        self.declare_parameter("camera_frame", "camera")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("image_encoding", "mono8")
        self.declare_parameter("rotation_degrees", -1)
        self.declare_parameter("auto_rotate_from_metadata", True)
        self.declare_parameter("auto_intrinsics_from_metadata", True)
        self.declare_parameter("focal_length_35mm_equivalent_mm", 0.0)
        self.declare_parameter("fx", 0.0)
        self.declare_parameter("fy", 0.0)
        self.declare_parameter("cx", -1.0)
        self.declare_parameter("cy", -1.0)
        self.declare_parameter("distortion_model", "plumb_bob")
        self.declare_parameter(
            "distortion_coefficients",
            [0.0, 0.0, 0.0, 0.0, 0.0],
        )

        self.video_path = str(self.get_parameter("video_path").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.fps = float(self.get_parameter("fps").value)
        self.image_encoding = str(self.get_parameter("image_encoding").value)
        self.auto_rotate_from_metadata = bool(
            self.get_parameter("auto_rotate_from_metadata").value
        )
        self.auto_intrinsics_from_metadata = bool(
            self.get_parameter("auto_intrinsics_from_metadata").value
        )
        self.rotation_parameter = int(self.get_parameter("rotation_degrees").value)
        self.fx_override = float(self.get_parameter("fx").value)
        self.fy_override = float(self.get_parameter("fy").value)
        self.cx_override = float(self.get_parameter("cx").value)
        self.cy_override = float(self.get_parameter("cy").value)
        self.distortion_model = str(self.get_parameter("distortion_model").value)
        self.distortion_coefficients = [
            float(value)
            for value in self.get_parameter("distortion_coefficients").value
        ]

        self.video_metadata = self._read_video_metadata(self.video_path)
        self.rotation_degrees = self._resolve_rotation_degrees()
        self.focal_length_35mm_equivalent_mm = (
            self._resolve_focal_length_35mm_equivalent()
        )
        self.metadata_camera_model = self.video_metadata.get("Model", "")
        self.metadata_lens_model = self.video_metadata.get("CameraLensModel", "")

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self.bridge = CvBridge()

        self.image_pub = self.create_publisher(Image, "/camera/image_rect", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", 10)

        self.capture_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.capture_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = self.capture_width
        self.height = self.capture_height
        self.apply_rotation = self.rotation_degrees != 0
        self.first_frame_processed = False

        self.fx = 3000.0
        self.fy = 3000.0
        self.cx = (self.width - 1.0) / 2.0
        self.cy = (self.height - 1.0) / 2.0
        self.intrinsics_source = "fallback placeholder intrinsics"

        self.get_logger().info(f"Opened video: {self.video_path}")
        self.get_logger().info(
            f"Encoded resolution: {self.capture_width}x{self.capture_height}"
        )
        self.get_logger().info(f"Publishing image encoding: {self.image_encoding}")
        if self.metadata_camera_model or self.metadata_lens_model:
            self.get_logger().info(
                "Metadata camera: "
                f"{self.metadata_camera_model or 'unknown'}, "
                f"lens: {self.metadata_lens_model or 'unknown'}"
            )
        if self.focal_length_35mm_equivalent_mm > 0.0:
            self.get_logger().info(
                "Using 35mm-equivalent focal length from metadata/params: "
                f"{self.focal_length_35mm_equivalent_mm:.2f} mm"
            )
        if self.rotation_degrees:
            self.get_logger().info(
                f"Video orientation requires {self.rotation_degrees} degree rotation."
            )

        period = 1.0 / self.fps
        self.timer = self.create_timer(period, self.publish_frame)

    def _read_video_metadata(self, video_path):
        if not shutil.which("exiftool"):
            self.get_logger().warn(
                "exiftool is not available; falling back to manual camera parameters."
            )
            return {}

        command = [
            "exiftool",
            "-j",
            "-n",
            "-Rotation",
            "-ImageWidth",
            "-ImageHeight",
            "-SourceImageWidth",
            "-SourceImageHeight",
            "-CameraFocalLength35mmEquivalent",
            "-CameraLensModel",
            "-Model",
            video_path,
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            self.get_logger().warn(
                "Failed to read video metadata with exiftool; "
                f"falling back to manual camera parameters: {exc}"
            )
            return {}

        if not payload:
            return {}

        return payload[0]

    def _normalize_right_angle(self, rotation_degrees):
        normalized_rotation = rotation_degrees % 360
        if normalized_rotation not in (0, 90, 180, 270):
            self.get_logger().warn(
                "rotation_degrees must be one of 0, 90, 180, 270, or -1 for auto. "
                f"Received {rotation_degrees}; using 0 instead."
            )
            return 0
        return normalized_rotation

    def _resolve_rotation_degrees(self):
        if self.rotation_parameter >= 0:
            return self._normalize_right_angle(self.rotation_parameter)

        if self.auto_rotate_from_metadata:
            metadata_rotation = self.video_metadata.get("Rotation")
            if metadata_rotation is not None:
                return self._normalize_right_angle(int(metadata_rotation))

        return 0

    def _resolve_focal_length_35mm_equivalent(self):
        configured_value = float(
            self.get_parameter("focal_length_35mm_equivalent_mm").value
        )
        if configured_value > 0.0:
            return configured_value

        if self.auto_intrinsics_from_metadata:
            metadata_value = self.video_metadata.get("CameraFocalLength35mmEquivalent")
            if metadata_value is not None:
                return float(metadata_value)

        return 0.0

    def _rotate_dimensions(self, width, height, rotation_degrees):
        if rotation_degrees in (90, 270):
            return height, width
        return width, height

    def _rotate_frame(self, frame):
        if self.rotation_degrees == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation_degrees == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation_degrees == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _encode_frame(self, frame):
        if self.image_encoding == "mono8":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.image_encoding == "bgr8":
            return frame

        self.get_logger().warn(
            f"Unsupported image_encoding '{self.image_encoding}', using bgr8."
        )
        self.image_encoding = "bgr8"
        return frame

    def _configure_output_model(self, output_width, output_height):
        self.width = output_width
        self.height = output_height

        diagonal_pixels = math.hypot(float(output_width), float(output_height))
        auto_focal_pixels = None
        if self.focal_length_35mm_equivalent_mm > 0.0:
            auto_focal_pixels = (
                diagonal_pixels
                * self.focal_length_35mm_equivalent_mm
                / self.FULL_FRAME_DIAGONAL_MM
            )

        if self.fx_override > 0.0:
            self.fx = self.fx_override
        elif auto_focal_pixels is not None:
            self.fx = auto_focal_pixels
        else:
            self.fx = 3000.0

        if self.fy_override > 0.0:
            self.fy = self.fy_override
        elif auto_focal_pixels is not None:
            self.fy = auto_focal_pixels
        else:
            self.fy = 3000.0

        if self.cx_override >= 0.0:
            self.cx = self.cx_override
        else:
            self.cx = (output_width - 1.0) / 2.0

        if self.cy_override >= 0.0:
            self.cy = self.cy_override
        else:
            self.cy = (output_height - 1.0) / 2.0

        if self.fx_override > 0.0 or self.fy_override > 0.0:
            self.intrinsics_source = "manual fx/fy parameters"
        elif auto_focal_pixels is not None:
            self.intrinsics_source = (
                "derived from 35mm-equivalent focal length metadata"
            )
        else:
            self.intrinsics_source = "fallback placeholder intrinsics"

        self.get_logger().info(
            "Publishing frames at "
            f"{self.width}x{self.height} with "
            f"fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"cx={self.cx:.2f}, cy={self.cy:.2f} "
            f"({self.intrinsics_source})."
        )
        if (
            auto_focal_pixels is None
            and self.fx_override <= 0.0
            and self.fy_override <= 0.0
        ):
            self.get_logger().warn(
                "No focal-length metadata or manual fx/fy provided. "
                "Pose estimates will remain approximate until you calibrate the camera."
            )

    def _process_first_frame(self, frame):
        actual_height, actual_width = frame.shape[:2]

        if self.rotation_degrees in (90, 270):
            rotated_width, rotated_height = self._rotate_dimensions(
                self.capture_width,
                self.capture_height,
                self.rotation_degrees,
            )

            if (actual_width, actual_height) == (rotated_width, rotated_height):
                self.apply_rotation = False
                self.get_logger().info(
                    "OpenCV already applied the QuickTime rotation metadata; "
                    "publishing frames as-is."
                )
            elif (actual_width, actual_height) == (
                self.capture_width,
                self.capture_height,
            ):
                self.apply_rotation = True
                self.get_logger().info(
                    f"Applying {self.rotation_degrees} degree rotation "
                    "to match the QuickTime orientation metadata."
                )
            else:
                self.apply_rotation = False
                self.get_logger().warn(
                    "Unexpected decoded frame size "
                    f"{actual_width}x{actual_height}; using frames as-is."
                )

        output_width = actual_width
        output_height = actual_height
        if self.apply_rotation:
            output_width, output_height = self._rotate_dimensions(
                actual_width,
                actual_height,
                self.rotation_degrees,
            )

        self._configure_output_model(output_width, output_height)
        self.first_frame_processed = True

    def make_camera_info(self, stamp):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame

        msg.width = self.width
        msg.height = self.height

        # MOV metadata does not provide a full distortion calibration, so we
        # publish zero distortion unless the user overrides it via parameters.
        msg.distortion_model = self.distortion_model
        msg.d = self.distortion_coefficients

        msg.k = [
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0
        ]

        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        msg.p = [
            self.fx, 0.0, self.cx, 0.0,
            0.0, self.fy, self.cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        return msg

    def publish_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().info("End of video. Restarting.")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        if not self.first_frame_processed:
            self._process_first_frame(frame)

        if self.apply_rotation:
            frame = self._rotate_frame(frame)

        frame = self._encode_frame(frame)

        stamp = self.get_clock().now().to_msg()

        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding=self.image_encoding)
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.camera_frame

        camera_info_msg = self.make_camera_info(stamp)

        self.image_pub.publish(image_msg)
        self.info_pub.publish(camera_info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VideoCameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
