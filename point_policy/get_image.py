# get_image_franka.py
"""Capture one RGB frame from each configured Franka camera and save to disk.

Usage examples
--------------
# Capture from default camera 0 and save to current directory
python get_image_franka.py

# Capture from cameras 0, 2, and 3 and save into ./captures
python get_image_franka.py --cam_ids 0 2 3 --save_dir ./captures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2

from frankateach.network import ZMQCameraSubscriber
from frankateach.constants import HOST, CAM_PORT


class CameraCaptureFranka:
    """Utility to grab a single RGB frame from one or more Franka cameras."""

    def __init__(self, cam_ids: List[int]):
        self.cam_ids = cam_ids
        self.image_subscribers: dict[int, ZMQCameraSubscriber] = {}

        for cam_id in self.cam_ids:
            port = CAM_PORT + cam_id  # Each camera gets its own port offset
            self.image_subscribers[cam_id] = ZMQCameraSubscriber(
                host=HOST,
                port=port,
                topic_type="RGB",
            )

    def capture_images(self) -> List[Tuple[int, "np.ndarray"]]:
        """Capture one RGB frame per camera.

        Returns
        -------
        List[Tuple[int, np.ndarray]]
            A list of (cam_id, image) tuples in the same order as *cam_ids*.
        """
        image_list = []
        for cam_id in self.cam_ids:
            subscriber = self.image_subscribers[cam_id]
            print(f"Capturing image from camera {cam_id} (port {CAM_PORT + cam_id})")
            rgb_image, _timestamp = subscriber.recv_rgb_image()
            image_list.append((cam_id, rgb_image))
        return image_list

    def close(self) -> None:
        """Stop all subscribers cleanly."""
        for subscriber in self.image_subscribers.values():
            subscriber.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture RGB images from Franka cameras and write them to disk."
    )
    parser.add_argument(
        "--cam_ids",
        nargs="+",
        type=int,
        default=[0],
        help="List of camera IDs to capture (default: 0)",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("."),
        help="Directory in which to save captured images (default: current directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure the output directory exists
    args.save_dir.mkdir(parents=True, exist_ok=True)

    capturer = CameraCaptureFranka(cam_ids=args.cam_ids)
    try:
        images = capturer.capture_images()
        for cam_id, img in images:
            filename = args.save_dir / f"camera_{cam_id}.jpg"
            cv2.imwrite(str(filename), img)
            print(f"Saved image from camera {cam_id} to {filename}")
    finally:
        capturer.close()


if __name__ == "__main__":
    main()



# import cv2
# import numpy as np
# from openteach.utils.network import ZMQCameraSubscriber
# from xarm_env.envs.constants import HOST_ADDRESS, CAMERA_PORT_OFFSET, CAM_SERIAL_NUMS

# class CameraCapture:
#     def __init__(self):     
#         # Camera subscribers
#         self.image_subscribers = []
#         for cam_idx in list(CAM_SERIAL_NUMS.keys()):
#             port = CAMERA_PORT_OFFSET + cam_idx
#             self.image_subscribers.append(
#                 ZMQCameraSubscriber(
#                     host=HOST_ADDRESS,
#                     port=port,
#                     topic_type="RGB",
#                 )
#             )
    
#     def capture_images(self):
#         """Capture images from all configured cameras."""
#         image_list = []
#         for idx, subscriber in enumerate(self.image_subscribers):
#             print(f"Capturing image from camera {idx}")
#             image = subscriber.recv_rgb_image()[0]
#             resized_image = image
#             image_list.append(resized_image)
#         return image_list

# if __name__ == "__main__":
#     camera_capture = CameraCapture()
#     images = camera_capture.capture_images()

#     # save images
#     for idx, image in enumerate(images):
#         cv2.imwrite(f"camera_{idx}.jpg", image)
#         print(f"Saved image {idx} to camera_{idx}.jpg")