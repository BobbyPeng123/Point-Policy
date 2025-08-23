import numpy as np
import sys
import pickle
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path

from point_utils.correspondence import Correspondence
from point_utils.depth import Depth


class PointsClass:
    def __init__(
        self,
        root_dir,
        dift_path,
        cotracker_checkpoint,
        task_name,
        pixel_keys,
        device,
        width,
        height,
        image_size_multiplier,
        ensemble_size,
        dift_layer,
        dift_steps,
        num_points,
        object_labels,
        use_gt_depth=False,
        **kwargs,
    ):
        """
        Initialize the Points Class for finding key points in the episode.
        [documentation omitted for brevity]
        """

        self.pixel_keys = pixel_keys
        self.device = device
        self.object_labels = object_labels
        # import ipdb; ipdb.set_trace()

        self.tracks = {pixel_key: None for pixel_key in self.pixel_keys}
        if "human_hand" in self.object_labels:
            # Do hand tracking with MediaPipe
            import mediapipe as mp

            mp_hands = mp.solutions.hands
            self.hands = mp_hands.Hands(
                static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
            )
            self.hand_tracks = {pixel_key: None for pixel_key in self.pixel_keys}
            # Remove "human_hand" from object_labels so it is not processed further
            self.object_labels.remove("human_hand")
            self.detect_hand = True
            self.num_hand_points = 9  # wrist + index finger + thumb
        else:
            self.detect_hand = False

        # # Set up the correspondence model and find the expert image features
        # self.correspondence_model = Correspondence(
        #     device,
        #     dift_path,
        #     width,
        #     height,
        #     image_size_multiplier,
        #     ensemble_size,
        #     dift_layer,
        #     dift_steps,
        # )

        # ---------------------------------------------------------------
        # 改动点 1：为每个 object_label 单独实例化一个 Correspondence
        # ---------------------------------------------------------------
        self.correspondence_models: dict[str, Correspondence] = {}
        for _obj in (self.object_labels or []):
            if _obj in ("human_hand", "empty_space"):
                continue
            self.correspondence_models[_obj] = Correspondence(
                device,
                dift_path,
                width,
                height,
                image_size_multiplier,
                ensemble_size,
                dift_layer,
                dift_steps,
            )

        # import ipdb; ipdb.set_trace()

        self.initial_coords, self.expert_correspondence_features = {}, {}

        # -------- 新：全局共享目录（所有 task 共用） --------
        objects_dir = Path(root_dir) / "coordinates" / "objects"
        # -------- 旧：按 task 的回退目录 --------
        images_dir  = Path(root_dir) / "coordinates" / task_name / "images"
        coords_dir  = Path(root_dir) / "coordinates" / task_name / "coords"

        def _pick_first(*cands: Path):
            for p in cands:
                if p.exists():
                    return p
            return None
        
        # def _harmonize_coords_to_image(expert_img: Image.Image, coords_arr: np.ndarray) -> np.ndarray:
        #     img_h, img_w = expert_img.size  # PIL: (W, H)
        #     coords = np.array(coords_arr, dtype=np.float32).copy()  # [N, 3] -> [layer, y, x]
        #     y, x = coords[:, 1], coords[:, 2]

        #     # 1) 如果是归一化坐标 [0,1]，映射到像素
        #     if (y.min() >= -1e-6 and y.max() <= 1.01) and (x.min() >= -1e-6 and x.max() <= 1.01):
        #         y = y * (img_h - 1)
        #         x = x * (img_w - 1)

        #     # 2) 如果明显大于图像尺寸（来自另一个分辨率的像素标注），按最大值等比缩回图像范围
        #     if y.max() >= img_h or x.max() >= img_w:
        #         # 等比各向缩放到当前 expert_img 边界内
        #         scale_y = (img_h - 1e-6) / max(y.max(), 1e-6)
        #         scale_x = (img_w - 1e-6) / max(x.max(), 1e-6)
        #         y = y * scale_y
        #         x = x * scale_x

        #     # 3) 最后裁剪 & 取整到像素索引
        #     y = np.clip(np.round(y), 0, img_h - 1)
        #     x = np.clip(np.round(x), 0, img_w - 1)

        #     coords[:, 1] = y
        #     coords[:, 2] = x
        #     return coords



        # 预读共享 objects（若任何对象缺文件，则整体回退旧结构）
        use_shared = objects_dir.exists()
        shared_imgs: dict[str, Image.Image] = {}
        shared_coords: dict[str, np.ndarray] = {}
        if use_shared and (self.object_labels or []):
            for obj in self.object_labels:
                if obj in ("empty_space", "human_hand"):
                    continue
                img_p = _pick_first(
                    objects_dir / f"{obj}.png",
                    objects_dir / f"{obj}.jpg",
                    objects_dir / f"{obj}.jpeg",
                )
                pkl_p = objects_dir / f"{obj}.pkl"
                if img_p is None or (not pkl_p.exists()):
                    use_shared = False
                    break
                shared_imgs[obj]   = Image.open(img_p).convert("RGB")
                shared_coords[obj] = np.array(pickle.load(open(pkl_p, "rb")))

        # 为每个 pixel_key × object_label 建 key
        for pixel_key in self.pixel_keys:
            default_img = Image.new("RGB", (224, 224), (0, 0, 0))
            for object_label in (self.object_labels or []):
                key = f"{pixel_key}_{object_label}"

                if object_label == "human_hand":
                    continue  # 由 MediaPipe 处理
                if object_label == "empty_space":
                    self.initial_coords[key] = np.array([[0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 0.0]])
                    continue

                # ① 优先：全局共享 objects/
                if use_shared and (object_label in shared_imgs):
                    coords_arr = shared_coords[object_label]
                    expert_img = shared_imgs[object_label]
                else:
                    # ② 回退：老结构（每相机一张图/坐标）
                    img_p = _pick_first(
                        images_dir / f"{pixel_key}.png",
                        images_dir / f"{pixel_key}.jpg",
                        images_dir / f"{pixel_key}.jpeg",
                    )
                    expert_img = Image.open(img_p).convert("RGB") if img_p else default_img
                    pkl_p = coords_dir / f"{pixel_key}_{object_label}.pkl"
                    if not pkl_p.exists():
                        raise FileNotFoundError(
                            f"Missing coords: {pkl_p} (and objects/{object_label}.pkl not found)"
                        )
                    coords_arr = np.array(pickle.load(open(pkl_p, "rb")))

                # coords = _harmonize_coords_to_image(expert_img, coords_arr)
                self.initial_coords[key] = coords_arr

                # -------------------------------------------------------
                # 改动点 2：调用每个对象自己的 Correspondence 实例
                # 并存储独立副本，避免后续被覆盖
                # -------------------------------------------------------
                if object_label not in ("human_hand", "empty_space"):
                    with torch.no_grad():
                        feat = self.correspondence_models[object_label] \
                            .set_expert_correspondence(
                                expert_img, pixel_key, object_label
                            )
                    self.expert_correspondence_features[key] = feat.detach().clone()

                # import ipdb; ipdb.set_trace()

                # with torch.no_grad():
                #     # self.expert_correspondence_features[key] = (
                #     #     self.correspondence_model.set_expert_correspondence(
                #     #         expert_img, pixel_key, object_label
                #     #     )
                #     # )
                #     feat = self.correspondence_model.set_expert_correspondence(
                #         expert_img, pixel_key, object_label
                #     )
                #     # save expert_img
                #     # create path if not exists
                #     import os
                #     os.makedirs("expert_images", exist_ok=True)
                #     expert_img.save(f"expert_images/{pixel_key}_{object_label}.png")
                #     print(f"expert_images/{pixel_key}_{object_label}.png saved.")
                #     print(self.expert_correspondence_features[key].shape)

                # self.expert_correspondence_features[key] = feat.detach().clone()
                # import ipdb; ipdb.set_trace()


                # self.initial_coords[key] = coords_arr
# #                 ipdb> coords_arr
# # array([[  0.        , 102.01612707,  89.41602476],
# #        [  0.        , 181.98924371,  69.0865715 ],
# #        [  0.        ,  98.65591208, 155.61275686],
# #        [  0.        , 174.93279224, 134.44324355]])
#                 with torch.no_grad():
#                     self.expert_correspondence_features[key] = (
#                         self.correspondence_model.set_expert_correspondence(
#                             expert_img, pixel_key, object_label
#                         )
#                     )
#                     # import ipdb; ipdb.set_trace()
#                     # 假设就在 PointsClass.__init__ 内、紧跟 set_expert_correspondence 之后
#                     eft = self.expert_correspondence_features[key]   # src_ft: [1, C, H, W] 通常
#                     # ipdb> eft.shape
#                     # torch.Size([1, 1280, 11, 13])
#                     H, W = int(eft.shape[-2]), int(eft.shape[-1])    # 特征图尺寸
#                     # ipdb> H, W
#                     # (11, 13)
#                     img_w, img_h = expert_img.size                   # 参考图像尺寸（注意 PIL 是 (W,H)）
#                     # ipdb> expert_img.size
#                     # (200, 179)

#                     coords = np.array(coords_arr, dtype=np.float32).copy()  # [N, 3] => [layer, y, x]

#                     # 如果坐标是归一化 [0,1]，映射到特征图；如果是像素坐标，按参考图→特征图线性缩放
#                     y = coords[:, 1]
#                     x = coords[:, 2]
#                     if (0.0 <= y).all() and (y <= 1.01).all() and (0.0 <= x).all() and (x <= 1.01).all():
#                         y = y * H
#                         x = x * W
#                     else:
#                         # 像素坐标 -> 特征图坐标
#                         y = y * (H / max(1, img_h))
#                         x = x * (W / max(1, img_w))

#                     # 最后做一次健壮性截断与取整（避免 200 这样的上边界越界）
#                     y = np.clip(np.round(y), 0, H - 1)
#                     x = np.clip(np.round(x), 0, W - 1)

#                     coords[:, 1] = y
#                     coords[:, 2] = x
#                     # ipdb> coords
#                     # np.array([[ 0.,  6.,  6.],
#                     #     [ 0., 10.,  4.],
#                     #     [ 0.,  6., 10.],
#                     #     [ 0., 10.,  9.]], dtype=float32)

#                     self.initial_coords[key] = coords


        # self.initial_coords, self.expert_correspondence_features = {}, {}

        # for pixel_key in self.pixel_keys:
        #     # If the only object label is "empty_space", create a blank expert image.
        #     if self.object_labels == ["empty_space"] or len(self.object_labels) == 0:
        #         expert_image = Image.new("RGB", (224, 224), (0, 0, 0))  # black blank image
        #     else:
        #         expert_image = Image.open(
        #             "%s/coordinates/%s/images/%s.png" % (root_dir, task_name, pixel_key)
        #         ).convert("RGB")

        #     if len(self.object_labels) > 0:
        #         for object_label in self.object_labels:
        #             key = f"{pixel_key}_{object_label}"
        #             if object_label == "empty_space":
        #                 # Skip loading coordinates and computing expert features.
        #                 # The semantic_similar_points for empty_space will be set externally.
        #                 self.initial_coords[key] = np.array([
        #                     [0.0, 0.0, 0.0],
        #                     [0.0, 0.0, 0.0],
        #                 ])
        #                 continue
        #             else:
        #                 self.initial_coords[key] = np.array(
        #                     pickle.load(
        #                         open(
        #                             "%s/coordinates/%s/coords/%s_%s.pkl"
        #                             % (root_dir, task_name, pixel_key, object_label),
        #                             "rb",
        #                         )
        #                     )
        #                 )
        #                 # import ipdb; ipdb.set_trace()
        #                 with torch.no_grad():
        #                     self.expert_correspondence_features[key] = \
        #                         self.correspondence_model.set_expert_correspondence(
        #                             expert_image, pixel_key, object_label
        #                         )

        # Initialize semantic similar points for every (pixel_key, object_label) combination.
        # For 'empty_space', this remains None (or can be set externally later).
        self.semantic_similar_points = {
            f"{pixel_key}_{object_label}": None
            for pixel_key in self.pixel_keys
            for object_label in self.object_labels
        }

        # Set up the depth model
        self.depth_model = Depth("/home/bobby/Point-Policy/Depth-Anything-V2", device)

        # Set up cotracker
        sys.path.append(root_dir + "/co-tracker/")
        from cotracker.predictor import CoTrackerOnlinePredictor

        self.cotracker = {}
        for pixel_key in self.pixel_keys:
            self.cotracker[pixel_key] = CoTrackerOnlinePredictor(
                checkpoint=cotracker_checkpoint,
                window_len=16,
            ).to(device)

        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.image_list = {
            f"{pixel_key}": torch.tensor([]).to(self.device)
            for pixel_key in self.pixel_keys
        }
        self.depth = {
            f"{pixel_key}": torch.tensor([]).to(self.device)
            for pixel_key in self.pixel_keys
        }
        

        if num_points == -1:
            self.num_points = 0 if not self.detect_hand else self.num_hand_points
            if len(self.object_labels) > 0:
                for object_label in self.object_labels:
                    key = f"{self.pixel_keys[0]}_{object_label}"
                    self.num_points += self.initial_coords.get(key, np.array([])).shape[0]
        else:
            self.num_points = num_points

        self.device = device
        # import ipdb; ipdb.set_trace()

        # in case image is cropped and resized
        self.original_image_size = None
        self.current_image_size = None
        self.crop_ratios = None

 # Image passed in here must be in RGB format
    def add_to_image_list(self, image, pixel_key):
        """
        Add an image to the image list for finding key points.

        Parameters:
        -----------
        image : np.ndarray
            The image to add to the image list. This image must be in RGB format.
        """

        key = f"{pixel_key}"

        transformed = (
            torch.from_numpy(image.astype(np.uint8)).permute(2, 0, 1).float() / 255
        )

        # We only want to track the last 16 images so pop the first one off if we have more than 16
        if self.image_list[key].shape[0] > 0 and self.image_list[key].shape[1] == 16:
            self.image_list[key] = self.image_list[key][:, 1:]

        # If it is the first image you want to repeat until the whole array is full
        # Otherwise it will just add the new image to the end of the array
        while self.image_list[key].shape[0] == 0 or self.image_list[key].shape[1] < 16:
            self.image_list[key] = torch.cat(
                (
                    self.image_list[key],
                    transformed.unsqueeze(0).unsqueeze(0).clone().to(self.device),
                ),
                dim=1,
            )

    def reset_episode(self):
        """
        Reset the image list for finding key points.
        """

        self.image_list = {
            f"{pixel_key}": torch.tensor([]).to(self.device)
            for pixel_key in self.pixel_keys
        }
        self.depth = {
            f"{pixel_key}": torch.tensor([]).to(self.device)
            for pixel_key in self.pixel_keys
        }
        self.tracks = {pixel_key: None for pixel_key in self.pixel_keys}
        self.hand_tracks = {pixel_key: None for pixel_key in self.pixel_keys}

    def find_semantic_similar_points(self, pixel_key, object_label="", object_bbox=None):
        """
        Find the semantic similar points between the expert image and the current image.
        """
        # Skip processing for 'human_hand' and 'empty_space'
        # import ipdb; ipdb.set_trace()
        if object_label == "human_hand":
            return
        if object_label == "empty_space":
            # resized_coords = self.semantic_similar_points[key].clone()
            # resized_coords[:, 1] *= self.width / self.original_size[0]   # x scale
            # resized_coords[:, 2] *= self.height / self.original_size[1]  # y scale

            # self.semantic_similar_points[key] = resized_coords
            return

        key = f"{pixel_key}_{object_label}"

        # ---------------------------------------------------------------
        # 改动点 3：按 object_label 取对应的模型来做匹配
        # ---------------------------------------------------------------
        corr = self.correspondence_models.get(object_label, None)
        if corr is None:
            # human_hand / empty_space 已在前面 return；若出现缺模型的 label，直接跳过更安全
            return
        self.semantic_similar_points[key] = corr.find_correspondence(
                self.expert_correspondence_features[key],
                self.image_list[pixel_key][0, -1],
                self.initial_coords[key],
                pixel_key,
                object_label,
                object_bbox,
        )

        # import ipdb; ipdb.set_trace()
#         self.semantic_similar_points[key] = self.correspondence_model.find_correspondence(
#             self.expert_correspondence_features[key],
#             self.image_list[pixel_key][0, -1],
#             self.initial_coords[key],
#             pixel_key,
#             object_label,
#             object_bbox,
#         )
# #         ipdb> self.expert_correspondence_features['pixels4_oven'].shape
# # torch.Size([1, 1280, 11, 13])
# # self.expert_correspondence_features['pixels4_bowl'].shape
# # torch.Size([1, 1280, 11, 13])

# # self.expert_correspondence_features['pixels4_oven'].shape
# # torch.Size([1, 1280, 6, 7])
# # ipdb> self.expert_correspondence_features['pixels4_bowl'].shape
# # torch.Size([1, 1280, 6, 7])
    def get_depth(self, pixel_key, original_image_size=None, current_image_size=None, crop_ratios=None, last_n_frames=1):
        """
        Get the depth map for the current image using Depth Anything. Depth is height x width.

        Parameters:
        -----------
        last_n_frames : int
            The number of frames to look back in the episode
        """
        # import ipdb; ipdb.set_trace()
        key = f"{pixel_key}"
        self.original_image_size = original_image_size
        self.current_image_size = current_image_size
        self.crop_ratios = crop_ratios

        self.depth[key] = np.zeros(
            (
                last_n_frames,
                self.image_list[key].shape[3],
                self.image_list[key].shape[4],
            )
        )
        for frame_num in range(last_n_frames):
            frame_idx = -1 * (last_n_frames - frame_num)
            numpy_image = (
                self.image_list[key][0, frame_idx].cpu().numpy().transpose(1, 2, 0)
                * 255
            )
            depth = self.depth_model.get_depth(numpy_image)
            self.depth[key][frame_idx] = depth

    def set_depth(
        self,
        depth,
        pixel_key,
        original_image_size=None,
        current_image_size=None,
        crop_ratios=None,
    ):
        """
        If you are using ground truth depth, you can set the depth here.

        Parameters:
        -----------
        depth : np.ndarray
            The depth map for the current image. Depth is height x width.
        original_image_size : tuple
            The original size of the image before it was cropped and resize.
        current_image_size : tuple
            The current size of the image.
        crop_ratios : tuple -> ((float, float), (float, float)) - (crop_h, crop_w)
            The crop ratios used to crop the image.
        """
        key = f"{pixel_key}"
        self.original_image_size = original_image_size
        self.current_image_size = current_image_size
        self.crop_ratios = crop_ratios

        if self.depth[key].shape[0] == 8:
            self.depth[key] = self.depth[key][1:]

        while self.depth[key].shape[0] < 8:
            if self.depth[key].shape[0] == 0:
                self.depth[key] = depth[None, ...].copy()
            self.depth[key] = np.concatenate(
                (self.depth[key], depth[None, ...].copy()), axis=0
            )

    def track_points(
        self, pixel_key, last_n_frames=1, is_first_step=False, one_frame=True
    ):
        """
        Track the key points in the current image using the CoTracker model.

        Parameters:
        -----------
        is_first_step : bool
            Whether or not this is the first step in the episode.
        """

        if self.detect_hand:
            hand_tracks = self.track_points_hand(pixel_key)
            hand_tracks = torch.tensor(hand_tracks)

            if not is_first_step:
                if self.hand_tracks[pixel_key] is None:
                    self.hand_tracks[pixel_key] = hand_tracks[None]
                else:
                    hand_tracks = hand_tracks[-last_n_frames:]
                    self.hand_tracks[pixel_key] = torch.cat(
                        [
                            self.hand_tracks[pixel_key],
                            hand_tracks[None].to(self.hand_tracks[pixel_key].device),
                        ],
                        dim=1,
                    )

        if len(self.object_labels) > 0:
            # import ipdb; ipdb.set_trace()
            if is_first_step:
                semantic_similar_points = []
                # import ipdb; ipdb.set_trace()
                for object_label in self.object_labels:
                    semantic_similar_points.append(
                        self.semantic_similar_points[f"{pixel_key}_{object_label}"]
                    )
                semantic_similar_points = torch.cat(semantic_similar_points, dim=0)
                # import ipdb; ipdb.set_trace()

                self.cotracker[pixel_key](
                    video_chunk=self.image_list[pixel_key][0, 0]
                    .unsqueeze(0)
                    .unsqueeze(0),
                    is_first_step=True,
                    add_support_grid=True,
                    # add_support_grid=False,
                    queries=semantic_similar_points[None].to(self.device),
                )
                self.tracks[pixel_key] = semantic_similar_points
            else:
                tracks, _ = self.cotracker[pixel_key](
                    self.image_list[pixel_key], one_frame=one_frame
                )
                # Remove the support points
                tracks = tracks[:, :, 0 : self.num_points, :]
                # import ipdb; ipdb.set_trace()

                if self.detect_hand:
                    self.hand_tracks[pixel_key] = self.hand_tracks[pixel_key].to(
                        tracks.device
                    )
                    self.tracks[pixel_key] = torch.cat(
                        [self.hand_tracks[pixel_key], tracks], dim=-2
                    )
                else:
                    self.tracks[pixel_key] = tracks.clone()
        else:
            self.tracks[pixel_key] = self.hand_tracks[pixel_key]

    def track_points_hand(self, pixel_key):
        frames = (
            self.image_list[pixel_key][0].cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        frames = frames.astype(np.uint8)

        hand_tracks = []
        for frame in frames:
            results = self.hands.process(frame)
            if results.multi_hand_landmarks is not None:
                hand_track = []
                for hand_landmarks in results.multi_hand_landmarks:
                    # Wrist landmarks: 0
                    # Index finger landmarks: 5, 6, 7, 8
                    # Thumb landmarks: 1, 2, 3, 4
                    wrist_landmark = hand_landmarks.landmark[0]
                    index_finger_landmarks = [
                        hand_landmarks.landmark[i] for i in [5, 6, 7, 8]
                    ]
                    thumb_landmarks = [hand_landmarks.landmark[i] for i in [1, 2, 3, 4]]

                    # Draw wrist
                    x = int(wrist_landmark.x * frame.shape[1])
                    y = int(wrist_landmark.y * frame.shape[0])
                    hand_track.append([x, y])

                    # Draw index finger
                    for landmark in index_finger_landmarks:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        hand_track.append([x, y])

                    # Draw thumb
                    for landmark in thumb_landmarks:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        hand_track.append([x, y])

                hand_track = np.array(hand_track)
                hand_tracks.append(hand_track)
            else:
                hand_tracks.append(
                    hand_tracks[-1]
                    if len(hand_tracks) > 0
                    else self.hand_tracks[pixel_key][0, -1].cpu()
                )
        return np.array(hand_tracks)

    def get_points(self, pixel_key, last_n_frames=1):
        """
        Get the list of points for the current frame.

        Parameters:
        -----------
        last_n_frames : int
            The number of frames to look back in the episode.

        Returns:
        --------
        final_points : torch.Tensor
            The list of points for the current frame.
        """

        final_points = torch.zeros((last_n_frames, self.num_points, 3))

        for frame_num in range(last_n_frames):
            for point in range(self.num_points):
                frame_idx = -1 * (last_n_frames - frame_num)
                # try:
                if self.original_image_size is None:
                    depth = self.depth[pixel_key][
                        frame_idx,
                        int(self.tracks[pixel_key][0, frame_idx, point][1]),
                        int(self.tracks[pixel_key][0, frame_idx, point][0]),
                    ]
                else:
                    crop_h, crop_w = self.crop_ratios
                    w_orig, h_orig = self.original_image_size
                    w_curr, h_curr = self.current_image_size

                    # compute point_h in original image
                    point_h = self.tracks[pixel_key][0, frame_idx, point][1]
                    h_orig_cropped = h_orig * (crop_h[1] - crop_h[0])
                    point_h_orig = int(
                        (point_h / h_curr) * h_orig_cropped + h_orig * crop_h[0]
                    )

                    # compute point_w in original image
                    point_w = int(self.tracks[pixel_key][0, frame_idx, point][0])
                    w_orig_cropped = w_orig * (crop_w[1] - crop_w[0])
                    point_w_orig = int(
                        (point_w / w_curr) * w_orig_cropped + w_orig * crop_w[0]
                    )
                    # import ipdb; ipdb.set_trace()
                    depth = self.depth[pixel_key][frame_idx, point_h_orig, point_w_orig]

                x = self.tracks[pixel_key][0, frame_idx, point][0]
                y = self.tracks[pixel_key][0, frame_idx, point][1]

                final_points[frame_num, point] = torch.tensor([x, y, depth])

        return final_points

    def get_points_on_image(self, pixel_key, last_n_frames=1):
        """
        Get the list of points for the current frame in pixel space.

        Parameters:
        -----------
        last_n_frames : int
            The number of frames to look back in the episode.

        Returns:
        --------
        final_points : torch.Tensor
            The list of points for the current frame.
        """

        final_points = torch.zeros((last_n_frames, self.num_points, 2))

        for frame_num in range(last_n_frames):
            for point in range(self.num_points):
                frame_idx = -1 * (last_n_frames - frame_num)

                x = self.tracks[pixel_key][0, frame_idx, point][0]
                y = self.tracks[pixel_key][0, frame_idx, point][1]
                final_points[frame_num, point] = torch.tensor([x, y])

        return final_points

    def plot_image(self, pixel_key, last_n_frames=1):
        # import ipdb; ipdb.set_trace()
        """
        Plot the image with the key points overlaid on top of it. Running this will slow down your tracking, but it's good for debugging.

        Parameters:
        -----------
        last_n_frames : int
            The number of frames to look back in the episode.

        Returns:
        --------
        img_list : list
            A list of images with the key points overlaid on top of them.
        """

        img_list = []

        for frame_num in range(last_n_frames):
            frame_idx = -1 * (last_n_frames - frame_num)
            curr_image = (
                self.image_list[pixel_key][0, frame_idx]
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                * 255
            )

            fig, ax = plt.subplots(1)
            ax.imshow(curr_image.astype(np.uint8))
            ax.axis("off")  # Hide the axes before saving and displaying

            rainbow = plt.get_cmap("rainbow")
            # Generate n evenly spaced colors from the colormap
            colors = [
                rainbow(i / self.tracks[pixel_key].shape[2])
                for i in range(self.tracks[pixel_key].shape[2])
            ]

            for idx, coord in enumerate(self.tracks[pixel_key][0, frame_idx]):
                ax.add_patch(
                    patches.Circle(
                        (coord[0].cpu(), coord[1].cpu()),
                        5,
                        facecolor=colors[idx],
                        edgecolor="black",
                    )
                )

            fig.canvas.draw()

            # Get buffer and size
            buffer, (width, height) = fig.canvas.print_to_buffer()
            img = np.frombuffer(buffer, dtype=np.uint8)

            # Handle potential RGBA buffer (4 channels) by converting to RGB
            expected_size = width * height * 4  # 4 channels per pixel (RGBA)
            if img.size == expected_size:
                img = img.reshape((height, width, 4))[:, :, :3]  # Drop alpha channel
            else:
                img = img.reshape((height, width, 3))  # If already RGB

            img_list.append(img.copy())

            # Save image for debugging
            import time
            plt.savefig(
                f"/home/bobby/Point-Policy/point_policy/debug_image/{pixel_key}_{time.time()}.png",
                bbox_inches="tight",
                pad_inches=0,
            )

            plt.close()

        return img_list