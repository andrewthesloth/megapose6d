import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from pathlib import Path
from typing import List, Tuple, Union
from tqdm import tqdm
# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R
import time
from copy import deepcopy
# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay
# from megapose.scripts.load_files import loader
# from megapose.ngp_renderer.ngp_render_api import ngp_render
from matplotlib.path import Path
from ultralytics import RTDETR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class MegaposeInference:
    def __init__(self, path_to_models: str, path_to_weights: str,
                 megapose_model_name: str = "megapose-1.0-RGB-multi-hypothesis", image_resize_ratio: int = 4):
        self.num_objects = 7
        self.path2models = path_to_models
        self.megapose_model_name = megapose_model_name

        self.object_database = self.create_object_database()
        self.pose_estimators = self.create_pose_estimators()
        self.detector = RTDETR(path_to_weights)

        self.image_resize_ratio = image_resize_ratio
        # self.static_mask = self.create_static_mask()

    def create_object_database(self) -> RigidObjectDataset:
        object_database = {}
        mesh_units = "mm"

        for obj_idx in tqdm(range(1, self.num_objects + 1), desc="Creating object database"):
            obj_label = str(obj_idx).zfill(6)
            mesh_path = os.path.join(self.path2models, f"obj_{obj_label}.ply")

            rigid_object_dataset = RigidObjectDataset(
                [RigidObject(label=obj_label, mesh_path=mesh_path, mesh_units=mesh_units)])
            object_database[obj_label] = rigid_object_dataset

        return object_database

    def create_pose_estimators(self):
        pose_estimators = {}
        for obj_label in tqdm(self.object_database.keys(), desc="Creating pose estimators"):
            model_name = f"megapose-1.0-RGB-multi-hypothesis"
            pose_estimator = load_named_model(
                model_name, self.object_database[obj_label]
            ).cuda()
            pose_estimators[obj_label] = pose_estimator

        return pose_estimators

    def inference_detection(self, rgb: np.ndarray) -> DetectionsType:
        predictions = self.detector(rgb)
        for pred in predictions:
            # print(pred.boxes)
            boxes = pred.boxes.xywh.detach().cpu().numpy()
            classes = pred.boxes.cls.detach().cpu().numpy() + 1
            confidences = pred.boxes.conf.detach().cpu().numpy()

            boxes = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], axis=-1)
            boxes = boxes.astype(int).tolist()
            classes = classes.astype(int).tolist()
            confidences = confidences.tolist()

        detection_data = []
        for i in range(len(boxes)):
            if confidences[i] > 0.5:
                bbox_center = [(boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2]
                # print(bbox_center)
                # if bbox_center[0] > 100 and bbox_center[1] > 100:
                detection_data.append(
                    {"label": str(classes[i]).zfill(6),
                        "bbox_modal": boxes[i],
                        "confidence": confidences[i]})

        return detection_data

    def inference_pose_esimtation(self, observation, detection, obj_id):
        model_info = NAMED_MODELS[self.megapose_model_name]

        output, _ = self.pose_estimators[obj_id].run_inference_pipeline(
            observation, detections=detection, **model_info["inference_parameters"]
        )
        label = output.infos["label"][0]
        pose_pred = output.poses.cpu().numpy()[0]
        return pose_pred, label

    def get_visualization(self, image, labels, pose, object_dataset, cam_data_json):
        object_data = [ObjectData(label=labels, TWO=Transform(pose))]
        camera_data = CameraData(K=cam_data_json["K"], resolution=cam_data_json["resolution"], TWC=Transform(np.eye(4)))
        renderer = Panda3dSceneRenderer(object_dataset)

        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=(1.0, 1.0, 1.0, 1),
            ),
        ]

        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_data)
        renderings = renderer.render_scene(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=False,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
            clear=True
        )[0]

        rendered_image = renderings.rgb
        rendered_image = rendered_image  # .transpose(1, 0, 2)
        rendered_object_mask = rendered_image > 0
        overlay = image * (1 - rendered_object_mask) + rendered_image

        return np.ascontiguousarray(overlay).astype(np.uint8)

    def inference_main(self, rgb_undistorted, cam_data_undistorted, ret_dict: dict):
        rgb_undistorted = rgb_undistorted * MegaposeInference.create_static_mask(rgb_undistorted)
        matrix_K = np.array(cam_data_undistorted["K"])
        camera_res = cam_data_undistorted["resolution"][::-1]

        resolution_resized = (int(camera_res[0] / self.image_resize_ratio),
                              int(camera_res[1] / self.image_resize_ratio))

        rgb_undistorted = cv2.resize(rgb_undistorted, resolution_resized)
        rgb_undistorted = cv2.cvtColor(rgb_undistorted, cv2.COLOR_BGR2RGB)
        rgb_orig = deepcopy(rgb_undistorted)

        matrix_K /= self.image_resize_ratio
        matrix_K[-1, -1] = 1

        observation = ObservationTensor.from_numpy(rgb_undistorted, None, matrix_K).cuda()

        detections = self.inference_detection(rgb_undistorted)
        ret_dict["poses"] = {}
        ret_dict["rgb_vis"] = {}

        for detection in detections:
            obj_id = detection["label"]
            bbox = detection["bbox_modal"]
            confidence = detection["confidence"]

            object_data = [{"label": str(obj_id).zfill(6), "bbox_modal": bbox}]
            object_data = [ObjectData.from_json(d) for d in object_data]
            detection = make_detections_from_object_data(object_data).cuda()

            pose, label = self.inference_pose_esimtation(observation, detection, obj_id)

            ret_dict["poses"][obj_id] = pose

            rgb_undistorted = self.get_visualization(rgb_undistorted, label, pose, self.object_database[obj_id], {
                "K": matrix_K, "resolution": resolution_resized[::-1]})

            rgb_undistorted = cv2.rectangle(rgb_undistorted, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            rgb_vis = np.hstack([rgb_orig, rgb_undistorted])
            rgb_vis = np.hstack([rgb_vis, rgb_undistorted * 0.5 + rgb_orig * 0.5])
            cv2.putText(rgb_undistorted, str(int(label)) + " " + str(confidence),
                        (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret_dict["rgb_undistorted"] = rgb_undistorted
            ret_dict["rgb_vis"][obj_id] = rgb_vis

        # return pose, label, rgb_undistorted, rgb_vis

    @staticmethod
    def create_static_mask(rgb):
        res_dist = (2048, 2448)
        res_undist = rgb.shape[:2]
        ratio_w = res_undist[1] / res_dist[1]
        ratio_h = res_undist[0] / res_dist[0]
        ratio = np.array([ratio_w, ratio_h])

        point_left_top = np.array([711, 263]) * ratio
        point_right_top = np.array([2397, 182]) * ratio

        point_right_bottom = np.array([2439, 2039]) * ratio
        point_left_bottom = np.array([255, 1847]) * ratio

        polygon = Path([point_left_top.astype(int), point_left_bottom.astype(int),
                        point_right_bottom.astype(int), point_right_top.astype(int)])
        mask = np.zeros((rgb.shape[0], rgb.shape[1], 1))

        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                # Check if the pixel (j, i) is inside the polygon
                if polygon.contains_point((j, i)):
                    mask[i, j] = 1  # Inside the polygon

        return mask.astype(np.uint8)


if __name__ == "__main__":
    # load img
    img = cv2.imread(
        "/home/testbed/.local/share/ov/pkg/isaac-sim-4.2.0/user_examples/kuka_r2e_omniverse/data/physical/calibration/2024-10-02-21-25-31.png")
    # mask = cv2.imread(
    #     "/home/testbed/.local/share/ov/pkg/isaac-sim-4.2.0/user_examples/kuka_r2e_omniverse/data/physical/calibration/test_mask.png")
    mask = create_static_mask(img)
    cv2.imwrite(
        "/home/testbed/.local/share/ov/pkg/isaac-sim-4.2.0/user_examples/kuka_r2e_omniverse/data/physical/calibration/test_mask.png",
        img *
        mask)
    cv2.imshow(
        "/home/testbed/.local/share/ov/pkg/isaac-sim-4.2.0/user_examples/kuka_r2e_omniverse/data/physical/calibration/test_mask.png",
        img *
        mask)
#     megapose_inference = MegaposeInference()
#     for img_idx in range(1, 75):
#         try:
#             rgb = cv2.imread(
#                 f"/home/shareduser/Projects/megapose6d/src/megapose/scripts/MSV_DATA/FramesKukaEducate2/image_{img_idx}.png")
#             megapose_inference.inference_main(rgb, img_idx)
#         except BaseException:
#             continue
