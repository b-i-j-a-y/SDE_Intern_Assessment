import glob
import os
import shutil
import time
from random import randint

import cv2
import numpy as np
import torch
from PIL import Image

from densepose import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.densepose_results import (
    DensePoseResultsFineSegmentationVisualizer
)
from densepose.vis.extractor import (
    create_extractor,
    CompoundExtractor
)

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor


class DensePose:
    """
    DensePose processor modified for Apple Silicon.

    Detectron2 DensePose does not support MPS properly,
    so inference is forced to CPU.
    """

    def __init__(
        self,
        model_path="./checkpoints/densepose_",
        device="cpu"
    ):

        self.device = device

        print(
            "DensePose device:",
            self.device
        )


        self.config_path = os.path.join(
            model_path,
            "densepose_rcnn_R_50_FPN_s1x.yaml"
        )


        self.model_path = os.path.join(
            model_path,
            "model_final_162be9.pkl"
        )


        self.visualizations = [
            "dp_segm"
        ]


        self.VISUALIZERS = {
            "dp_segm":
            DensePoseResultsFineSegmentationVisualizer
        }


        self.min_score = 0.8


        self.cfg = self.setup_config()


        # Detectron2 works reliably on CPU here
        self.cfg.defrost()

        self.cfg.MODEL.DEVICE = "cpu"

        self.cfg.freeze()


        self.predictor = DefaultPredictor(
            self.cfg
        )



    def setup_config(self):

        opts = [
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
            str(self.min_score)
        ]


        cfg = get_cfg()

        add_densepose_config(
            cfg
        )


        cfg.merge_from_file(
            self.config_path
        )


        cfg.merge_from_list(
            opts
        )


        cfg.MODEL.WEIGHTS = (
            self.model_path
        )


        return cfg



    @staticmethod
    def _get_input_file_list(
        input_spec
    ):

        if os.path.isdir(
            input_spec
        ):

            return [
                os.path.join(
                    input_spec,
                    f
                )
                for f in os.listdir(
                    input_spec
                )
            ]


        elif os.path.isfile(
            input_spec
        ):

            return [
                input_spec
            ]


        return glob.glob(
            input_spec
        )



    def create_context(
        self,
        cfg,
        output_path
    ):

        visualizers = []
        extractors = []


        for vis_spec in self.visualizations:

            vis = self.VISUALIZERS[
                vis_spec
            ](
                cfg=cfg,
                texture_atlas=None,
                texture_atlases_dict=None,
                alpha=1.0
            )


            visualizers.append(
                vis
            )


            extractors.append(
                create_extractor(
                    vis
                )
            )


        return {

            "extractor":
            CompoundExtractor(
                extractors
            ),

            "visualizer":
            CompoundVisualizer(
                visualizers
            ),

            "out_fname":
            output_path,

            "entry_idx":
            0
        }



    def execute_on_outputs(
        self,
        context,
        entry,
        outputs
    ):


        data = context["extractor"](
            outputs
        )


        H, W, _ = entry["image"].shape


        result = np.zeros(
            (H, W),
            dtype=np.uint8
        )


        try:

            data, box = data[0]

            x, y, w, h = [
                int(v)
                for v in box[0].cpu().numpy()
            ]


            labels = (
                data[0]
                .labels[None]
                .cpu()
                .numpy()[0]
            )


            result[
                y:y+h,
                x:x+w
            ] = labels


        except Exception:

            pass



        Image.fromarray(
            result
        ).save(
            context["out_fname"]
        )



    def __call__(
        self,
        image_or_path,
        resize=512
    ):


        tmp_path = "./densepose_/tmp/"


        os.makedirs(
            tmp_path,
            exist_ok=True
        )


        image_path = os.path.join(
            tmp_path,
            f"{int(time.time())}-{randint(0,100000)}.png"
        )


        if isinstance(
            image_or_path,
            str
        ):

            shutil.copy(
                image_or_path,
                image_path
            )


        elif isinstance(
            image_or_path,
            Image.Image
        ):

            image_or_path.save(
                image_path
            )


        else:

            raise TypeError(
                "Invalid image input"
            )



        output_path = (
            image_path
            .replace(
                ".png",
                "_dense.png"
            )
        )


        original_size = Image.open(
            image_path
        ).size



        context = self.create_context(
            self.cfg,
            output_path
        )



        img = read_image(
            image_path,
            format="BGR"
        )


        if max(img.shape) > resize:

            scale = resize / max(img.shape)

            img = cv2.resize(
                img,
                (
                    int(img.shape[1]*scale),
                    int(img.shape[0]*scale)
                )
            )



        with torch.no_grad():

            outputs = self.predictor(
                img
            )["instances"]


            self.execute_on_outputs(
                context,
                {
                    "image": img,
                    "file_name": image_path
                },
                outputs
            )



        dense_gray = Image.open(
            output_path
        ).convert(
            "L"
        )


        dense_gray = dense_gray.resize(
            original_size,
            Image.NEAREST
        )


        os.remove(
            image_path
        )

        os.remove(
            output_path
        )


        return dense_gray