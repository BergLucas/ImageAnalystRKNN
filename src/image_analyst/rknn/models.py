from __future__ import annotations
from image_analyst.exceptions import DownloadFailedException
from image_analyst.utils import download_file, ReportFunction
from image_analyst.exceptions import (
    ModelLoadingFailedException,
    InvalidDtypeException,
    DetectionFailedException,
)
from image_analyst.image import (
    ImageFormat,
    BoundingBox,
    ImageEmbedder,
    EmbeddingFunction,
)
from image_analyst.utils import NmsFunction, NmsPython, sigmoid
from image_analyst.models import ODModel, Detection
from rknnlite.api import RKNNLite
from typing import Optional
import numpy as np
import logging


logger = logging.getLogger(__name__)


class YoloV3Rknn(ODModel):
    """This type represents a RKNN YoloV3 model."""

    @staticmethod
    def coco(
        score_threshold: float = 0.5,
        nms_function: Optional[NmsFunction] = None,
        embedding_function: Optional[EmbeddingFunction] = None,
        report_callback: Optional[ReportFunction] = None,
    ) -> YoloV3Rknn:
        """Creates a new YoloV3Rknn pretrained with the coco dataset.

        Args:
            score_threshold (float, optional): the score threshold to use.
                Defaults to 0.5.
            nms_function (Optional[NmsFunction], optional): the nms function to use.
                Defaults to a new NmsPython.
            embedding_function (Optional[EmbeddingFunction], optional): the embedding
                function to use. Defaults to an ImageEmbedder.
            report_callback (Optional[ReportFunction], optional): the report function
                that is called while downloading the files. Defaults to None.

        Raises:
            ModelLoadingFailedException: if the model cannot be loaded.
            ValueError: if the score threshold is not between 0 and 1.

        Returns:
            YoloV3Rknn: the new YoloV3Rknn.
        """
        try:
            model_path = download_file(
                "https://github.com/BergLucas/ImageAnalystRKNN/releases/download/v0.1.1/rknn-yolov3-coco.rknn",  # noqa: E501
                "rknn-yolov3-coco.rknn",
                report_callback,
            )
            labels_path = download_file(
                "https://github.com/BergLucas/ImageAnalystRKNN/releases/download/v0.1.1/rknn-yolov3-coco.names",  # noqa: E501
                "rknn-yolov3-coco.names",
                report_callback,
            )
        except DownloadFailedException as e:
            raise ModelLoadingFailedException(
                "Cannot download the YoloV3 model."
            ) from e

        if nms_function is None:
            nms_function = NmsPython()

        if embedding_function is None:
            embedding_function = ImageEmbedder()

        return YoloV3Rknn(
            model_path,
            labels_path,
            (416, 416),
            [(6, 7, 8), (3, 4, 5), (0, 1, 2)],
            [
                (10, 13),
                (16, 30),
                (33, 23),
                (30, 61),
                (62, 45),
                (59, 119),
                (116, 90),
                (156, 198),
                (373, 326),
            ],
            score_threshold,
            nms_function,
            embedding_function,
        )

    @staticmethod
    def tiny_coco(
        score_threshold: float = 0.25,
        nms_function: Optional[NmsFunction] = None,
        embedding_function: Optional[EmbeddingFunction] = None,
        report_callback: Optional[ReportFunction] = None,
    ) -> YoloV3Rknn:
        """Creates a new tiny YoloV3Rknn pretrained with the coco dataset.

        Args:
            score_threshold (float, optional): the score threshold to use.
                Defaults to 0.25.
            nms_function (Optional[NmsFunction], optional): the nms function to use.
                Defaults to a new NmsPython.
            embedding_function (Optional[EmbeddingFunction], optional): the embedding
                function to use. Defaults to an ImageEmbedder.
            report_callback (Optional[ReportFunction], optional): the report function
                that is called while downloading the files. Defaults to None.

        Raises:
            ModelLoadingFailedException: if the model cannot be loaded.
            ValueError: if the score threshold is not between 0 and 1.

        Returns:
            YoloV3Rknn: the new YoloV3Rknn.
        """
        try:
            model_path = download_file(
                "https://github.com/BergLucas/ImageAnalystRKNN/releases/download/v0.1.1/rknn-yolov3-tiny-coco.rknn",  # noqa: E501
                "rknn-yolov3-tiny-coco.rknn",
                report_callback,
            )
            labels_path = download_file(
                "https://github.com/BergLucas/ImageAnalystRKNN/releases/download/v0.1.1/rknn-yolov3-tiny-coco.names",  # noqa: E501
                "rknn-yolov3-tiny-coco.names",
                report_callback,
            )
        except DownloadFailedException as e:
            raise ModelLoadingFailedException(
                "Cannot download the YoloV3 model."
            ) from e

        if nms_function is None:
            nms_function = NmsPython()

        if embedding_function is None:
            embedding_function = ImageEmbedder()

        return YoloV3Rknn(
            model_path,
            labels_path,
            (416, 416),
            [(3, 4, 5), (0, 1, 2)],
            [(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)],
            score_threshold,
            nms_function,
            embedding_function,
        )

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        model_size: tuple[int, int],
        model_masks: list[tuple[int, int, int]],
        model_anchors: list[tuple[int, int]],
        score_threshold: float,
        nms_function: NmsFunction,
        embedding_function: EmbeddingFunction,
    ) -> None:
        """Initialises `self` to a new YoloV3Rknn.

        Args:
            model_path (str): the path to the model file.
            labels_path (str): the path to the labels file.
            model_size (tuple[int, int]): the size of the model.
            model_masks (list[tuple[int, int, int]]): the masks of the model.
            model_anchors (list[tuple[int, int]]): the anchors of the model.
            score_threshold (float): the score threshold to use.
            nms_function (NmsFunction): the NMS function to use.
            embedding_function (EmbeddingFunction): the embedding function to use.

        Raises:
            ModelLoadingFailedException: if the model cannot be loaded.
            ValueError: if the score threshold is not between 0 and 1.
        """
        if score_threshold < 0 or score_threshold > 1:
            raise ValueError("score_threshold must be between 0 and 1.")

        try:
            with open(labels_path, "rt") as f:
                self.__supported_classes = tuple(f.read().splitlines())
        except OSError:
            raise ModelLoadingFailedException("Cannot load the supported classes.")

        self.__rknn = RKNNLite()

        if self.__rknn.load_rknn(model_path) != 0:
            raise ModelLoadingFailedException("Cannot load the YoloV3 model.")

        self.__model_size = model_size
        self.__model_masks = model_masks
        self.__model_anchors = model_anchors
        self.__nms_function = nms_function
        self.__score_threshold = score_threshold
        self.__embedding_function = embedding_function

    @property
    def supported_classes(self) -> tuple[str, ...]:  # noqa: D102
        return self.__supported_classes

    @property
    def supported_dtype(self) -> type:  # noqa: D102
        return np.uint8

    @property
    def supported_format(self) -> ImageFormat:  # noqa: D102
        return ImageFormat.RGB

    def __call__(self, image: np.ndarray) -> list[Detection]:  # noqa: D102
        if image.dtype != self.supported_dtype:
            raise InvalidDtypeException("The image dtype is not supported.")

        logger.info("Started Image preprocessing")
        embedded_image, bounding_box = self.__embedding_function(
            image, *self.__model_size
        )
        logger.info("Completed Image preprocessing")

        logger.info("Started Image detection")
        if self.__rknn.init_runtime() != 0:
            raise DetectionFailedException("Failed to initialise runtime.")

        try:
            outputs = self.__rknn.inference(inputs=[embedded_image])
            self.__rknn.release()
        except Exception as e:
            raise DetectionFailedException("Failed to detect objects.") from e
        logger.info("Completed Image detection")

        logger.info("Started Bounding boxes creation")
        height, width, _ = image.shape
        model_height, model_width = self.__model_size

        x_scale = width / bounding_box.width
        y_scale = height / bounding_box.height

        detections = []
        for out, mask in zip(outputs, self.__model_masks):
            grid_height, grid_width = out.shape[2:]

            raw_detections = np.transpose(
                out.reshape(
                    -1, 4 + 1 + len(self.__supported_classes), grid_height, grid_width
                ),
                (2, 3, 0, 1),
            )

            for row in range(grid_height):
                for col in range(grid_width):
                    for raw_detection, anchor in zip(
                        raw_detections[row, col],
                        (self.__model_anchors[i] for i in mask),
                    ):
                        objectness, *class_probabilities = sigmoid(raw_detection[4:])
                        class_id = int(np.argmax(class_probabilities))
                        score = float(objectness * class_probabilities[class_id])

                        if score < self.__score_threshold:
                            continue

                        x, y = sigmoid(raw_detection[:2])
                        w, h = np.exp(raw_detection[2:4]) * anchor

                        xcenter = (col + x) / grid_width * model_width
                        ycenter = (row + y) / grid_height * model_height

                        xmin = int((xcenter - w / 2 - bounding_box.xmin) * x_scale)
                        ymin = int((ycenter - h / 2 - bounding_box.ymin) * y_scale)
                        xmax = int((xcenter + w / 2 - bounding_box.xmin) * x_scale)
                        ymax = int((ycenter + h / 2 - bounding_box.ymin) * y_scale)

                        detections.append(
                            Detection(
                                class_name=self.__supported_classes[class_id],
                                score=score,
                                bounding_box=BoundingBox(
                                    xmin=xmin,
                                    ymin=ymin,
                                    xmax=xmax,
                                    ymax=ymax,
                                ),
                            )
                        )
        logger.info("Completed Bounding boxes creation")

        logger.info("Started NMS")
        result = self.__nms_function(detections)
        logger.info("Completed NMS")

        return result
