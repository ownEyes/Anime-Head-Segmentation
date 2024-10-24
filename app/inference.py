from transformers import AutoModelForObjectDetection, AutoImageProcessor
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from ultralytics.models.fastsam import FastSAMPredictor
import supervision as sv
import torch
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
from supervision.dataset.utils import approximate_mask_with_polygons
from supervision.detection.utils import (
    contains_holes,
    contains_multiple_segments,
)

detector = AutoModelForObjectDetection.from_pretrained("Yifeng-Liu/rt-detr-finetuned-for-satellite-image-roofs-detection")
detector_processor = AutoImageProcessor.from_pretrained("Yifeng-Liu/rt-detr-finetuned-for-satellite-image-roofs-detection")


overrides = dict(conf=0.25, task="segment", mode="predict", model="FastSAM-x.pt", save=False)
segment_predictor = FastSAMPredictor(overrides=overrides)

# IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes


class ImageInferenceDataset(Dataset):
    def __init__(self, image_paths: Path, image_processor):
        """
        A custom dataset class for image inference without annotations or masks.

        Args:
            image_folder (Path): The path to the folder containing images.
            image_processor: A callable for processing images (usually a transformer or feature extractor).
            image_formats (set): A set of supported image formats to be filtered.
        """
        self.image_processor = image_processor
        # Filter out files that are not supported image formats
        self.image_files = image_paths

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get an image from the dataset at the specified index.

        Args:
            idx (int): The index of the image.

        Returns:
            Tuple[torch.Tensor, str]: A tuple containing the processed image tensor and the image file path.
        """
        image_path = self.image_files[idx]
        # Open image using PIL and process it using the provided image processor
        with Image.open(image_path) as img:
            orig_size = img.size
            img = img.convert("RGB")  # Ensure all images are in RGB format for consistency
            processed_img = self.image_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        return processed_img, str(image_path), orig_size


def collate_fn_inference(batch: List[Tuple[torch.Tensor, str]]) -> dict:
    """
    Collate function for batching images for inference.

    Args:
        batch (List[Tuple[torch.Tensor, str]]): A list of tuples where each tuple contains
                                                the processed image tensor and image path.

    Returns:
        dict: A dictionary containing the batched image tensors and corresponding image file paths.
    """
    pixel_values = [item[0] for item in batch]  # Extract processed images
    image_paths = [item[1] for item in batch]   # Extract image paths
    orig_sizes = [item[2] for item in batch]

    # Pad the images to match the largest image in the batch
    encoding = detector_processor.pad(pixel_values, return_tensors="pt")

    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],  # Padding mask (if needed by the model)
        'image_paths': image_paths,
        'orig_sizes': orig_sizes
    }


class ModelInference:
    def __init__(self, detector, detector_processor, segment_predictor, id2label, CONFIDENCE_TRESHOLD):
        self.detector = detector
        self.detector_processor = detector_processor
        self.segment_predictor = segment_predictor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CONFIDENCE_TRESHOLD = CONFIDENCE_TRESHOLD
        self.id2label = id2label
        self.mask_annotator = sv.MaskAnnotator()
        self.detector.to(self.device)

    def predict_one(self, image_path):
        image = cv2.imread(image_path)
        with torch.no_grad():

            # load image and predict
            inputs = self.detector_processor(images=image, return_tensors='pt').to(self.device)
            outputs = self.detector(**inputs)

            # post-process
            target_sizes = torch.tensor([image.shape[:2]]).to(self.device)
            results = detector_processor.post_process_object_detection(
                outputs=outputs,
                threshold=self.CONFIDENCE_TRESHOLD,
                target_sizes=target_sizes
            )[0]
            if results['boxes'].numel() == 0:
                print("No bounding box detected")
                return None, None
            else:
                det_detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.5)

            everything_results = self.segment_predictor(image)
        if everything_results[0].masks is not None:
            bbox_results = self.segment_predictor.prompt(everything_results, det_detections.xyxy.tolist())[0]
            seg_detections = sv.Detections.from_ultralytics(bbox_results)
            seg_detections = self.filter_small_masks(seg_detections)

            max_length = max(len(name) for name in self.id2label.values())

            # Create a new NumPy array with the appropriate dtype based on the longest string
            seg_detections.data['class_name'] = np.array(seg_detections.data['class_name'], dtype=f'<U{max_length}')

            for idx, class_name in enumerate(seg_detections.data['class_name']):
                if class_name == 'object':
                    seg_detections.data['class_name'][idx] = self.id2label[seg_detections.class_id[idx]]

            annotated_frame = image.copy()
            annotated_frame = self.mask_annotator.annotate(scene=annotated_frame, detections=seg_detections)

            return seg_detections, annotated_frame
        else:
            print("No segmentation mask generated")
            return None, None

    def predict_folder(self, image_paths, batch_size=4):
        dataset = ImageInferenceDataset(image_paths=image_paths, image_processor=detector_processor)

        # Create DataLoader instance with the custom collate function
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_inference)

        detector_failed_list = []
        segmentor_failed_list = []

        id2label = {0: 'building'}
        max_length = max(len(name) for name in id2label.values())

        all_image_paths = []

        all_results = []

        for idx, batch in enumerate(tqdm(dataloader)):
            pixel_values = batch["pixel_values"].to(self.device)
            pixel_mask = batch["pixel_mask"].to(self.device)
            image_paths = batch["image_paths"]
            orig_sizes = batch["orig_sizes"]

            orig_target_sizes = torch.tensor(orig_sizes, device=self.device)

            with torch.no_grad():
                outputs = self.detector(
                    pixel_values=pixel_values, pixel_mask=pixel_mask)

            # orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)

            detector_results = detector_processor.post_process_object_detection(
                outputs,
                target_sizes=orig_target_sizes)

            detector_detections = []
            detector_to_remove = []

            for idx, detector_result in enumerate(detector_results):
                if detector_result['boxes'].numel() == 0:
                    # The tensor is empty
                    detector_to_remove.append(idx)
                else:
                    detector_detections.append(sv.Detections.from_transformers(transformers_results=detector_result))

            if detector_to_remove is not None:
                # Remove items from detector_results and image_ids by reversing the indices to avoid index shifting
                for idx in sorted(detector_to_remove, reverse=True):
                    detector_failed_list.append(image_paths[idx])
                    del image_paths[idx]

            images_raw = [cv2.imread(image_path) for image_path in image_paths]

            boxes = [detections.xyxy.tolist() for detections in detector_detections]

            results = []

            to_remove_seg = []

            for idx, (image_path, image, box) in enumerate(zip(image_paths, images_raw, boxes)):
                try:
                    with torch.no_grad():
                        # segmentation_result = segment_model(image, bboxes=box)[0]
                        everything_results = self.segment_predictor(image)

                        if everything_results[0].masks is not None:
                            bbox_results = self.segment_predictor.prompt(everything_results, box)[0]
                            seg_detections = sv.Detections.from_ultralytics(bbox_results)
                            seg_detections = self.filter_small_masks(seg_detections)
                            seg_detections.data['class_name'] = np.array(seg_detections.data['class_name'], dtype=f'<U{max_length}')
                            for idx, class_name in enumerate(seg_detections.data['class_name']):
                                if class_name == 'object':
                                    seg_detections.data['class_name'][idx] = id2label[seg_detections.class_id[idx]]
                            results.append(seg_detections)
                        else:
                            to_remove_seg.append(idx)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print(f"box: {box}")
                    print(f"image id: {image_path}")
                # result = sv.Detections.from_ultralytics(segmentation_result)
                # results.append(result)

            if to_remove_seg is not None:
                for idx in sorted(to_remove_seg, reverse=True):
                    segmentor_failed_list.append(image_paths[idx])
                    del image_paths[idx]

            if len(results) != len(image_paths):
                print(f"Length of results ({len(results)}) does not match the length of image_ids ({len(image_paths)})")
                continue

            all_image_paths.extend(image_paths)
            all_results.extend(results)

            annotated_frame = cv2.imread(all_image_paths[0]).copy()
            annotated_frame = self.mask_annotator.annotate(scene=annotated_frame, detections=all_results[0])

        return all_image_paths, all_results, annotated_frame, detector_failed_list, segmentor_failed_list

    def filter_small_masks(self, detections: sv.Detections) -> sv.Detections:
        valid_indices = []
        min_image_area_percentage = 0.002
        max_image_area_percentage = 0.80
        approximation_percentage = 0.75
        for i, mask in enumerate(detections.mask):

            # Check for structural issues in the mask
            if not (contains_holes(mask) or contains_multiple_segments(mask)):
                # Check if the mask can be approximated to a polygon successfully
                if not approximate_mask_with_polygons(mask=mask,
                                                      min_image_area_percentage=min_image_area_percentage,
                                                      max_image_area_percentage=max_image_area_percentage,
                                                      approximation_percentage=approximation_percentage,
                                                      ):
                    print(f"Skipping mask {i} due to structural issues")
                    continue

            # If all checks pass, add index to valid_indices
            valid_indices.append(i)

        filtered_xyxy = detections.xyxy[valid_indices]
        filtered_mask = detections.mask[valid_indices]
        filtered_confidence = detections.confidence[valid_indices]
        filtered_class_id = detections.class_id[valid_indices]
        filtered_class_name = detections.data['class_name'][valid_indices]

        detections.xyxy = filtered_xyxy
        detections.mask = filtered_mask
        detections.confidence = filtered_confidence
        detections.class_id = filtered_class_id
        detections.data['class_name'] = filtered_class_name
        return detections

    def get_dict(
        self,
        image_paths: List[Any],
        detections: List[Any]
    ) -> Dict[str, Any]:

        detections_dict = {}

        for idx, image_path in enumerate(image_paths):
            detections_dict[image_path] = detections[idx]

        return detections_dict

    def save_annotations(self,
                         image_paths,
                         detections,
                         class_names,
                         annotation_path,
                         MIN_IMAGE_AREA_PERCENTAGE=0.002,
                         MAX_IMAGE_AREA_PERCENTAGE=0.80,
                         APPROXIMATION_PERCENTAGE=0.75):
        # image_dir = annotation_path.parent
        detections_dict = self.get_dict(image_paths, detections)
        sv.DetectionDataset(
            classes=class_names,
            images=image_paths,
            annotations=detections_dict
        ).as_coco(
            images_directory_path=None,
            annotations_path=annotation_path,
            min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
            max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
            approximation_percentage=APPROXIMATION_PERCENTAGE
        )

        return
