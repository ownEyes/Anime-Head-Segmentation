from transformers import AutoModelForObjectDetection, AutoImageProcessor
from ultralytics.models.fastsam import FastSAMPredictor
import supervision as sv
import torch
import numpy as np
import cv2

detector = AutoModelForObjectDetection.from_pretrained("Yifeng-Liu/rt-detr-finetuned-for-satellite-image-roofs-detection")
detector_processor = AutoImageProcessor.from_pretrained("Yifeng-Liu/rt-detr-finetuned-for-satellite-image-roofs-detection")


overrides = dict(conf=0.25, task="segment", mode="predict", model="FastSAM-x.pt", save=False)
segment_predictor = FastSAMPredictor(overrides=overrides)

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes


class ModelInference:
    def __init__(self, detector, detector_processor, segment_predictor, id2label, CONFIDENCE_TRESHOLD):
        self.detector = detector
        self.detector_processor = detector_processor
        self.segment_predictor = segment_predictor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CONFIDENCE_TRESHOLD = CONFIDENCE_TRESHOLD
        self.id2label = id2label
        self.mask_annotator = sv.MaskAnnotator()

    def predict_one(self, image_path):
        image = cv2.imread(image_path)
        with torch.no_grad():
            self.detector.to(self.device)

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
                return None
            else:
                det_detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.5)

            everything_results = self.segment_predictor(image)
        if everything_results[0].masks is not None:
            bbox_results = self.segment_predictor.prompt(everything_results, det_detections.xyxy.tolist())[0]
            seg_detections = sv.Detections.from_ultralytics(bbox_results)

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
            return None

    def predict_folder(self, image_folder, batch_size=4):
        files = [path for path in image_folder.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]
