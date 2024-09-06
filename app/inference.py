from transformers import AutoModelForObjectDetection, AutoImageProcessor
from ultralytics.models.fastsam import FastSAMPredictor
import supervision as sv

detector = AutoModelForObjectDetection.from_pretrained("Yifeng-Liu/rt-detr-finetuned-for-satellite-image-roofs-detection")
detector_processor = AutoImageProcessor.from_pretrained("Yifeng-Liu/rt-detr-finetuned-for-satellite-image-roofs-detection")


overrides = dict(conf=0.25, task="segment", mode="predict", model="FastSAM-x.pt", save=False)
segment_predictor = FastSAMPredictor(overrides=overrides)


class ModelInference:
    def __init__(self, detector, detector_processor, segment_predictor):
        self.detector = detector
        self.detector_processor = detector_processor
        self.segment_predictor = segment_predictor

    def predict_one(self, image):

        everything_results = self.segment_predictor(image)
        if everything_results[0].masks is not None:
            bbox_results = self.segment_predictor.prompt(everything_results, det_detections.xyxy.tolist())[0]
            seg_detections = sv.Detections.from_ultralytics(bbox_results)
            return seg_detections
        else:
            print("No segmentation mask generated")
            return None
