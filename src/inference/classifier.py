import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

from utils.config import (
    CONFIG, INFERENCE_CFG, MODEL_CFG, CLASSES, NUM_CLASSES,
    get_project_root, get_device, setup_logging
)
from model.network import build_model

logger = setup_logging("inference")


class SpinachClassifier:
    def __init__(self, model_path=None, device=None):
        self.root = get_project_root()
        self.device = device or get_device()

        if model_path is None:
            model_path = self.root / INFERENCE_CFG["model_path"]

        self.confidence_threshold = INFERENCE_CFG["confidence_threshold"]
        self.use_half = INFERENCE_CFG.get("half_precision", False)

        self.model = build_model(num_classes=NUM_CLASSES)
        self._load_weights(str(model_path))
        self.model.to(self.device)
        self.model.eval()

        self.input_size = tuple(MODEL_CFG["input_size"][1:])
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        if self.use_half and self.device.type == "cuda":
            self.model.half()

        logger.info(
            "Classifier loaded | Model: {} | Device: {} | Half: {}".format(
                model_path, self.device, self.use_half
            )
        )

    def _load_weights(self, model_path):
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded checkpoint from epoch {}".format(
                    checkpoint.get("epoch", "?")
                ))
            else:
                self.model.load_state_dict(checkpoint)
            logger.info("Weights loaded from {}".format(model_path))
        else:
            logger.warning("Model not found at {}. Using untrained model.".format(
                model_path
            ))

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std

        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        tensor = tensor.unsqueeze(0)

        if self.use_half and self.device.type == "cuda":
            tensor = tensor.half()

        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Dict:
        start_time = time.time()

        tensor = self.preprocess(image)
        outputs = self.model(tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)
        predicted_class = CLASSES[predicted_idx.item()]
        confidence_score = confidence.item()
        probs_dict = {
            CLASSES[i]: round(probabilities[0][i].item(), 4)
            for i in range(NUM_CLASSES)
        }

        inference_time = (time.time() - start_time) * 1000

        result = {
            "class": predicted_class,
            "confidence": confidence_score,
            "probabilities": probs_dict,
            "inference_ms": round(inference_time, 2),
            "below_threshold": confidence_score < self.confidence_threshold,
        }

        return result

    def predict_batch(self, images: list) -> list:
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0)

        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted_idx = probabilities.max(1)

        results = []
        for i in range(len(images)):
            results.append({
                "class": CLASSES[predicted_idx[i].item()],
                "confidence": confidences[i].item(),
                "probabilities": {
                    CLASSES[j]: round(probabilities[i][j].item(), 4)
                    for j in range(NUM_CLASSES)
                },
            })
        return results

    def warmup(self, num_runs=3):
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for _ in range(num_runs):
            self.predict(dummy)
        logger.info("Warmup complete ({} runs)".format(num_runs))


class InferenceVisualizer:
    @staticmethod
    def draw_prediction(
        image: np.ndarray,
        result: dict,
        position: Tuple[int, int] = (10, 30),
    ) -> np.ndarray:
        vis = image.copy()
        cls = result["class"]
        conf = result["confidence"]

        color_map = {
            "healthy": (0, 200, 0),
            "mild": (0, 200, 255),
            "moderate": (0, 140, 255),
            "severe": (0, 0, 255),
        }
        color = color_map.get(cls, (255, 255, 255))

        text = "{}: {:.1f}%".format(cls.upper(), conf * 100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2

        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position
        cv2.rectangle(vis, (x - 5, y - h - 10), (x + w + 10, y + 10), (0, 0, 0), -1)
        cv2.putText(vis, text, (x, y), font, font_scale, color, thickness)

        prob_y = y + 40
        for cls_name in CLASSES:
            prob = result["probabilities"].get(cls_name, 0)
            bar_w = int(prob * 200)
            bar_color = color_map.get(cls_name, (200, 200, 200))
            cv2.rectangle(vis, (x, prob_y - 12), (x + bar_w, prob_y), bar_color, -1)
            cv2.putText(
                vis, "{}: {:.0f}%".format(cls_name, prob * 100),
                (x + 210, prob_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
            prob_y += 18

        return vis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spinach Disease Classifier")
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--camera", action="store_true", help="Use webcam")
    args = parser.parse_args()

    classifier = SpinachClassifier(model_path=args.model)
    classifier.warmup()

    if args.camera:
        cap = cv2.VideoCapture(0)
        viz = InferenceVisualizer()
        logger.info("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = classifier.predict(frame)
            vis = viz.draw_prediction(frame, result)
            cv2.imshow("Spinach Classifier", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif args.image:
        image = cv2.imread(args.image)
        if image is None:
            logger.error("Failed to load image: {}".format(args.image))
        else:
            result = classifier.predict(image)
            print("\nPrediction:")
            print("  Class: {}".format(result["class"]))
            print("  Confidence: {:.2f}%".format(result["confidence"] * 100))
            print("  Probabilities: {}".format(result["probabilities"]))
            print("  Inference Time: {}ms".format(result["inference_ms"]))

            vis = InferenceVisualizer.draw_prediction(image, result)
            cv2.imwrite("output_prediction.jpg", vis)
            print("  Saved: output_prediction.jpg")
