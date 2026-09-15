import os
import cv2
import time
from pathlib import Path
from datetime import datetime

from utils.config import CLASSES, get_project_root, setup_logging

logger = setup_logging("data_collector")


class DataCollector:
    def __init__(self, output_dir=None, camera_source=0):
        self.root = get_project_root()
        self.output_dir = Path(output_dir or self.root / "data" / "raw")
        self.camera_source = camera_source

        for cls in CLASSES:
            (self.output_dir / cls).mkdir(parents=True, exist_ok=True)

        self.cap = None
        self.is_capturing = False
        self.current_label = None
        self.capture_count = {cls: 0 for cls in CLASSES}

    def _count_existing(self):
        for cls in CLASSES:
            cls_dir = self.output_dir / cls
            count = len(list(cls_dir.glob("*.jpg")))
            self.capture_count[cls] = count
            logger.info("  {}: {} images".format(cls, count))

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            logger.error("Cannot open camera {}".format(self.camera_source))
            return

        self._count_existing()

        print("\n=== Spinach Dataset Collector ===")
        print("Keys:")
        for i, cls in enumerate(CLASSES):
            print("  [{}] - {} mode".format(i + 1, cls.upper()))
        print("  [SPACE] - Capture frame")
        print("  [Q]     - Quit")
        print("=" * 35)

        self.is_capturing = True

        while self.is_capturing:
            ret, frame = self.cap.read()
            if not ret:
                break

            display = frame.copy()
            label_text = "Label: {}".format(
                self.current_label.upper() if self.current_label else "NONE"
            )
            cv2.putText(
                display, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            count_text = "Count: {}".format(
                self.capture_count.get(self.current_label, 0)
                if self.current_label else "-"
            )
            cv2.putText(
                display, count_text, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            for i, cls in enumerate(CLASSES):
                color = (0, 255, 0) if cls == self.current_label else (128, 128, 128)
                text = "[{}] {} ({})".format(i + 1, cls, self.capture_count[cls])
                cv2.putText(
                    display, text, (10, 110 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

            cv2.imshow("Data Collector", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                self._capture_frame(frame)
            elif ord("1") <= key <= ord("1") + len(CLASSES) - 1:
                idx = key - ord("1")
                self.current_label = CLASSES[idx]
                logger.info("Label set to: {}".format(self.current_label))

        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Collection complete")
        self._count_existing()

    def _capture_frame(self, frame):
        if self.current_label is None:
            logger.warning("No label selected. Press 1-{} first.".format(
                len(CLASSES)
            ))
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = "{}.jpg".format(timestamp)
        filepath = self.output_dir / self.current_label / filename

        cv2.imwrite(str(filepath), frame)
        self.capture_count[self.current_label] += 1
        logger.info("Captured: {} (Total: {})".format(
            filename, self.capture_count[self.current_label]
        ))

    def capture_from_webcam(self, frames_per_class=100, delay=0.1):
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            logger.error("Cannot open camera")
            return

        for cls in CLASSES:
            cls_dir = self.output_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Collecting {} (target: {} frames)".format(cls, frames_per_class))

            for i in range(frames_per_class):
                ret, frame = self.cap.read()
                if not ret:
                    break

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filepath = cls_dir / "{}.jpg".format(timestamp)
                cv2.imwrite(str(filepath), frame)

                display = frame.copy()
                cv2.putText(
                    display,
                    "{}: {}/{}".format(cls, i + 1, frames_per_class),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.imshow("Collecting", display)
                cv2.waitKey(1)
                time.sleep(delay)

            logger.info("Done: {}".format(cls))

        self.cap.release()
        cv2.destroyAllWindows()
        self._count_existing()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect training data")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--auto", action="store_true",
                        help="Auto-capture mode (fixed frames per class)")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()

    collector = DataCollector(output_dir=args.output, camera_source=args.camera)

    if args.auto:
        collector.capture_from_webcam(frames_per_class=args.frames)
    else:
        collector.start()
