import os
import sys
import time
import signal
import argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils.config import (
    CONFIG, CLASSES, INFERENCE_CFG, SPRAY_PROFILES,
    CAMERA_CFG, get_project_root, setup_logging, ensure_dirs,
)
from model.network import build_model, count_parameters
from model.trainer import Trainer
from inference.classifier import SpinachClassifier, InferenceVisualizer
from spray_control.controller import SprayDecisionEngine, SpraySimulator, SprayAction
from drone_control.controller import DroneController, DroneState
from camera.capture import CameraCapture, FramePreprocessor
from preprocessing.dataset import split_dataset

logger = setup_logging("main")


class PrecisionSpraySystem:
    def __init__(self, args):
        self.args = args
        self.root = get_project_root()
        self.running = False

        self.drone = DroneController(
            connection_string=args.drone_conn,
            simulate=args.simulate,
        )
        self.classifier: "SpinachClassifier | None" = None
        self.spray_engine = SprayDecisionEngine()
        self.simulator = SpraySimulator()
        self.camera = CameraCapture(source=args.camera_source)
        self.preprocessor = FramePreprocessor()
        self.viz = InferenceVisualizer()

        self.frame_count = 0
        self.spray_count = 0
        self.session_log = []
        self.session_start = None

        self.spray_engine.register_callbacks(
            on_spray=self._handle_spray,
        )
        self.drone.register_state_callback(self._on_drone_state_change)

        ensure_dirs()
        logger.info("PrecisionSpraySystem initialized")

    def _on_drone_state_change(self, old, new):
        logger.info("Drone state: {} -> {}".format(old.value, new.value))

    def _handle_spray(self, command):
        self.spray_count += 1
        if self.args.simulate:
            self.simulator.execute_spray(command)
        else:
            self.drone.activate_spray(
                command.duration_sec, command.flow_rate_lpm
            )

    def _process_frame(self, frame):
        assert self.classifier is not None
        result = self.classifier.predict(frame)
        decision = self.spray_engine.decide(result)
        self.frame_count += 1

        log_entry = {
            "frame": self.frame_count,
            "timestamp": datetime.now().isoformat(),
            "prediction": result["class"],
            "confidence": result["confidence"],
            "action": decision.action.value,
            "drone_state": self.drone.state.value,
            "altitude": self.drone.current_altitude,
        }
        self.session_log.append(log_entry)

        if self.args.display:
            vis = self.viz.draw_prediction(frame, result)
            status_text = "SPRAY: {}".format(decision.action.value)
            color = (0, 0, 255) if decision.action != SprayAction.NO_SPRAY else (0, 200, 0)
            cv2.putText(
                vis, status_text, (10, vis.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            )
            cv2.putText(
                vis, "Drone: {}".format(self.drone.state.value), (10, vis.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )
            try:
                cv2.imshow("Precision Spray System", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.running = False
                elif key == ord("l"):
                    self.drone.land()
                elif key == ord("s"):
                    self.drone.emergency_stop()
            except cv2.error:
                pass

    def run_mission(self):
        logger.info("=" * 60)
        logger.info("PRECISION PESTICIDE SPRAY MISSION")
        logger.info("=" * 60)

        model_path = self.args.model or str(
            self.root / INFERENCE_CFG["model_path"]
        )
        self.classifier = SpinachClassifier(model_path=model_path)
        self.classifier.warmup(INFERENCE_CFG.get("warmup_runs", 3))

        if not self.args.offline:
            logger.info("Connecting to drone...")
            if not self.drone.connect():
                logger.error("Failed to connect to drone")
                return False

            logger.info("Arming motors...")
            if not self.drone.arm():
                logger.error("Failed to arm")
                return False

            logger.info("Taking off...")
            if not self.drone.takeoff():
                logger.error("Failed to takeoff")
                return False

        if not self.camera.open():
            logger.error("Failed to open camera")
            return False

        self.running = True
        self.session_start = time.time()
        logger.info("Mission started. Press 'q' to abort.")

        max_flight_time = DRONE_CFG.get("max_flight_time_min", 20) * 60

        try:
            while self.running:
                elapsed = time.time() - self.session_start
                if elapsed > max_flight_time:
                    logger.warning("Max flight time reached. Returning.")
                    break

                frame = self.camera.read()
                if frame is None:
                    logger.warning("No frame captured")
                    time.sleep(0.1)
                    continue

                self._process_frame(frame)

        except KeyboardInterrupt:
            logger.info("Mission interrupted by user")
        finally:
            self._end_mission()

        return True

    def run_demo(self):
        logger.info("Running in DEMO mode (no drone)")
        model_path = self.args.model or str(
            self.root / INFERENCE_CFG["model_path"]
        )
        self.classifier = SpinachClassifier(model_path=model_path)
        self.classifier.warmup()

        if self.args.image:
            image = cv2.imread(self.args.image)
            if image is None:
                logger.error("Failed to load: {}".format(self.args.image))
                return False

            assert self.classifier is not None
            result = self.classifier.predict(image)
            decision = self.spray_engine.decide(result)

            print("\n" + "=" * 50)
            print("  PREDICTION RESULT")
            print("=" * 50)
            print("  Class:        {}".format(result["class"].upper()))
            print("  Confidence:   {:.1f}%".format(result["confidence"] * 100))
            print("  Action:       {}".format(decision.action.value))
            print("  Duration:     {}s".format(decision.duration_sec))
            print("  Flow Rate:    {} LPM".format(decision.flow_rate_lpm))
            print("  Probabilities:")
            for cls, prob in result["probabilities"].items():
                print("    {}: {:.1f}%".format(cls, prob * 100))
            print("=" * 50)

            vis = self.viz.draw_prediction(image, result)
            output_path = str(self.root / "outputs" / "demo_result.jpg")
            cv2.imwrite(output_path, vis)
            print("\nSaved: {}".format(output_path))

            if self.args.display:
                try:
                    cv2.imshow("Demo", vis)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
            return True

        if self.camera.open():
            logger.info("Demo mode: live camera. Press 'q' to quit.")
            self.session_start = time.time()

            while True:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.1)
                    continue

                assert self.classifier is not None
                result = self.classifier.predict(frame)
                vis = self.viz.draw_prediction(frame, result)
                try:
                    cv2.imshow("Demo - Spinach Classifier", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    pass

            self.camera.close()
            cv2.destroyAllWindows()
            return True

        logger.warning("No image or camera specified. Use --image or --camera-source")
        return False

    def _end_mission(self):
        self.running = False

        if not self.args.offline:
            logger.info("Landing drone...")
            self.drone.land()
            self.drone.disconnect()

        self.camera.close()

        if self.args.display:
            cv2.destroyAllWindows()

        duration = time.time() - self.session_start if self.session_start else 0
        stats = self.spray_engine.get_stats()

        logger.info("=" * 60)
        logger.info("MISSION SUMMARY")
        logger.info("=" * 60)
        logger.info("Duration:     {:.1f}s".format(duration))
        logger.info("Frames:       {}".format(self.frame_count))
        logger.info("Total decisions: {}".format(stats["total_decisions"]))
        logger.info("Spray commands:  {}".format(stats["spray_commands_issued"]))
        logger.info("No-spray:        {}".format(stats["no_spray_decisions"]))
        logger.info("=" * 60)

        if self.session_log:
            import json
            log_path = self.root / "logs" / "session_{}.json".format(
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            with open(str(log_path), "w") as f:
                json.dump(self.session_log, f, indent=2)
            logger.info("Session log saved: {}".format(log_path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Based Autonomous Drone for Precision Pesticide Spraying"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data-dir", type=str, help="Data directory")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")

    mission_parser = subparsers.add_parser("mission", help="Run spray mission")
    mission_parser.add_argument("--model", type=str, help="Model path")
    mission_parser.add_argument("--camera-source", type=int, default=0)
    mission_parser.add_argument("--drone-conn", type=str, default=None)
    mission_parser.add_argument("--simulate", action="store_true", default=True)
    mission_parser.add_argument("--display", action="store_true", default=True)
    mission_parser.add_argument("--offline", action="store_true")

    demo_parser = subparsers.add_parser("demo", help="Run demo/inference")
    demo_parser.add_argument("--model", type=str, help="Model path")
    demo_parser.add_argument("--image", type=str, help="Image path")
    demo_parser.add_argument("--camera-source", type=int, default=0)
    demo_parser.add_argument("--display", action="store_true", default=True)

    split_parser = subparsers.add_parser("split", help="Split dataset")
    split_parser.add_argument("--raw-dir", type=str, default=None)
    split_parser.add_argument("--output-dir", type=str, default=None)

    gen_parser = subparsers.add_parser("generate", help="Generate synthetic images")
    gen_parser.add_argument("--num", type=int, default=200, help="Images per class")
    gen_parser.add_argument("--quick-test", action="store_true", help="20 per class")

    dl_parser = subparsers.add_parser("download", help="Download public dataset")
    dl_parser.add_argument(
        "--source",
        choices=["plantvillage", "spinach", "all", "list"],
        default="list",
        help="Which dataset to download",
    )
    dl_parser.add_argument("--url", type=str, help="Custom download URL")

    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--camera-source", type=int, default=0)
    parser.add_argument("--drone-conn", type=str, default=None)
    parser.add_argument("--simulate", action="store_true", default=True)
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--offline", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    root = get_project_root()

    if args.command == "train":
        logger.info("Starting training...")
        data_dir = args.data_dir or str(root / "data" / "processed")
        trainer = Trainer(data_dir=data_dir)
        if args.epochs:
            trainer.epochs = args.epochs
        trainer.train()

    elif args.command == "mission":
        system = PrecisionSpraySystem(args)
        system.run_mission()

    elif args.command == "demo":
        system = PrecisionSpraySystem(args)
        system.run_demo()

    elif args.command == "split":
        raw_dir = args.raw_dir or str(root / "data" / "raw")
        out_dir = args.output_dir or str(root / "data" / "processed")
        split_dataset(raw_dir, out_dir)

    elif args.command == "generate":
        from src.data_collection.synthetic_generator import generate_dataset
        num = 20 if args.quick_test else args.num
        generate_dataset(images_per_class=num)

    elif args.command == "download":
        from src.data_collection.downloader import (
            download_plantvillage, download_spinach_specific,
            list_available_datasets, download_from_url,
        )
        if args.source == "list":
            list_available_datasets()
        elif args.source == "plantvillage":
            download_plantvillage()
        elif args.source == "spinach":
            download_spinach_specific()
        elif args.source == "all":
            download_spinach_specific()
            download_plantvillage()
        elif args.url:
            download_from_url(args.url)

    else:
        print("AI-Based Autonomous Drone for Precision Pesticide Spraying")
        print("=" * 60)
        print("Usage:")
        print("  python main.py generate       - Generate synthetic images")
        print("  python main.py download       - Download public dataset")
        print("  python main.py split          - Split dataset")
        print("  python main.py train          - Train the model")
        print("  python main.py demo           - Run demo/inference")
        print("  python main.py mission        - Run spray mission")
        print("=" * 60)


DRONE_CFG = CONFIG["drone"]

if __name__ == "__main__":
    main()
