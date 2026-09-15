"""
Synthetic Leaf Image Generator
Creates realistic-looking synthetic spinach leaf images for testing the pipeline.
Generates 4 severity classes with distinct visual patterns.

Use this to validate the entire training + inference pipeline before
collecting real images.
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
from typing import Tuple

from utils.config import CLASSES, get_project_root, setup_logging

logger = setup_logging("synthetic_generator")

ROOT = get_project_root()


class SyntheticLeafGenerator:
    """Generates synthetic leaf-like images with disease patterns."""

    def __init__(self, img_size: Tuple[int, int] = (224, 224), seed: int = 42):
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)

    def _draw_leaf_shape(self, img, center, axes, angle, color):
        overlay = img.copy()
        cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1)
        return cv2.addWeighted(overlay, 0.8, img, 0.2, 0)

    def _add_leaf_texture(self, img, intensity=0.1):
        noise = self.rng.normal(0, 25, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise * intensity, 0, 255)
        return noisy.astype(np.uint8)

    def _draw_veins(self, img, center, angle, length, thickness=1, color=(0, 80, 0)):
        rad = np.radians(angle)
        end_x = int(center[0] + length * np.cos(rad))
        end_y = int(center[1] + length * np.sin(rad))
        cv2.line(img, center, (end_x, end_y), color, thickness)
        return img

    def _add_spots(self, img, num_spots, color, min_r=3, max_r=12):
        h, w = img.shape[:2]
        for _ in range(num_spots):
            cx = self.rng.randint(max_r, w - max_r)
            cy = self.rng.randint(max_r, h - max_r)
            r = self.rng.randint(min_r, max_r)
            cv2.circle(img, (cx, cy), r, color, -1)
            cv2.circle(img, (cx, cy), r + 1, (0, 0, 0), 1)
        return img

    def _add_brown_patch(self, img, center, radius, color=(40, 80, 120)):
        overlay = img.copy()
        cv2.circle(overlay, center, radius, color, -1)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
        result = overlay.astype(np.float32) * mask_3ch + img.astype(np.float32) * (1 - mask_3ch)
        return result.astype(np.uint8)

    def _add_yellowing(self, img, center, radius):
        return self._add_brown_patch(img, center, radius, color=(80, 180, 220))

    def _add_wilt_curl(self, img):
        h, w = img.shape[:2]
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        margin = self.rng.randint(10, 25)
        dst_pts = np.float32([
            [margin, margin],
            [w - margin, margin // 2],
            [w - margin // 2, h - margin],
            [margin, h - margin // 2],
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(img, M, (w, h))

    def generate_background(self):
        bg_type = self.rng.choice(["soil", "dark", "field"])

        if bg_type == "soil":
            bg = np.full((*self.img_size[::-1], 3), (60, 70, 80), dtype=np.uint8)
            noise = self.rng.normal(0, 15, bg.shape).astype(np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif bg_type == "dark":
            bg = self.rng.randint(10, 40, (*self.img_size[::-1], 3), dtype=np.uint8)
        else:
            bg = self.rng.randint(30, 80, (*self.img_size[::-1], 3), dtype=np.uint8)
            for _ in range(50):
                x = self.rng.randint(0, self.img_size[0])
                y = self.rng.randint(0, self.img_size[1])
                r = self.rng.randint(2, 8)
                green_val = self.rng.randint(40, 80)
                cv2.circle(bg, (x, y), r, (green_val, green_val + 20, green_val - 10), -1)

        return bg

    def generate_leaf_mask(self):
        h, w = self.img_size
        mask = np.zeros((h, w), dtype=np.uint8)

        cx = w // 2 + self.rng.randint(-20, 20)
        cy = h // 2 + self.rng.randint(-20, 20)
        rx = self.rng.randint(70, 95)
        ry = self.rng.randint(80, 100)
        angle = self.rng.randint(-15, 15)

        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 2)

        return mask, (cx, cy), (rx, ry), angle

    def generate_healthy(self):
        bg = self.generate_background()
        mask, center, axes, angle = self.generate_leaf_mask()

        base_green = np.array([
            self.rng.randint(50, 70),
            self.rng.randint(140, 190),
            self.rng.randint(40, 70),
        ], dtype=np.uint8)

        leaf = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
        leaf[:] = base_green

        noise = self.rng.normal(0, 12, leaf.shape).astype(np.int16)
        leaf = np.clip(leaf.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        for i in range(5):
            vein_len = axes[0] - 10
            self._draw_veins(
                leaf, center,
                angle + 90 + (i - 2) * 18,
                vein_len, thickness=1, color=(30, 100, 30)
            )
        self._draw_veins(leaf, center, angle, axes[0], thickness=2, color=(30, 90, 30))

        mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
        result = bg.astype(np.float32) * (1 - mask_3ch) + leaf.astype(np.float32) * mask_3ch
        return result.astype(np.uint8)

    def generate_mild(self):
        img = self.generate_healthy()
        mask, center, axes, angle = self.generate_leaf_mask()

        num_spots = self.rng.randint(3, 8)
        for _ in range(num_spots):
            sx = self.rng.randint(30, self.img_size[0] - 30)
            sy = self.rng.randint(30, self.img_size[1] - 30)
            r = self.rng.randint(3, 8)
            brown = (
                self.rng.randint(50, 80),
                self.rng.randint(90, 130),
                self.rng.randint(140, 180),
            )
            cv2.circle(img, (sx, sy), r, brown, -1)

        if self.rng.random() < 0.3:
            self._add_yellowing(img, center, self.rng.randint(15, 30))

        return img

    def generate_moderate(self):
        img = self.generate_healthy()
        mask, center, axes, angle = self.generate_leaf_mask()

        for _ in range(self.rng.randint(5, 12)):
            sx = self.rng.randint(20, self.img_size[0] - 20)
            sy = self.rng.randint(20, self.img_size[1] - 20)
            r = self.rng.randint(6, 16)
            brown = (
                self.rng.randint(30, 60),
                self.rng.randint(70, 110),
                self.rng.randint(120, 160),
            )
            cv2.circle(img, (sx, sy), r, brown, -1)
            cv2.circle(img, (sx, sy), r + 2, (40, 40, 40), 1)

        self._add_yellowing(img, center, self.rng.randint(25, 50))

        if self.rng.random() < 0.5:
            img = self._add_wilt_curl(img)

        for _ in range(self.rng.randint(3, 7)):
            sx = self.rng.randint(30, self.img_size[0] - 30)
            sy = self.rng.randint(30, self.img_size[1] - 30)
            r = self.rng.randint(2, 5)
            cv2.circle(img, (sx, sy), r, (20, 40, 80), -1)

        return img

    def generate_severe(self):
        img = self.generate_healthy()
        mask, center, axes, angle = self.generate_leaf_mask()

        for _ in range(self.rng.randint(10, 20)):
            sx = self.rng.randint(15, self.img_size[0] - 15)
            sy = self.rng.randint(15, self.img_size[1] - 15)
            r = self.rng.randint(8, 22)
            dark_brown = (
                self.rng.randint(15, 40),
                self.rng.randint(40, 80),
                self.rng.randint(80, 130),
            )
            cv2.circle(img, (sx, sy), r, dark_brown, -1)
            cv2.circle(img, (sx, sy), r + 2, (20, 20, 20), 2)

        self._add_yellowing(img, center, self.rng.randint(40, 70))

        cv2.circle(img, center, self.rng.randint(20, 40), (10, 30, 60), -1)

        img = self._add_wilt_curl(img)

        for _ in range(self.rng.randint(5, 10)):
            sx = self.rng.randint(20, self.img_size[0] - 20)
            sy = self.rng.randint(20, self.img_size[1] - 20)
            r = self.rng.randint(1, 4)
            cv2.circle(img, (sx, sy), r, (10, 20, 40), -1)

        return img

    def generate(self, severity: str) -> np.ndarray:
        generators = {
            "healthy": self.generate_healthy,
            "mild": self.generate_mild,
            "moderate": self.generate_moderate,
            "severe": self.generate_severe,
        }
        return generators[severity]()


def generate_dataset(
    images_per_class: int = 200,
    output_dir: str = None,
    img_size: Tuple[int, int] = (224, 224),
):
    if output_dir is None:
        output_dir = str(ROOT / "data" / "raw")

    output_path = Path(output_dir)
    gen = SyntheticLeafGenerator(img_size=img_size)

    total = images_per_class * len(CLASSES)
    print("\nGenerating {} synthetic images...".format(total))
    print("Output: {}\n".format(output_dir))

    count = 0
    for cls in CLASSES:
        cls_dir = output_path / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        print("  {}:".format(cls.upper()), end=" ", flush=True)
        for i in range(images_per_class):
            img = gen.generate(cls)
            filename = "synthetic_{}_{:04d}.jpg".format(cls, i)
            filepath = cls_dir / filename
            cv2.imwrite(str(filepath), img)
            count += 1

        print("{} images".format(images_per_class))

        pct = count / total * 100
        bar_len = 40
        filled = int(bar_len * count / total)
        bar = "=" * filled + "-" * (bar_len - filled)
        print("  [{}] {:.0f}%\n".format(bar, pct))

    print("Done! {} images generated.\n".format(count))

    print("Class distribution:")
    for cls in CLASSES:
        cls_dir = output_path / cls
        n = len(list(cls_dir.glob("*.jpg")))
        print("  {}: {} images".format(cls, n))

    return count


def visualize_samples(output_path: str = None, num_per_class: int = 4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = SyntheticLeafGenerator()

    fig, axes = plt.subplots(len(CLASSES), num_per_class, figsize=(4 * num_per_class, 4 * len(CLASSES)))
    fig.suptitle("Synthetic Leaf Samples by Severity Class", fontsize=16, y=1.02)

    for row, cls in enumerate(CLASSES):
        for col in range(num_per_class):
            ax = axes[row][col]
            img = gen.generate(cls)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if col == 0:
                ax.set_ylabel(cls.upper(), fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    if output_path is None:
        output_path = str(ROOT / "outputs" / "synthetic_samples.png")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Visualization saved: {}".format(output_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic leaf images")
    parser.add_argument(
        "--num", type=int, default=200,
        help="Images per class (default: 200)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualization of samples",
    )
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Generate 20 images per class for quick testing",
    )
    args = parser.parse_args()

    if args.visualize:
        visualize_samples()
    else:
        num = 20 if args.quick_test else args.num
        generate_dataset(images_per_class=num, output_dir=args.output)
