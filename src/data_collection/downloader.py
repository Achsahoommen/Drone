"""
Spinach Leaf Dataset Downloader
Downloads free public datasets and maps them to our 4-class system:
  healthy, mild, moderate, severe

Available sources:
  1. PlantVillage (54K images, 14 crops) - largest, general plant disease
  2. IDDMSLD Spinach (3,006 images) - spinach-specific, from HuggingFace
"""

import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path

from utils.config import get_project_root, setup_logging, CLASSES

logger = setup_logging("dataset_downloader")

ROOT = get_project_root()
RAW_DIR = ROOT / "data" / "raw"


def download_with_progress(url, dest_path):
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print("\r  Downloading: {:.1f}/{:.1f} MB ({:.0f}%)".format(
                mb, total_mb, pct
            ), end="", flush=True)

    print("  URL: {}".format(url[:80] + "..." if len(url) > 80 else url))
    urllib.request.urlretrieve(url, str(dest_path), reporthook=progress)
    print()


def download_plantvillage():
    """
    Download PlantVillage dataset from HuggingFace using huggingface_hub.
    Maps tomato disease classes to our 4 spinach severity classes.
    """
    logger.info("Downloading PlantVillage dataset...")
    output_dir = ROOT / "data" / "plantvillage_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        print("  Loading PlantVillage from HuggingFace (this may take a few minutes)...")
        dataset = load_dataset("mohanty/PlantVillage", "color", cache_dir=str(output_dir / "cache"))

        class_mapping = _get_plantvillage_class_mapping()

        for split_name in ["train"]:
            if split_name not in dataset:
                continue
            split_data = dataset[split_name]
            logger.info("Processing {} split ({} images)...".format(
                split_name, len(split_data)
            ))

            for idx, item in enumerate(split_data):
                label_name = item["label"]
                class_name = class_mapping.get(label_name, None)
                if class_name is None:
                    continue

                dest_dir = RAW_DIR / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)

                img = item["image"]
                img_path = dest_dir / "pv_{}_{:06d}.jpg".format(split_name, idx)
                img.save(str(img_path), quality=95)

                if (idx + 1) % 500 == 0:
                    print("  Processed {}/{} images".format(
                        idx + 1, len(split_data)
                    ))

        logger.info("PlantVillage download complete")
        _print_class_counts()
        return True

    except ImportError:
        logger.warning("huggingface_hub not installed. Trying alternative download...")
        return _download_plantvillage_manual(output_dir)
    except Exception as e:
        logger.error("Failed: {}".format(e))
        return False


def _get_plantvillage_class_mapping():
    """
    Map PlantVillage classes to our 4-class severity system.

    PlantVillage tomato classes we use:
      healthy -> healthy
      bacterial_spot -> mild
      early_blight -> moderate
      late_blight -> severe
      leaf_mold -> moderate
      septoria_leaf_spot -> moderate
      spider_mites -> mild
      target_spot -> moderate
      yellow_leaf_curl_virus -> severe
      mosaic_virus -> moderate
    """
    mapping = {}

    pv_classes = {
        0: "Tomato___Bacterial_spot",
        1: "Tomato___Early_blight",
        2: "Tomato___Late_blight",
        3: "Tomato___Leaf_Mold",
        4: "Tomato___Septoria_leaf_spot",
        5: "Tomato___Spider_mites Two-spotted_spider_mite",
        6: "Tomato___Target_Spot",
        7: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        8: "Tomato___Tomato_mosaic_virus",
        9: "Tomato___healthy",
    }

    severity_map = {
        "healthy": "healthy",
        "Bacterial_spot": "mild",
        "Early_blight": "moderate",
        "Late_blight": "severe",
        "Leaf_Mold": "moderate",
        "Septoria_leaf_spot": "moderate",
        "Spider_mites": "mild",
        "Target_Spot": "moderate",
        "Yellow_Leaf_Curl_Virus": "severe",
        "mosaic_virus": "moderate",
    }

    for idx, full_name in pv_classes.items():
        disease = full_name.replace("Tomato___", "")
        if disease == "healthy":
            mapped = "healthy"
        else:
            mapped = None
            for key, val in severity_map.items():
                if key.lower() in disease.lower():
                    mapped = val
                    break
            if mapped is None:
                mapped = "mild"
        mapping[idx] = mapped

    return mapping


def _download_plantvillage_manual(output_dir):
    """Fallback: download the raw zip from GitHub."""
    url = (
        "https://github.com/spMohanty/PlantVillage-Dataset/"
        "raw/master/raw/color"
    )
    zip_url = (
        "https://figshare.com/ndownloader/files/55035897"
    )

    logger.info("Trying direct download...")
    zip_path = output_dir / "plantvillage.zip"

    try:
        print("  This may take a while (400MB+)...")
        download_with_progress(zip_url, zip_path)
        print("  Extracting...")
        with zipfile.ZipFile(str(zip_path), "r") as z:
            z.extractall(str(output_dir))
        logger.info("Extracted to {}".format(output_dir))
        return True
    except Exception as e:
        logger.error("Direct download failed: {}".format(e))
        logger.info(
            "\n  MANUAL INSTRUCTIONS:\n"
            "  1. Go to: https://www.kaggle.com/datasets/emmarex/plantdisease\n"
            "  2. Download the PlantVillage dataset\n"
            "  3. Extract to: {}\n"
            "  4. Run this script again\n".format(output_dir)
        )
        return False


def download_spinach_specific():
    """
    Download the IDDMSLD Spinach dataset from HuggingFace.
    Uses parquet API for fast download (avoids slow datasets library).
    Contains 3,006 images of Malabar spinach with 5 disease classes.
    Maps to our 4-class system.
    """
    logger.info("Downloading IDDMSLD Spinach dataset...")
    output_dir = ROOT / "data" / "spinach_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        from PIL import Image
        import io

        print("  Downloading parquet file from HuggingFace...")
        parquet_path = hf_hub_download(
            repo_id="Project-AgML/IDDMSLD_spinach_leaf_disease_classification",
            filename="data/train-00000-of-00001.parquet",
            repo_type="dataset",
        )
        print("  Downloaded: {}".format(parquet_path))

        print("  Reading parquet...")
        df = pd.read_parquet(parquet_path)
        print("  Loaded {} rows".format(len(df)))

        label_col = "label"
        if label_col not in df.columns:
            for col in df.columns:
                if col != "image" and col != "Filename":
                    label_col = col
                    break

        class_names = sorted(df[label_col].unique())
        print("  Classes found: {}".format(class_names))

        label_to_class = {}
        for name in class_names:
            name_lower = str(name).lower()
            if "healthy" in name_lower:
                label_to_class[name] = "healthy"
            elif "bacterial" in name_lower:
                label_to_class[name] = "mild"
            elif "pest" in name_lower:
                label_to_class[name] = "mild"
            elif "downy" in name_lower:
                label_to_class[name] = "moderate"
            elif "anthracnose" in name_lower:
                label_to_class[name] = "severe"
            else:
                label_to_class[name] = "mild"

        print("  Mapping: {}".format(label_to_class))

        for cls in CLASSES:
            (RAW_DIR / cls).mkdir(parents=True, exist_ok=True)

        counts = {cls: 0 for cls in CLASSES}
        for idx, row in df.iterrows():
            label = row[label_col]
            class_name = label_to_class.get(label, "mild")

            img_data = row["image"]
            if isinstance(img_data, dict) and "bytes" in img_data:
                img = Image.open(io.BytesIO(img_data["bytes"]))
            elif isinstance(img_data, Image.Image):
                img = img_data
            else:
                continue

            dest = RAW_DIR / class_name / "spinach_{:06d}.jpg".format(idx)
            img.save(str(dest), quality=95)
            counts[class_name] += 1

            if (idx + 1) % 500 == 0:
                print("  Processed {}/{} images".format(idx + 1, len(df)))

        print("\n  Download complete!")
        for cls in CLASSES:
            print("    {}: {} images".format(cls, counts[cls]))

        _print_class_counts()
        return True

    except ImportError:
        logger.error("Please install: pip install huggingface_hub pandas pillow")
        return False
    except Exception as e:
        logger.error("Failed: {}".format(e))
        import traceback
        traceback.print_exc()
        return False


def download_from_url(url, extract=True):
    """Generic download and extract."""
    filename = url.split("/")[-1].split("?")[0]
    dest = ROOT / "data" / "downloads"
    dest.mkdir(parents=True, exist_ok=True)
    filepath = dest / filename

    logger.info("Downloading: {}".format(filename))
    download_with_progress(url, filepath)

    if extract and filepath.suffix == ".zip":
        logger.info("Extracting...")
        with zipfile.ZipFile(str(filepath), "r") as z:
            z.extractall(str(dest))
        logger.info("Extracted to: {}".format(dest))

    return filepath


def _print_class_counts():
    print("\nClass distribution in data/raw/:")
    for cls in CLASSES:
        cls_dir = RAW_DIR / cls
        if cls_dir.exists():
            count = len(list(cls_dir.glob("*.*")))
            print("  {}: {} images".format(cls, count))
        else:
            print("  {}: 0 images (directory not found)".format(cls))


def list_available_datasets():
    print("\n" + "=" * 60)
    print("  AVAILABLE FREE SPINACH/PLANT LEAF DATASETS")
    print("=" * 60)

    datasets = [
        {
            "name": "IDDMSLD Spinach Leaf Disease (RECOMMENDED)",
            "url": "https://huggingface.co/datasets/Project-AgML/IDDMSLD_spinach_leaf_disease_classification",
            "images": 3006,
            "classes": "Anthracnose, Bacterial-Spot, Downy-Mildew, Healthy, Pest-Damage",
            "command": "python -m src.data_collection.downloader --source spinach",
        },
        {
            "name": "PlantVillage (54K images, 14 crops)",
            "url": "https://huggingface.co/datasets/mohanty/PlantVillage",
            "images": 54306,
            "classes": "38 classes (mapped to 4 severity levels)",
            "command": "python -m src.data_collection.downloader --source plantvillage",
        },
        {
            "name": "Local Spinach Leaf (Mendeley)",
            "url": "https://data.mendeley.com/datasets/skf6w2s2h2/2",
            "images": 3999,
            "classes": "Fresh/Non-fresh spinach varieties",
            "command": "python -m src.data_collection.downloader --url <MENDELEY_URL>",
        },
        {
            "name": "Kaggle Plant Disease",
            "url": "https://www.kaggle.com/datasets/emmarex/plantdisease",
            "images": 54309,
            "classes": "Tomato, Potato, Pepper diseases",
            "command": "python -m src.data_collection.downloader --url <KAGGLE_URL>",
        },
    ]

    for i, ds in enumerate(datasets, 1):
        print("\n  [{}] {}".format(i, ds["name"]))
        print("      Images: {}".format(ds["images"]))
        print("      Classes: {}".format(ds["classes"]))
        print("      URL: {}".format(ds["url"]))
        print("      Run: {}".format(ds["command"]))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download spinach leaf datasets")
    parser.add_argument(
        "--source",
        choices=["plantvillage", "spinach", "all", "list"],
        default="list",
        help="Dataset to download",
    )
    parser.add_argument("--url", type=str, help="Custom download URL")
    args = parser.parse_args()

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

    _print_class_counts()
