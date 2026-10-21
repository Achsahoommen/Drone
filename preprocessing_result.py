import os
import random
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

DATASET_DIR = r"C:\Users\achsa\OneDrive\Documents\Default Project\data\raw"
classes = ["healthy", "mild", "moderate", "severe"]

resize = transforms.Resize((256, 256))

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for col, class_name in enumerate(classes):
    folder = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not files:
        raise ValueError(f"No images found in {folder}")

    image_path = os.path.join(folder, random.choice(files))

    # Load original image
    original = Image.open(image_path).convert("RGB")

    # Resize the image
    resized_image = resize(original)

    # Convert resized image explicitly to a PyTorch tensor
    tensor_image: torch.Tensor = TF.to_tensor(resized_image)

    # Display original image
    axes[0, col].imshow(original)
    axes[0, col].set_title(f"{class_name}\nOriginal")
    axes[0, col].axis("off")

    # Convert tensor from [C, H, W] to [H, W, C]
    processed_image = tensor_image.permute(1, 2, 0).numpy()

    # Display preprocessed image
    axes[1, col].imshow(processed_image)
    axes[1, col].set_title(f"{class_name}\nAfter Preprocessing")
    axes[1, col].axis("off")

plt.suptitle("Spinach Leaf Image Preprocessing", fontsize=15)
plt.tight_layout()

plt.savefig(
    "preprocessing_result.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved figure as preprocessing_result.png")
