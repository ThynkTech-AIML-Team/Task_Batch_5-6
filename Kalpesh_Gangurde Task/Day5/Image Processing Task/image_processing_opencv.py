import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ----------------------------
# Configuration (easy to tweak)
# ----------------------------
CANNY_THRESHOLD_1 = 100
CANNY_THRESHOLD_2 = 200
BINARY_THRESHOLD_VALUE = 127
BRIGHTNESS_DELTA = 50


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_sample_image(project_dir: Path) -> Path:
    images = [p for p in project_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise FileNotFoundError("No image file found in project directory.")
    return sorted(images)[0]


def load_image_bgr(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def apply_canny(gray_image: np.ndarray, threshold1: int, threshold2: int) -> np.ndarray:
    return cv2.Canny(gray_image, threshold1, threshold2)


def apply_thresholds(gray_image: np.ndarray, binary_value: int) -> tuple[np.ndarray, np.ndarray]:
    _, binary = cv2.threshold(gray_image, binary_value, 255, cv2.THRESH_BINARY)
    adaptive = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return binary, adaptive


def adjust_brightness(image_bgr: np.ndarray, delta: int) -> tuple[np.ndarray, np.ndarray]:
    brighter = cv2.convertScaleAbs(image_bgr, alpha=1.0, beta=delta)
    darker = cv2.convertScaleAbs(image_bgr, alpha=1.0, beta=-delta)
    return brighter, darker


def apply_augmentations(image_bgr: np.ndarray, brightness_delta: int) -> dict[str, np.ndarray]:
    brighter, darker = adjust_brightness(image_bgr, brightness_delta)
    return {
        "Original": image_bgr,
        "Horizontal Flip": cv2.flip(image_bgr, 1),
        "Vertical Flip": cv2.flip(image_bgr, 0),
        "Rotate 90": cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE),
        "Rotate 180": cv2.rotate(image_bgr, cv2.ROTATE_180),
        "Brighter": brighter,
        "Darker": darker,
    }


def show_edge_detection(original_bgr: np.ndarray, edges: np.ndarray, t1: int, t2: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(bgr_to_rgb(original_bgr))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title(f"Canny Edges (t1={t1}, t2={t2})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def show_thresholding(gray_image: np.ndarray, binary: np.ndarray, adaptive: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gray_image, cmap="gray")
    axes[0].set_title("Grayscale")
    axes[0].axis("off")

    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title("Binary Threshold")
    axes[1].axis("off")

    axes[2].imshow(adaptive, cmap="gray")
    axes[2].set_title("Adaptive Threshold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def show_augmentations(augmentations: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, (title, image_bgr) in enumerate(augmentations.items()):
        axes[i].imshow(bgr_to_rgb(image_bgr))
        axes[i].set_title(title)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    image_path = find_sample_image(project_dir)
    print(f"Using image: {image_path.name}")

    image_bgr = load_image_bgr(image_path)
    gray = to_gray(image_bgr)

    edges = apply_canny(gray, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    show_edge_detection(image_bgr, edges, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)

    binary, adaptive = apply_thresholds(gray, BINARY_THRESHOLD_VALUE)
    show_thresholding(gray, binary, adaptive)

    aug = apply_augmentations(image_bgr, BRIGHTNESS_DELTA)
    show_augmentations(aug)


if __name__ == "__main__":
    main()
