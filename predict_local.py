from __future__ import annotations

import argparse
from pathlib import Path

from cifar10_inference import CIFAR10ResNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CIFAR-10 model on a local image.")
    parser.add_argument("image", nargs="?", default="img.jpg", help="Path to the image file")
    parser.add_argument("--weights", default="model.weights.h5", help="Path to the weights file")
    parser.add_argument(
        "--config", default="model_config.json", help="Path to the class/config metadata file"
    )
    parser.add_argument(
        "--architecture",
        default="config.json",
        help="Path to the saved model architecture JSON",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions to show")
    args = parser.parse_args()

    model = CIFAR10ResNet(
        weights_path=args.weights,
        config_path=args.config,
        architecture_path=args.architecture,
    )

    image_path = Path(args.image)
    predictions = model.top_k(image_path, k=args.top_k)

    print(f"Image: {image_path}")
    print("Top predictions:")
    for index, (label, probability) in enumerate(predictions, start=1):
        print(f"{index}. {label:12s} {probability:.4f}")

    top_label, top_probability = predictions[0]
    print(f"\nTop-1: {top_label} ({top_probability:.4f})")


if __name__ == "__main__":
    main()
