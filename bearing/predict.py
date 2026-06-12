import argparse
import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def predict_image(model, image_path: Path, transform, classes, threshold: float, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu()

    probabilities = {
        label: round(float(prob), 6)
        for label, prob in zip(classes, probs)
    }
    predicted_labels = [
        label
        for label, prob in probabilities.items()
        if prob >= threshold
    ]

    return {
        "image": str(image_path),
        "threshold": threshold,
        "predicted_labels": predicted_labels,
        "probabilities": probabilities,
    }


def collect_images(input_path: Path):
    if input_path.is_file():
        return [input_path]

    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in valid_suffixes
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Predict labels from a trained EfficientNet model.")
    parser.add_argument(
        "input",
        type=Path,
        help="Path ke file gambar atau folder berisi gambar.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pth"),
        help="Path checkpoint model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold multi-label prediction.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device inference.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output dalam format JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input tidak ditemukan: {args.input}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {args.checkpoint}")

    device = torch.device(args.device)
    model, classes = load_model(args.checkpoint, device)
    transform = build_transform()
    image_paths = collect_images(args.input)

    if not image_paths:
        raise FileNotFoundError(f"Tidak ada gambar yang ditemukan di: {args.input}")

    results = [
        predict_image(model, image_path, transform, classes, args.threshold, device)
        for image_path in image_paths
    ]

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
        return

    print(f"Classes: {', '.join(classes)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print()

    for result in results:
        print(f"Gambar: {result['image']}")
        print(f"Prediksi: {result['predicted_labels'] or ['tidak ada label di atas threshold']}")
        for label, prob in result["probabilities"].items():
            print(f"  - {label}: {prob:.4f}")
        print()


if __name__ == "__main__":
    main()
