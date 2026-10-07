import os
import sys
import argparse
from PIL import Image

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from src.models import build_resnet18_feature_extractor
from src.dataset import get_transforms
from src.utils import load_checkpoint

def predict_single_image(image_path: str, checkpoint_path: str, device: str = "cpu"):
    """
    Loads saved model checkpoint and runs inference on a single image.
    Returns predicted class, confidence score, and top-5 predictions.
    """
    device = torch.device(device)

    # 1. Load checkpoint metadata first to get number of classes and mapping
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    num_classes = len(classes)

    # 2. Build model architecture & reload weights
    model = build_resnet18_feature_extractor(num_classes=num_classes, freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Preprocess image
    transform = get_transforms(is_train=False)
    image = Image.open(image_path).convert("RGB")
    tensor_img = transform(image).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)

    top_prob, top_idx = torch.topk(probabilities, k=min(5, num_classes))

    top5_results = []
    for prob, idx in zip(top_prob.tolist(), top_idx.tolist()):
        top5_results.append({
            "class": classes[idx],
            "confidence": float(prob)
        })

    predicted_class = classes[top_idx[0].item()]
    confidence = float(top_prob[0].item())

    return {
        "image_path": image_path,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top5": top5_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-image inference with Plant Disease ResNet18 model")
    parser.add_argument("--image", type=str, required=True, help="Path to input leaf image")
    parser.add_argument("--checkpoint", type=str, default="models/resnet18_feature_extractor_smoke.pth", help="Path to checkpoint .pth")
    args = parser.parse_args()

    res = predict_single_image(args.image, args.checkpoint)
    print("\n--- Inference Result ---")
    print(f"Image: {res['image_path']}")
    print(f"Predicted Class: {res['predicted_class']}")
    print(f"Confidence: {res['confidence']:.4f} ({res['confidence']*100:.2f}%)")
    print("\nTop 5 Candidates:")
    for i, c in enumerate(res["top5"], 1):
        print(f"  {i}. {c['class']}: {c['confidence']*100:.2f}%")
