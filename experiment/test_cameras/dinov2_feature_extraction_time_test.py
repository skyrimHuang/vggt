import os
import time
import csv
import argparse
from glob import glob
import torch
import torch.nn.functional as F
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

def parse_args():
    parser = argparse.ArgumentParser(description="Test DinoV2 feature extraction time")
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images to process"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./dinov2_feature_extraction_times.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )
    return parser.parse_args()

def load_image(image_path, img_size=518):
    """Load and preprocess an image"""
    image = Image.open(image_path).convert("RGB")
    # Resize image to match model input size
    image = image.resize((img_size, img_size))
    # Convert to tensor and normalize
    image = torch.tensor(list(image.getdata()), dtype=torch.float32).view(img_size, img_size, 3)
    image = image.permute(2, 0, 1) / 255.0  # (3, H, W) in [0, 1]
    # Apply ResNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std
    return image

def get_model(model_type, device):
    """Initialize and load DinoV2 model from VGGT weights"""
    # Load the complete VGGT model with pretrained weights
    print("Loading VGGT model with pretrained weights...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.load("./../../models/model.pt"))
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model.to(device)
    print("VGGT model loaded successfully")
    
    # Extract the DinoV2 feature extractor from the aggregator
    # The patch_embed in aggregator is the DinoV2 model
    dinov2_model = model.aggregator.patch_embed
    dinov2_model.eval()
    dinov2_model.to(device)
    print("Extracted DinoV2 feature extractor")
    
    return dinov2_model

def extract_features(model, image, device):
    """Extract features from an image using DinoV2"""
    with torch.no_grad():
        # Add batch dimension
        image = image.unsqueeze(0).to(device)
        # Forward pass to get features
        start_time = time.time()
        # For the DinoV2 model extracted from VGGT, we need to use the forward_features method
        # This is because the patch_embed in aggregator is a DinoVisionTransformer instance
        features = model.forward_features(image)
        end_time = time.time()
        extraction_time = end_time - start_time
    return features, extraction_time

def main():
    args = parse_args()
    
    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(args.images_dir, f"*{ext}")))
    
    if not image_files:
        print(f"No images found in directory: {args.images_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Using device: {args.device}")
    
    # Load model
    model = get_model("dinov2_vitl14_reg", args.device)  # model_type is not used anymore
    
    # Process images and collect times
    results = []
    total_time = 0.0
    
    print("\nProcessing images...")
    for i, image_path in enumerate(image_files):
        try:
            # Load and preprocess image
            image = load_image(image_path)
            
            # Extract features and measure time
            features, extraction_time = extract_features(model, image, args.device)
            
            # Store results
            results.append({
                "image": os.path.basename(image_path),
                "extraction_time": extraction_time
            })
            
            total_time += extraction_time
            
            # Print progress
            print(f"[{i+1}/{len(image_files)}] Processed {os.path.basename(image_path)} in {extraction_time:.4f}s")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Calculate average time
    if results:
        average_time = total_time / len(results)
        print(f"\nProcessing complete!")
        print(f"Total time: {total_time:.4f}s")
        print(f"Average time per image: {average_time:.4f}s")
        
        # Save results to CSV
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        with open(args.output_csv, 'w', newline='') as csvfile:
            fieldnames = ['image', 'extraction_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
            
            # Add summary row
            writer.writerow({'image': 'TOTAL', 'extraction_time': total_time})
            writer.writerow({'image': 'AVERAGE', 'extraction_time': average_time})
        
        print(f"Results saved to: {args.output_csv}")
    else:
        print("No images processed successfully")

if __name__ == "__main__":
    main()