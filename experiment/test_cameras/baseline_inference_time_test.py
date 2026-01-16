# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import os
import torch
import torch.nn.functional as F
import argparse
import csv
import time
from pathlib import Path

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Baseline Camera Pose Inference Time Test")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", type=str, default=None, help="Path to the MP4 video file")
    input_group.add_argument("--images_dir", type=str, default=None, help="Path to directory containing images")
    
    # Output options
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for experiment results")
    
    # Video processing options
    parser.add_argument("--num_frames", type=int, default=None, 
                        help="Maximum number of frames to extract from video. If not provided, use all frames")
    parser.add_argument("--frame_indices", type=int, nargs="+", default=None, 
                        help="Specific frame indices to extract (0-based). If not provided, will extract frames sequentially")
    parser.add_argument("--extracted_frames_dir", type=str, default=None, 
                        help="Directory to save extracted frames. If not provided, will use a temporary directory")
    
    # Experiment options
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--vggt_resolution", type=int, default=518, help="VGGT model resolution")
    parser.add_argument("--img_load_resolution", type=int, default=1024, help="Image loading resolution")
    parser.add_argument("--max_images", type=int, default=None, 
                        help="Maximum number of images to test. If not provided, use all available images")
    parser.add_argument("--warmup_runs", type=int, default=1, 
                        help="Number of warmup runs before actual testing (default: 1)")
    
    return parser.parse_args()


def run_VGGT_cameras_only(model, images, dtype, resolution=518):
    """
    Run VGGT to extract camera poses only.
    
    Args:
        model: VGGT model
        images: Input images tensor [B, 3, H, W]
        dtype: Data type for computation
        resolution: Resolution for VGGT processing
    
    Returns:
        tuple: (extrinsic, intrinsic, inference_time) camera parameters and inference time in seconds
    """
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # Resize to VGGT resolution
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    # Start timing
    start_time = time.time()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # End timing
    end_time = time.time()
    inference_time = end_time - start_time

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    
    return extrinsic, intrinsic, inference_time


def save_results_to_csv(results, output_dir, filename="baseline_inference_time_results.csv"):
    """
    Save experiment results to CSV file.
    
    Args:
        results: List of dictionaries containing experiment results
        output_dir: Output directory path
        filename: CSV filename
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Define CSV headers
    fieldnames = [
        'num_images',
        'inference_time_seconds',
        'avg_time_per_image_seconds',
        'vggt_resolution',
        'img_load_resolution',
        'device',
        'dtype'
    ]
    
    # Write to CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nResults saved to {output_path}")
    return output_path


def run_incremental_test(model, all_images, device, dtype, args, results):
    """
    Run incremental test from 1 to N images.
    
    Args:
        model: VGGT model
        all_images: All loaded images tensor
        device: Device to run on
        dtype: Data type for computation
        args: Command line arguments
        results: List to store results
    
    Returns:
        list: Updated results list
    """
    total_images = all_images.shape[0]
    
    # Determine maximum number of images to test
    max_images = min(args.max_images, total_images) if args.max_images else total_images
    
    print(f"\n{'='*60}")
    print(f"Starting incremental inference time test")
    print(f"{'='*60}")
    print(f"Total available images: {total_images}")
    print(f"Testing from 1 to {max_images} images")
    print(f"VGGT resolution: {args.vggt_resolution}")
    print(f"Image load resolution: {args.img_load_resolution}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"{'='*60}\n")
    
    # Warmup run
    if args.warmup_runs > 0:
        print(f"Performing {args.warmup_runs} warmup run(s)...")
        for _ in range(args.warmup_runs):
            warmup_images = all_images[:1].clone()
            with torch.no_grad():
                _ = run_VGGT_cameras_only(model, warmup_images, dtype, args.vggt_resolution)
        print("Warmup completed.\n")
    
    # Run incremental tests
    for num_images in range(1, max_images + 1):
        # Select subset of images
        test_images = all_images[:num_images]
        
        # Clear cache before each test
        torch.cuda.empty_cache()
        
        # Run inference
        print(f"Testing with {num_images} image(s) [{num_images}/{max_images}]...", end=" ", flush=True)
        start_test = time.time()
        
        extrinsic, intrinsic, inference_time = run_VGGT_cameras_only(
            model, test_images, dtype, args.vggt_resolution
        )
        
        end_test = time.time()
        total_test_time = end_test - start_test
        
        avg_time_per_image = inference_time / num_images
        
        print(f"Done! Time: {inference_time:.4f}s, Avg: {avg_time_per_image:.4f}s/image")
        
        # Store result
        result = {
            'num_images': num_images,
            'inference_time_seconds': round(inference_time, 4),
            'avg_time_per_image_seconds': round(avg_time_per_image, 4),
            'vggt_resolution': args.vggt_resolution,
            'img_load_resolution': args.img_load_resolution,
            'device': device,
            'dtype': str(dtype)
        }
        results.append(result)
        
        # Save intermediate results
        if num_images % 5 == 0 or num_images == max_images:
            save_results_to_csv(results, args.output_dir, "baseline_inference_time_results.csv")
    
    return results


def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Load VGGT model
    print("\nLoading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.load("./models/model.pt"))
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Process input (video or images)
    if args.video_path is not None:
        # Process video file
        print(f"\nProcessing video: {args.video_path}")
        
        # Extract frames from video
        from vggt.utils.videos_utils import extract_frames_from_video
        
        if args.num_frames is None:
            # Extract all frames
            print("Extracting all frames from video...")
            frame_paths, frame_indices = extract_frames_from_video(
                args.video_path, 
                num_frames=100000,  # Large number to extract all frames
                frame_indices=None, 
                output_dir=args.extracted_frames_dir
            )
        elif args.frame_indices is not None:
            # Use specified frame indices
            print(f"Extracting specified frames: {args.frame_indices}")
            frame_paths, frame_indices = extract_frames_from_video(
                args.video_path, 
                num_frames=len(args.frame_indices), 
                frame_indices=args.frame_indices, 
                output_dir=args.extracted_frames_dir
            )
        else:
            # Extract specified number of frames
            print(f"Extracting {args.num_frames} frames from video...")
            frame_paths, frame_indices = extract_frames_from_video(
                args.video_path, 
                num_frames=args.num_frames, 
                frame_indices=None, 
                output_dir=args.extracted_frames_dir
            )
        
        image_names = [os.path.basename(path) for path in frame_paths]
        
    else:
        # Process images directory
        print(f"\nProcessing images from directory: {args.images_dir}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        frame_paths = []
        for ext in image_extensions:
            frame_paths.extend(glob.glob(os.path.join(args.images_dir, f"*{ext}")))
            frame_paths.extend(glob.glob(os.path.join(args.images_dir, f"*{ext.upper()}")))
        
        frame_paths = sorted(frame_paths)
        
        if len(frame_paths) == 0:
            raise ValueError(f"No images found in {args.images_dir}")
        
        print(f"Found {len(frame_paths)} images")
        image_names = [os.path.basename(path) for path in frame_paths]

    # Load all images at once
    print(f"\nLoading all {len(frame_paths)} images...")
    all_images, original_coords = load_and_preprocess_images_square(frame_paths, args.img_load_resolution)
    all_images = all_images.to(device)
    print(f"Images loaded with shape: {all_images.shape}")

    # Initialize results list
    results = []

    # Run incremental test
    results = run_incremental_test(model, all_images, device, dtype, args, results)

    # Save final results
    csv_path = save_results_to_csv(results, args.output_dir, "baseline_inference_time_results.csv")

    # Print summary
    print(f"\n{'='*60}")
    print("Experiment Summary")
    print(f"{'='*60}")
    print(f"Total tests: {len(results)}")
    print(f"Images tested: 1 to {results[-1]['num_images']}")
    print(f"Results saved to: {csv_path}")
    print(f"\nKey Results:")
    print(f"  - 1 image: {results[0]['inference_time_seconds']:.4f}s ({results[0]['avg_time_per_image_seconds']:.4f}s/image)")
    print(f"  - 5 images: {results[4]['inference_time_seconds']:.4f}s ({results[4]['avg_time_per_image_seconds']:.4f}s/image)")
    print(f"  - 10 images: {results[9]['inference_time_seconds']:.4f}s ({results[9]['avg_time_per_image_seconds']:.4f}s/image)")
    if len(results) >= 20:
        print(f"  - 20 images: {results[19]['inference_time_seconds']:.4f}s ({results[19]['avg_time_per_image_seconds']:.4f}s/image)")
    if len(results) >= 25:
        print(f"  - 25 images: {results[24]['inference_time_seconds']:.4f}s ({results[24]['avg_time_per_image_seconds']:.4f}s/image)")
    print(f"  - {results[-1]['num_images']} images: {results[-1]['inference_time_seconds']:.4f}s ({results[-1]['avg_time_per_image_seconds']:.4f}s/image)")
    print(f"{'='*60}")

    print("\nDone!")
    return True


if __name__ == "__main__":
    import glob
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)
