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
import json
import time
from pathlib import Path

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.videos_utils import extract_frames_from_video, get_video_info


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Camera Pose Estimation Only")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", type=str, default=None, help="Path to the MP4 video file")
    input_group.add_argument("--images_dir", type=str, default=None, help="Path to directory containing images")
    
    # Output options
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for camera pose results")
    parser.add_argument("--output_format", type=str, default="json", choices=["json", "npy", "txt"], 
                        help="Output format for camera poses")
    
    # Video processing options
    parser.add_argument("--num_frames", type=int, default=4, help="Number of frames to extract from video")
    parser.add_argument("--frame_indices", type=int, nargs="+", default=None, 
                        help="Specific frame indices to extract (0-based). If not provided, will randomly select frames")
    parser.add_argument("--extracted_frames_dir", type=str, default=None, 
                        help="Directory to save extracted frames. If not provided, will use a temporary directory")
    
    # Processing options
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--vggt_resolution", type=int, default=518, help="VGGT model resolution")
    parser.add_argument("--img_load_resolution", type=int, default=1024, help="Image loading resolution")
    
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
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # End timing
    end_time = time.time()
    inference_time = end_time - start_time

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    
    return extrinsic, intrinsic, inference_time


def save_camera_poses(extrinsic, intrinsic, image_names, output_dir, output_format="json"):
    """
    Save camera poses to file in specified format.
    
    Args:
        extrinsic: Extrinsic camera matrices [N, 3, 4]
        intrinsic: Intrinsic camera matrices [N, 3, 3]
        image_names: List of image names
        output_dir: Output directory path
        output_format: Output format (json, npy, txt)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_cameras = extrinsic.shape[0]
    camera_data = []
    
    for i in range(num_cameras):
        camera_info = {
            "image_name": image_names[i],
            "camera_id": i + 1,
            "extrinsic": extrinsic[i].tolist(),
            "intrinsic": intrinsic[i].tolist(),
            "rotation_matrix": extrinsic[i, :3, :3].tolist(),
            "translation_vector": extrinsic[i, :3, 3].tolist(),
            "focal_length": intrinsic[i, 0, 0],
            "principal_point": [intrinsic[i, 0, 2], intrinsic[i, 1, 2]]
        }
        camera_data.append(camera_info)
    
    if output_format == "json":
        output_path = os.path.join(output_dir, "camera_poses.json")
        with open(output_path, "w") as f:
            json.dump(camera_data, f, indent=2)
        print(f"Saved camera poses to {output_path}")
    
    elif output_format == "npy":
        output_path = os.path.join(output_dir, "camera_poses.npz")
        np.savez(output_path, 
                 extrinsic=extrinsic, 
                 intrinsic=intrinsic, 
                 image_names=np.array(image_names))
        print(f"Saved camera poses to {output_path}")
    
    elif output_format == "txt":
        output_path = os.path.join(output_dir, "camera_poses.txt")
        with open(output_path, "w") as f:
            for i, camera in enumerate(camera_data):
                f.write(f"Camera {i + 1}: {camera['image_name']}\n")
                f.write(f"  Extrinsic:\n{camera['extrinsic']}\n")
                f.write(f"  Intrinsic:\n{camera['intrinsic']}\n")
                f.write(f"  Rotation:\n{camera['rotation_matrix']}\n")
                f.write(f"  Translation: {camera['translation_vector']}\n")
                f.write(f"  Focal Length: {camera['focal_length']}\n")
                f.write(f"  Principal Point: {camera['principal_point']}\n")
                f.write("\n")
        print(f"Saved camera poses to {output_path}")


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
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Process input (video or images)
    if args.video_path is not None:
        # Process video file
        print(f"\nProcessing video: {args.video_path}")
        
        # Get video info
        video_info = get_video_info(args.video_path)
        print(f"Video info: {video_info}")
        
        # Extract frames from video
        print(f"\nExtracting frames from video...")
        frame_paths, frame_indices = extract_frames_from_video(
            args.video_path, 
            args.num_frames, 
            args.frame_indices, 
            args.extracted_frames_dir
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
        
        # Save frame indices info (all frames are used)
        frame_indices = list(range(len(frame_paths)))

    # Load images
    print(f"\nLoading {len(frame_paths)} images...")
    images, original_coords = load_and_preprocess_images_square(frame_paths, args.img_load_resolution)
    images = images.to(device)
    print(f"Images loaded with shape: {images.shape}")

    # Run VGGT to estimate camera poses
    print(f"\nRunning VGGT for camera pose estimation...")
    extrinsic, intrinsic, inference_time = run_VGGT_cameras_only(model, images, dtype, args.vggt_resolution)
    print(f"Camera poses estimated")
    print(f"  Extrinsic shape: {extrinsic.shape}")
    print(f"  Intrinsic shape: {intrinsic.shape}")
    print(f"  Inference time: {inference_time:.4f} seconds")

    # Save camera poses
    print(f"\nSaving camera poses to {args.output_dir}...")
    save_camera_poses(extrinsic, intrinsic, image_names, args.output_dir, args.output_format)

    # Save metadata
    metadata = {
        "input_type": "video" if args.video_path is not None else "images",
        "input_path": args.video_path if args.video_path is not None else args.images_dir,
        "num_images": len(frame_paths),
        "image_names": image_names,
        "frame_indices": frame_indices,
        "vggt_resolution": args.vggt_resolution,
        "img_load_resolution": args.img_load_resolution,
        "device": device,
        "dtype": str(dtype),
        "inference_time_seconds": round(inference_time, 4)
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Print summary
    print("\n" + "="*50)
    print("Camera Pose Estimation Summary")
    print("="*50)
    print(f"Number of cameras: {len(image_names)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output format: {args.output_format}")
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Average time per image: {inference_time / len(image_names):.4f} seconds")
    
    for i, name in enumerate(image_names):
        print(f"\nCamera {i+1}: {name}")
        print(f"  Focal length: {intrinsic[i, 0, 0]:.2f}")
        print(f"  Principal point: ({intrinsic[i, 0, 2]:.2f}, {intrinsic[i, 1, 2]:.2f})")
        print(f"  Translation: {extrinsic[i, :3, 3]}")

    print("\nDone!")
    return True


if __name__ == "__main__":
    import glob
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)
