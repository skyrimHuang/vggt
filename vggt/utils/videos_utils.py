import random
import os
import cv2
import tempfile
from PIL import Image


def extract_frames_from_video(video_path, num_frames, frame_indices=None, output_dir=None):
    """
    Extract frames from a video file.
    
    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of frames to extract
        frame_indices (list, optional): Specific frame indices to extract. If None, will randomly select frames
        output_dir (str, optional): Directory to save extracted frames. If None, will use a temporary directory
    
    Returns:
        tuple: (list of frame paths, list of frame indices used)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames")
    
    # Determine frame indices to extract
    if frame_indices is not None:
        # Validate provided frame indices
        frame_indices = [idx for idx in frame_indices if 0 <= idx < total_frames]
        if len(frame_indices) == 0:
            raise ValueError(f"No valid frame indices provided. Video has {total_frames} frames (0 to {total_frames-1})")
        if len(frame_indices) > num_frames:
            print(f"Warning: {len(frame_indices)} frame indices provided but only {num_frames} will be used")
            frame_indices = frame_indices[:num_frames]
        print(f"Using specified frame indices: {frame_indices}")
    else:
        # Randomly select frames
        if num_frames > total_frames:
            print(f"Warning: Requested {num_frames} frames but video only has {total_frames} frames")
            num_frames = total_frames
        frame_indices = sorted(random.sample(range(total_frames), num_frames))
        print(f"Randomly selected frame indices: {frame_indices}")
    
    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="vggt_frames_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames
    frame_paths = []
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {frame_idx}")
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save frame
        frame_filename = f"frame_{frame_idx:06d}.png"
        frame_path = os.path.join(output_dir, frame_filename)
        Image.fromarray(frame_rgb).save(frame_path)
        frame_paths.append(frame_path)
        print(f"Extracted frame {frame_idx} to {frame_path}")
    
    cap.release()
    
    if len(frame_paths) == 0:
        raise ValueError("Failed to extract any frames from the video")
    
    print(f"Extracted {len(frame_paths)} frames to {output_dir}")
    return frame_paths, frame_indices


def get_video_info(video_path):
    """
    Get basic information about a video file.
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        dict: Dictionary containing video information (fps, frame_count, width, height, duration)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    info = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }
    
    return info


def validate_frame_indices(frame_indices, total_frames):
    """
    Validate frame indices against total frame count.
    
    Args:
        frame_indices (list): List of frame indices to validate
        total_frames (int): Total number of frames in the video
    
    Returns:
        list: List of valid frame indices
    """
    valid_indices = [idx for idx in frame_indices if 0 <= idx < total_frames]
    if len(valid_indices) == 0:
        raise ValueError(f"No valid frame indices. Video has {total_frames} frames (0 to {total_frames-1})")
    return valid_indices


def generate_uniform_frame_indices(total_frames, num_frames):
    """
    Generate uniformly distributed frame indices.
    
    Args:
        total_frames (int): Total number of frames in the video
        num_frames (int): Number of frames to select
    
    Returns:
        list: List of uniformly distributed frame indices
    """
    if num_frames > total_frames:
        num_frames = total_frames
    
    if num_frames == 1:
        return [total_frames // 2]
    
    step = (total_frames - 1) / (num_frames - 1)
    indices = [int(i * step) for i in range(num_frames)]
    return indices
