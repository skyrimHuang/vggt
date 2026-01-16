# Baseline Inference Time Test

## Description

This script performs baseline inference time testing for VGGT camera pose estimation without any optimizations. It tests how inference time scales with the number of input images.

## Features

- **Incremental Testing**: Tests from 1 to N images (incrementing by 1 each time)
- **Progress Tracking**: Prints real-time progress during testing
- **CSV Output**: Saves detailed results to CSV file
- **Warmup Runs**: Optional warmup runs to stabilize performance
- **Flexible Input**: Supports both video files and image directories

## Usage

### Basic Usage

```bash
# Test with images from a directory
python baseline_inference_time_test.py --images_dir /path/to/images --output_dir ./results

# Test with video file
python baseline_inference_time_test.py --video_path /path/to/video.mp4 --output_dir ./results --num_frames 25

# Test with specific frame indices from video
python baseline_inference_time_test.py --video_path /path/to/video.mp4 --output_dir ./results --frame_indices 0 10 20 30 40
```

### Advanced Options

```bash
# Limit maximum number of images to test
python baseline_inference_time_test.py --images_dir /path/to/images --output_dir ./results --max_images 20

# Change VGGT resolution
python baseline_inference_time_test.py --images_dir /path/to/images --output_dir ./results --vggt_resolution 448

# Change image loading resolution
python baseline_inference_time_test.py --images_dir /path/to/images --output_dir ./results --img_load_resolution 768

# Set number of warmup runs
python baseline_inference_time_test.py --images_dir /path/to/images --output_dir ./results --warmup_runs 2
```

## Arguments

### Input Options (Required, choose one)
- `--video_path`: Path to MP4 video file
- `--images_dir`: Path to directory containing images

### Output Options
- `--output_dir`: Output directory for experiment results (required)

### Video Processing Options
- `--num_frames`: Maximum number of frames to extract from video (default: extract all frames)
- `--frame_indices`: Specific frame indices to extract (0-based)
- `--extracted_frames_dir`: Directory to save extracted frames (default: temporary directory)

### Experiment Options
- `--seed`: Random seed for reproducibility (default: 42)
- `--vggt_resolution`: VGGT model resolution (default: 518)
- `--img_load_resolution`: Image loading resolution (default: 1024)
- `--max_images`: Maximum number of images to test (default: use all available)
- `--warmup_runs`: Number of warmup runs before testing (default: 1)

## Output

### CSV File

The script generates a CSV file named `baseline_inference_time_results.csv` with the following columns:

| Column | Description |
|---------|-------------|
| `num_images` | Number of images in the test |
| `inference_time_seconds` | Total inference time in seconds |
| `avg_time_per_image_seconds` | Average time per image in seconds |
| `vggt_resolution` | VGGT model resolution used |
| `img_load_resolution` | Image loading resolution used |
| `device` | Device used (cuda/cpu) |
| `dtype` | Data type used (bfloat16/float16) |

### Console Output

The script prints:
- Configuration information
- Progress for each test (1 to N images)
- Inference time and average time per image for each test
- Summary of key results

## Example Output

```
Arguments: {'video_path': None, 'images_dir': './test_images', ...}
Setting seed as: 42
Using device: cuda
Using dtype: torch.bfloat16

Loading VGGT model...
Model loaded

Processing images from directory: ./test_images
Found 25 images

Loading all 25 images...
Images loaded with shape: torch.Size([25, 3, 1024, 1024])

============================================================
Starting incremental inference time test
============================================================
Total available images: 25
Testing from 1 to 25 images
VGGT resolution: 518
Image load resolution: 1024
Device: cuda
Dtype: torch.bfloat16
============================================================

Performing 1 warmup run(s)...
Warmup completed.

Testing with 1 image(s) [1/25]... Done! Time: 0.2345s, Avg: 0.2345s/image
Testing with 2 image(s) [2/25]... Done! Time: 0.4123s, Avg: 0.2062s/image
Testing with 3 image(s) [3/25]... Done! Time: 0.5987s, Avg: 0.1996s/image
...
Testing with 25 image(s) [25/25]... Done! Time: 5.1234s, Avg: 0.2049s/image

Results saved to ./results/baseline_inference_time_results.csv

============================================================
Experiment Summary
============================================================
Total tests: 25
Images tested: 1 to 25
Results saved to: ./results/baseline_inference_time_results.csv

Key Results:
  - 1 image: 0.2345s (0.2345s/image)
  - 5 images: 1.0234s (0.2047s/image)
  - 10 images: 2.0456s (0.2046s/image)
  - 20 images: 4.0891s (0.2045s/image)
  - 25 images: 5.1234s (0.2049s/image)
============================================================

Done!
```

## Notes

- This is a **baseline** experiment with no optimizations applied
- Results should be compared with optimized versions to measure improvements
- The script saves intermediate results every 5 tests to prevent data loss
- GPU memory is cleared between tests using `torch.cuda.empty_cache()`
- Warmup runs help stabilize performance by initializing GPU kernels

## Future Experiments

After running this baseline test, you can compare results with:
1. **Resolution optimization**: Lower VGGT resolution
2. **Sparse attention**: Enable Flash Attention
3. **torch.compile**: Use PyTorch 2.0 compilation
4. **Early exit**: Exit from earlier layers
5. **Dynamic resolution**: Adjust resolution based on image complexity
6. **Batch processing**: Optimize batch sizes

Each optimized version should have a similar naming convention:
- `baseline_inference_time_test.py` (this file)
- `resolution_optimization_test.py`
- `sparse_attention_test.py`
- `torch_compile_test.py`
- etc.

This makes it easy to compare results across different optimization strategies.
