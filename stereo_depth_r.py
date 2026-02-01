import numpy as np
import cv2

def get_depth_map_kitti_optimized(img_l, img_r, P2, P3):
    """
    StereoSGBM parameter rationale (current settings):
    - minDisparity=0: KITTI disparity typically starts at 0; keep default offset.
    - numDisparities=16*10: Must be a multiple of 16; expanded range for near objects.
    - blockSize=7: Mid-size odd window (3-11 recommended); balances noise vs detail.
    - P1=8*3*block_size^2: Penalty for small disparity changes; encourages mild smoothness.
    - P2=32*3*block_size^2: Penalty for larger jumps; stronger smoothness across edges.
    - disp12MaxDiff=1: Tight left-right consistency to reduce mismatches.
    - preFilterCap=63: Strong prefilter clipping for stable BT matching cost.
    - uniquenessRatio=10: Mid-range ambiguity filter (5-15 typical).
    - speckleWindowSize=200: Aggressive speckle removal for large planar regions.
    - speckleRange=2: Allows modest disparity variation within speckle regions.
    - mode=SGBM_3WAY: Higher accuracy aggregation than default SGBM.
    """
    # Compute focal length and baseline
    focal_length = P2[0, 0]
    baseline = abs(P2[0, 3] - P3[0, 3]) / focal_length

    min_disparity = 0          # KITTI minimum disparity is 0
    num_disparities = 16 * 10  # Increase search range; 16 * 10 or more is suitable
    block_size = 7             # Moderate size, balances accuracy and speed

    # Parameters optimized for KITTI road and vehicle scenarios
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,      # Matching cost penalty 1 (small gradients)
        P2=32 * 3 * block_size**2,     # Matching cost penalty 2 (large gradients)
        disp12MaxDiff=1,               # Max difference in left-right consistency check
        preFilterCap=63,               # Pre-filter cap value
        uniquenessRatio=10,            # Uniqueness ratio
        speckleWindowSize=200,         # Speckle filter window size
        speckleRange=2,               # Speckle filter range
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Use 3-way SGBM for higher accuracy
    )

    # Compute disparity
    disparity = stereo.compute(img_l, img_r).astype(np.float32) / 16.0

    # Handle invalid regions
    # far objects have small disparity, near objects have large disparity
    disparity[disparity <= 0] = 0.1  # Avoid divide-by-zero errors
    # Set a reasonable maximum disparity cap
    max_valid_disparity = num_disparities - 1
    disparity[disparity > max_valid_disparity] = max_valid_disparity

    # Convert to depth map
    depth_map = (focal_length * baseline) / disparity

    depth_map[:, :num_disparities] = 0

    # KITTI depth clipping (typical range)
    # Most valid KITTI depths are within 3-80 meters
    depth_map = np.clip(depth_map, 1.0, 100.0)  # Limit depth range

    return depth_map

