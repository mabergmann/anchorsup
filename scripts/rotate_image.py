import argparse
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from upright_anchor.scripts.rotation import synthesizeRotation


def compute_rotation_matrix(current_up, target_up):
    """Compute rotation matrix that aligns current_up with target_up.
    
    Args:
        current_up (np.ndarray): Current upright vector (3,)
        target_up (np.ndarray): Target upright vector (3,)
    
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    # Normalize vectors
    current_up = current_up / np.linalg.norm(current_up)
    target_up = target_up / np.linalg.norm(target_up)
    
    # If vectors are already aligned, return identity
    if np.allclose(current_up, target_up):
        return np.eye(3)
    
    # If vectors are opposite, rotate 180° around any perpendicular axis
    if np.allclose(current_up, -target_up):
        # Find any vector perpendicular to current_up
        perp = np.array([1, 0, 0]) if not np.allclose(current_up, [1, 0, 0]) else np.array([0, 1, 0])
        perp = perp - np.dot(perp, current_up) * current_up
        perp = perp / np.linalg.norm(perp)
        return R.from_rotvec(np.pi * perp).as_matrix()
    
    # Compute rotation axis and angle
    rotation_axis = np.cross(current_up, target_up)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    cos_angle = np.dot(current_up, target_up)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Create rotation matrix
    return R.from_rotvec(angle * rotation_axis).as_matrix()


def main():
    parser = argparse.ArgumentParser(description='Rotate an equirectangular image to align with a target upright vector')
    parser.add_argument('input_image', help='Path to input equirectangular image')
    parser.add_argument('output_image', help='Path to save rotated image')
    parser.add_argument('--target-up', type=float, nargs=3, default=[0, 0, 1],
                      help='Target upright vector (x y z)')
    parser.add_argument('--current-up', type=float, nargs=3, default=[0, 0, 1],
                      help='Current upright vector (x y z)')
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.input_image)
    if image is None:
        raise ValueError(f"Could not load image: {args.input_image}")
    
    # Compute rotation matrix
    R_align = compute_rotation_matrix(
        np.array(args.current_up),
        np.array(args.target_up)
    )
    
    # Apply rotation
    rotated_image = synthesizeRotation(image, R_align)
    
    # Save result
    cv2.imwrite(args.output_image, rotated_image)


if __name__ == "__main__":
    main() 