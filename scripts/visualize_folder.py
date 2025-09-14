import cv2
import torch
import pathlib as pl
import numpy as np
from scipy.spatial.transform import Rotation as R
from upright_anchor.utils.model_setup import setup_model
from upright_anchor.scripts.rotate_image import compute_rotation_matrix
from upright_anchor.scripts.rotation import synthesizeRotation


def compute_upright_vector(prediction, anchors):
    """Compute the predicted upright vector from model predictions and anchors.
    
    Args:
        prediction: Model output tensor of shape (1, num_anchors, 3) containing
                   [classification_logits, ry, rx] for each anchor
        anchors: Tensor of shape (num_anchors, 3) containing anchor unit vectors
    
    Returns:
        numpy array of shape (3,) containing the predicted upright vector
    """
    # Get the predicted anchor index
    anchor_probs = torch.softmax(prediction[0, :, 0], dim=0)
    best_anchor_idx = torch.argmax(anchor_probs)
    
    # Get the predicted angles
    ry, rx = prediction[0, best_anchor_idx, 1:].cpu().numpy()
    
    # Get the best anchor
    best_anchor = anchors[best_anchor_idx].cpu().numpy()
    
    # Create rotation matrices
    R_y = R.from_euler('y', ry, degrees=False).as_matrix()
    R_x = R.from_euler('x', rx, degrees=False).as_matrix()
    
    # Apply rotations to the anchor
    rotated_vector = R_x @ R_y @ best_anchor
    
    return rotated_vector


def main():
    checkpoint_path = "upright_anchors/og7a6x7z/checkpoints/best_model.ckpt"
    checkpoint = torch.load(checkpoint_path)
    cfg = checkpoint['hyper_parameters']

    model, anchors = setup_model(cfg)
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    input_folder = pl.Path("../../GRAVITY-ALIGNMENT/data/test")
    output_folder = pl.Path("visualization")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Define target upright vector
    target_up = np.array([0, 0, 1])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for image_path in input_folder.glob("*.jpg"):
        # Read original image at full resolution
        original_image = cv2.imread(str(image_path))
        model_input = original_image[..., ::-1] / 255
        model_input = (model_input - mean) / std

        
        # Create resized copy for model input
        model_input = cv2.resize(model_input, (442, 221))
        image_tensor = torch.from_numpy(model_input).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            prediction = model(image_tensor)
            
        upright_vector = compute_upright_vector(prediction, anchors)
        
        # Compute rotation matrix to align predicted vector with [0, 0, 1]
        R_align = compute_rotation_matrix(upright_vector, target_up)

        # Apply rotation to the original image
        rotated_image = synthesizeRotation(original_image, R_align)
        
        # Save original and rotated images separately
        original_output_path = output_folder / f"{image_path.stem}_original.jpg"
        rotated_output_path = output_folder / f"{image_path.stem}_rotated.jpg"
        cv2.imwrite(str(original_output_path), original_image)
        cv2.imwrite(str(rotated_output_path), rotated_image)

if __name__ == "__main__":
    main()
