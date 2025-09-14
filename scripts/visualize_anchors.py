import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from upright_anchor.anchor_generation import create_anchor_generator
import hydra
from omegaconf import DictConfig
import numpy as np
import random


def plot_sphere_wireframe(ax, alpha=0.8, color='gray'):
    """Plot a transparent wireframe sphere."""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=alpha, color=color)


def plot_great_circle_arc(ax, v1, v2, color='r', n_points=100, linewidth=0.5):
    """Plot the great circle arc between two vectors on a unit sphere."""
    # Normalize vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Calculate the angle between vectors
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot_product)
    
    # Create points along the great circle
    t = np.linspace(0, 1, n_points)
    points = []
    for ti in t:
        # Spherical linear interpolation (SLERP)
        sin_angle = np.sin(angle)
        if sin_angle == 0:
            p = v1
        else:
            p = (np.sin((1-ti)*angle)/sin_angle) * v1 + (np.sin(ti*angle)/sin_angle) * v2
            p = p / np.linalg.norm(p)  # Normalize to ensure we stay on unit sphere
        points.append(p)
    
    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=linewidth)


def connect_nearby_anchors(ax, anchors, threshold=0.6):
    """Connect nearby anchor points with lines."""
    n_anchors = len(anchors)
    for i in range(n_anchors):
        for j in range(i+1, n_anchors):
            # Calculate dot product between anchors
            dot_product = np.dot(anchors[i], anchors[j])
            if dot_product > threshold:
                plot_great_circle_arc(ax, anchors[i], anchors[j])


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot white sphere in the background
    plot_sphere_wireframe(ax, alpha=0.3, color='white')
    
    # Generate anchors
    anchor_generator = create_anchor_generator(cfg)
    anchors = anchor_generator.generate()
    anchors_np = anchors.numpy()
    
    # Connect nearby anchors to create wireframe structure
    connect_nearby_anchors(ax, anchors_np, threshold=0.6)
    
    # Plot anchor points (smaller and more subtle)
    ax.scatter(anchors_np[:, 0], anchors_np[:, 1], anchors_np[:, 2], 
              color='black', s=10, alpha=0.5)
    
    # Select a unit vector closer to the viewer (negative z)
    random_anchor = [-0.2, -0.2, -0.8]
    random_anchor = random_anchor / np.linalg.norm(random_anchor)
    # Find its closest neighbor
    distances = np.array([np.arccos(np.clip(np.dot(random_anchor, anchor), -1.0, 1.0)) 
                         for anchor in anchors_np])
    closest_idx = np.argmin(distances)
    closest_anchor = anchors_np[closest_idx]
    
    # Highlight the random anchor and its closest neighbor
    ax.scatter([random_anchor[0]], [random_anchor[1]], [random_anchor[2]], 
              color='red', s=50, label='Random Anchor')
    ax.scatter([closest_anchor[0]], [closest_anchor[1]], [closest_anchor[2]], 
              color='blue', s=50, label='Closest Neighbor')
    
    # Plot the great circle arc between them
    plot_great_circle_arc(ax, random_anchor, closest_anchor, color='purple', linewidth=2)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Remove grid and background for cleaner look
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Save the plot
    plt.savefig('anchors_visualization.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


if __name__ == "__main__":
    main() 