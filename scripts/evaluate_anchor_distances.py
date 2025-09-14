import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from upright_anchor.anchor_generation import create_anchor_generator


def compute_anchor_distances(anchors):
    """Compute the angular distance between each anchor and its closest neighbor.
    
    Args:
        anchors: numpy array of shape (N, 3) containing unit vectors
        
    Returns:
        distances: numpy array of shape (N,) containing the angular distance to closest neighbor
        indices: numpy array of shape (N,) containing the indices of closest neighbors
    """
    n_anchors = len(anchors)
    distances = np.zeros(n_anchors)
    indices = np.zeros(n_anchors, dtype=int)
    
    for i in range(n_anchors):
        # Compute dot products with all other anchors
        dot_products = np.clip(np.dot(anchors, anchors[i]), -1.0, 1.0)
        # Set self-distance to -1 to exclude it from minimum calculation
        dot_products[i] = -1
        # Find closest neighbor (maximum dot product)
        closest_idx = np.argmax(dot_products)
        # Convert to angular distance
        distances[i] = np.arccos(dot_products[closest_idx])
        indices[i] = closest_idx
    
    return distances, indices


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Generate anchors
    anchor_generator = create_anchor_generator(cfg)
    anchors = anchor_generator.generate()
    anchors_np = anchors.numpy()
    
    # Compute distances
    distances, indices = compute_anchor_distances(anchors_np)
    
    # Convert distances to degrees for better readability
    distances_deg = np.degrees(distances)
    
    # Print statistics
    print(f"Statistics of angular distances to closest neighbors:")
    print(f"Mean distance: {np.mean(distances_deg):.2f}°")
    print(f"Median distance: {np.median(distances_deg):.2f}°")
    print(f"Min distance: {np.min(distances_deg):.2f}°")
    print(f"Max distance: {np.max(distances_deg):.2f}°")
    print(f"Standard deviation: {np.std(distances_deg):.2f}°")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(distances_deg, bins=30, edgecolor='black')
    plt.xlabel('Angular Distance to Closest Neighbor (degrees)')
    plt.ylabel('Count')
    plt.title('Distribution of Angular Distances to Closest Neighbors')
    plt.grid(True, alpha=0.3)
    plt.savefig('anchor_distances_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main() 