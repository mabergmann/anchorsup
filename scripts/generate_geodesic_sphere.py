import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import hydra
from omegaconf import DictConfig
from upright_anchor.anchor_generation import create_anchor_generator
from mpl_toolkits.mplot3d import art3d


def plot_straight_line(ax, v1, v2, color='blue', linewidth=5):
    """Plot a straight line between two points."""
    ax.plot([v1[0]+.1, v2[0]+.1], [v1[1], v2[1]], [v1[2], v2[2]], 
            color=color, linewidth=linewidth)


def find_closest_anchor(point, anchors):
    """Find the closest anchor point to the given point."""
    distances = np.linalg.norm(anchors - point, axis=1)
    closest_idx = np.argmin(distances)
    return anchors[closest_idx]


def plot_geodesic_arc(ax, p1, p2, n_points=50, color='blue', linewidth=2):
    """Plot a geodesic arc between two points on a unit sphere."""
    # Generate points along the great circle
    t = np.linspace(0, 1, n_points)
    # Use slerp (spherical linear interpolation)
    sin_omega = np.linalg.norm(np.cross(p1, p2))
    cos_omega = np.dot(p1, p2)
    omega = np.arctan2(sin_omega, cos_omega)
    
    points = np.zeros((n_points, 3))
    for i, ti in enumerate(t):
        points[i] = (np.sin((1-ti)*omega)*p1 + np.sin(ti*omega)*p2) / np.sin(omega)
    
    # Scale points slightly above surface for visibility
    # Move points slightly toward viewpoint for visibility
    viewpoint = np.array([1, 0, 0])  # Assuming default matplotlib 3D viewpoint
    view_direction = viewpoint / np.linalg.norm(viewpoint)
    points = points + 0.3 * view_direction
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=linewidth)


def plot_faces(ax, vertices, faces, color='black'):
    """Plot the faces of the icosahedron."""
    for face in faces:
        # Draw the triangular face itself
        triangle = art3d.Poly3DCollection([
            [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
        ])
        triangle.set_alpha(1)
        triangle.set_facecolor("white")
        triangle.set_edgecolor("black")
        triangle.set_linewidth(3)
        ax.add_collection3d(triangle)


def plot_point(ax, point, color='red', size=100):
    """Plot a single point."""
    # Plot point slightly above surface for visibility
    point_scaled = point * 1.02
    ax.scatter(point_scaled[0], point_scaled[1], point_scaled[2], color=color, s=size)


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Create figure with square aspect ratio
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    
    # Generate vertices using anchor generator
    anchor_generator = create_anchor_generator(cfg)
    
    # Get vertices and faces (only works with IcosahedronTessellation)
    vertices = anchor_generator.generate().numpy()
    if hasattr(anchor_generator, '_IcosahedronTessellation__faces'):
        faces = anchor_generator._IcosahedronTessellation__faces
        # Plot the outer faces
        plot_faces(ax, vertices, faces)
    else:
        # For other generators, just plot the points
        for vertex in vertices:
            plot_point(ax, vertex, color='black', size=50)
    
    # Create and plot a random unit vector
    random_vector = np.array([1.33, 0.3, .8])
    unit_vector = random_vector / np.linalg.norm(random_vector)
    plot_point(ax, unit_vector*0.1)
    
    # Find and plot closest anchor
    closest_anchor = find_closest_anchor(unit_vector, vertices)
    print(closest_anchor)
    print(unit_vector)
    
    # Set equal aspect ratio and viewing angle
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=0, azim=0)  # Set optimal viewing angle
    
    # Turn off axis completely and make background transparent
    ax.axis('off')
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    
    # Remove grid and background for cleaner look
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Adjust layout to ensure no cropping
    plt.tight_layout()
    
    # Save the plot with white background
    plt.savefig('geodesic_sphere.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    main() 