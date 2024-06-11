import numpy as np
import matplotlib.pyplot as plt

from .render import render_rays
from .nerf import NeRF

# Normalize function
def normalize(arr):
    min_val = arr.min()
    max_val = arr.max()
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def viz_dataset(training_dataset):
    # Split the dataset into three sets of images; visualize the first 3 images

    # print the max and min of the training set 
    ray_origins = training_dataset[:, :3]
    ray_directions = training_dataset[:, 3:6]
    ground_truth_px_values = training_dataset[:, 6:]

    nerf_model = NeRF()
    regenerated_px_values = render_rays(nerf_model, ray_origins[:5, :], ray_directions[:5, :])
    print(regenerated_px_values)
    print(f'Min and Max of Ray Origins: {ray_origins.min()}, {ray_origins.max()}')
    print(f'Min and Max of Ray Directions: {ray_directions.min()}, {ray_directions.max()}')
    print(f'Min and Max of Ground Truth PX Values: {ground_truth_px_values.min()}, {ground_truth_px_values.max()}')
    print(f'Min and Max of Regenerated PX Values: {regenerated_px_values.min()}, {regenerated_px_values.max()}')

    images_per_set = 4
    H, W = 400, 400

    # Plot the images
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for set_index in range(3):
        start_idx = set_index * images_per_set * H * W
        end_idx = (set_index + 1) * images_per_set * H * W
        batch = training_dataset[start_idx:end_idx].reshape(images_per_set, H, W, 9)
        
        # Normalize and split the data
        ray_origins = normalize(batch[0, :, :, :3])
        ray_directions = normalize(batch[0, :, :, 3:6])
        ground_truth_px_values = normalize(batch[0, :, :, 6:])

        # Plot the images
        axs[set_index, 0].imshow(ray_origins)
        axs[set_index, 0].set_title(f'Set {set_index + 1} - Ray Origins')
        axs[set_index, 1].imshow(ray_directions)
        axs[set_index, 1].set_title(f'Set {set_index + 1} - Ray Directions')
        axs[set_index, 2].imshow(ground_truth_px_values)
        axs[set_index, 2].set_title(f'Set {set_index + 1} - Ground Truth PX Values')

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# def get_camera_ray_origins_and_directions(phi, R, H=400, W=400, fov=60):
#     # Convert degrees to radians
#     phi_rad = np.deg2rad(phi)
    
#     # Calculate the camera position on the circle
#     camera_pos = np.array([R * np.cos(phi_rad), R * np.sin(phi_rad), 0])
    
#     # Calculate the direction vectors from the camera to each pixel
#     # Initialize ray origins and directions
#     ray_origins = np.tile(camera_pos, (H, W, 1))  # Constant across all pixels
#     ray_directions = np.zeros((H, W, 3))
    
#     for i in range(H):
#         for j in range(W):
#             # Calculate normalized device coordinates (NDC)
#             ndc_x = (2 * (j + 0.5) / W - 1) * np.tan(np.deg2rad(fov / 2)) * W / H
#             ndc_y = (1 - 2 * (i + 0.5) / H) * np.tan(np.deg2rad(fov / 2))
            
#             # Calculate ray direction in camera space
#             ray_dir_cam = np.array([ndc_x, ndc_y, -1])
            
#             # Normalize ray direction
#             ray_dir_cam /= np.linalg.norm(ray_dir_cam)
            
#             # Assign ray direction
#             ray_directions[i, j] = ray_dir_cam
    
#     return ray_origins, ray_directions

# def viz_synth():
#     # Visualize the synthetic data
#     H, W = 400, 400
#     radius = 5
#     fov = 60
#     phi = 0

#     ray_origins, ray_directions = get_camera_ray_origins_and_directions(phi, radius, H, W, fov)
#     ray_origins = ray_origins.reshape(H, W, 3)
#     ray_directions = ray_directions.reshape(H, W, 3)

#     # Plot the images
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#     axs[0].imshow(ray_origins)
#     axs[0].set_title('Ray Origins')
#     axs[1].imshow(ray_directions)
#     axs[1].set_title('Ray Directions')

#     for ax in axs.flat:
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()

