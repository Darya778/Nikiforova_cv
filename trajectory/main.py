import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist


def track_objects_optimized(positions_per_frame):
    paths = {}
    for frame_num, positions in enumerate(positions_per_frame):
        if frame_num == 0:
            for obj_id, (position, _label) in enumerate(positions):
                paths[obj_id] = [position]
        else:
            previous_positions = []
            for path in paths.values():
                previous_positions.append(path[-1])
            previous_positions = np.array(previous_positions)
            current_positions = []
            for p in positions:
                current_positions.append(p[0])
            current_positions_array = np.array(current_positions)
            distance_matrix = cdist(previous_positions, current_positions_array)
            for obj_id, path in paths.items():
                min_dist_idx = np.argmin(distance_matrix[obj_id])
                path.append(current_positions_array[min_dist_idx])
    return paths


positions_per_frame = []
for i in range(100):
    motion_data = np.load(f'motion/out/h_{i}.npy')
    labeled_image = label(motion_data)
    regions = regionprops(labeled_image)
    frame_positions = []
    for region in regions:
        frame_positions.append([region.centroid, region.label])
    positions_per_frame.append(frame_positions)


object_paths = track_objects_optimized(positions_per_frame)
for obj_id, path in object_paths.items():
    path = np.array(path)
    plt.plot(path[:, 1], path[:, 0], marker='o')
plt.show()
