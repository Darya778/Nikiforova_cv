import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.optimize import linear_sum_assignment

def track_objects_optimized(centroids_list):
    num_objects = len(centroids_list[0])
    trajectories = [[] for _ in range(num_objects)]

    for i, centroids in enumerate(centroids_list):
        if i == 0:
            for j, (centroid, _) in enumerate(centroids):
                trajectories[j].append(centroid)
        else:
            previous_centroids = np.array([traj[-1] for traj in trajectories])
            current_centroids = np.array([c[0] for c in centroids])
            cost_matrix = np.sum((previous_centroids[:, np.newaxis, :] - current_centroids[np.newaxis, :, :])*2, axis=2)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for j, k in zip(row_ind, col_ind):
                trajectories[j].append(current_centroids[k])

    return trajectories


centroids_list = []
for i in range(100):
    data = np.load(f'out/h_{i}.npy')
    labeled = label(data)
    regions = regionprops(labeled)
    current_centroid = []
    for region in regions:
        current_centroid.append([region.centroid, region.label])
    centroids_list.append(current_centroid)


trajectories = track_objects_optimized(centroids_list)
for trajectory in trajectories:
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 1], trajectory[:, 0], marker='o')
plt.show()

