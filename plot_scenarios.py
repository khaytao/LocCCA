import numpy as np
import matplotlib.pyplot as plt
from source.data_processing.location_sampler import LocationSampler

# Set random seed for reproducibility
np.random.seed(42)

# Generate uniformly distributed points (Scenario 1)
num_points = 1000
room_size = (6, 6)  # Width x Height

sampler = LocationSampler(room_size[0], room_size[1])

Y_uniform = sampler.sample_uniform(num_points)
# x_uniform = np.random.uniform(0, room_size[0], num_points)
# y_uniform = np.random.uniform(0, room_size[1], num_points)

# Generate Gaussian-distributed points around fixed centroids (Scenario 2)
centroids = np.array([(2, 2), (4, 2), (6, 2), (2, 4), (6, 4), (2, 6), (6, 6), (2, 8), (4, 8), (6, 8)])  #todo update them to a 6x6 room
num_clusters = len(centroids)
points_per_cluster = num_points // num_clusters
std_dev = 0.35  # Spread of the clusters

Y_gaussian = sampler.sample_gaussians(centroids, std_dev, num_points)
# x_gaussian, y_gaussian = [], []
# for cx, cy in centroids:
#     x_gaussian.extend(np.random.normal(cx, std_dev, points_per_cluster))
#     y_gaussian.extend(np.random.normal(cy, std_dev, points_per_cluster))

# Plot both scenarios
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Scenario 1: Uniform Distribution
axes[0].scatter(Y_uniform[:, 0], Y_uniform[:, 1], alpha=0.5, s=10)
axes[0].set_title("Scenario 1: Uniform Speaker Distribution")
axes[0].set_xlim(0, room_size[0])
axes[0].set_ylim(0, room_size[1])
axes[0].set_xlabel("Room X axis [meters]")
axes[0].set_ylabel("Room Y axis [meters]")

# Scenario 2: Gaussian Clusters
axes[1].scatter(Y_gaussian[:, 0], Y_gaussian[:, 1], alpha=0.5, s=10, label="Speaker Positions")
axes[1].scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label="Cluster Centers")
axes[1].set_title("Scenario 2: MoG Speaker Distribution")
axes[1].set_xlim(0, room_size[0])
axes[1].set_ylim(0, room_size[1])
axes[1].set_xlabel("Room X axis [meters]")
axes[1].set_ylabel("Room Y axis [meters]")
axes[1].legend()

plt.tight_layout()
plt.show()
