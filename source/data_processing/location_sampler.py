import numpy as np


class LocationSampler:

    def __init__(self, room_X, room_Y):
        self.room_shape = (room_X, room_Y)

    def sample_uniform(self, n):
        x_uniform = np.random.uniform(0, self.room_shape[0], n)
        y_uniform = np.random.uniform(0, self.room_shape[1], n)

        return np.concatenate([x_uniform[:, None], y_uniform[:, None]], axis=1)

    def sample_gaussians(self, centroids, std, n):
        num_clusters = len(centroids)
        points_per_cluster = n // num_clusters

        x_gaussian, y_gaussian = [], []
        for cx, cy in centroids:
            x_gaussian.extend(np.random.normal(cx, std, points_per_cluster))
            y_gaussian.extend(np.random.normal(cy, std, points_per_cluster))

        Y = np.column_stack([x_gaussian, y_gaussian])

        # clip values to ensure all locations are inside the room
        Y[:, 0] = np.clip(Y[:, 0], 0, self.room_shape[0])
        Y[:, 1] = np.clip(Y[:, 1], 0, self.room_shape[1])

        return Y