# import numpy as np

# def emeans(data: np.ndarray, centroids: np.ndarray, k: int, batch_size: int):
#     # At each epoch, shuffle the data and divide it into batches.
#     rng = np.random.default_rng()
#     shuffled_data = rng.shuffle(data)
#     batches = [shuffled_data[i:i+batch_size] for i in range(0, len(shuffled_data), batch_size)]

#     for batch in batches:
#         assignments = np.zeros(len(batch), dtype=int)
#         for point in batch:
#             # For every point in the current batch, choose the closest current center:
#             closest_idx = np.argmin(np.linalg.norm(point - centroids, axis=1))
#             # each centroid has its own index, so we can use it to update the centroid.
#             assignments[closest_idx] += 1

import math
import random
import matplotlib.pyplot as plt

# start from 1D case, then extend to n-dimensional case.

class Cluster1D:
    def __init__(self, n_j, c_j):
        self.nj = n_j
        self.cj = c_j
        self.sj = n_j * c_j


def emeans_1d(data: list[float], k: int, batch_size: int, num_epochs: int, alpha: float, revival_frequency: int=5, revival_threshold: float=0.1, perturbation_size: float=2**-7) -> list[Cluster1D]:
    # initialize centers. sample uniformly from the data.
    sampled_centers = random.sample(data, k)
    clusters = [Cluster1D(1e-10, center) for center in sampled_centers]
    t = 0
    M = num_epochs * math.ceil(len(data) / batch_size)

    for epoch in range(num_epochs):
        # At each epoch, shuffle the data and divide it into batches.
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        batched_data = [shuffled_data[i:i + batch_size]
                        for i in range(0, len(shuffled_data), batch_size)]

        for batch in batched_data:
            cluster_dict = {c: [] for c in clusters}
            for point in batch:
                closest_cluster = min(clusters, key=lambda c: abs(point - c.cj))
                cluster_dict[closest_cluster].append(point)

            # cosine scheduler
            alpha_t = alpha * (0.1 + 0.9 * (1 + math.cos(t * math.pi / M)) / 2)

            for cluster, points in cluster_dict.items():
                N_j = (1 - alpha_t) * cluster.nj + alpha_t * len(points)
                S_j = (1 - alpha_t) * cluster.sj + alpha_t * sum(points)

                # If alpha_t is 1 and the cluster is empty, retain its location.
                if N_j > 0:
                    cluster.cj = S_j / N_j
                cluster.nj = N_j
                cluster.sj = S_j

            t += 1
            if t % revival_frequency == 0:
                # i would like to mark all the clusters that have no points for revival.
                # but this runs into the edge case where there are more clusters needed for revival than healthy clusters
                # and then we get pidgenholed and we can't revive any clusters.
                # so they need to be updated sequentially
                for cluster in clusters:
                    average_running_count = sum(c.nj for c in clusters) / k

                    if cluster.nj < revival_threshold * average_running_count:
                        donors = [c for c in clusters
                                  if c is not cluster
                                  and c.nj >= revival_threshold * average_running_count
                                  and c.nj > 0]
                        donor_cluster = random.choices(
                            donors, weights=[c.nj for c in donors], k=1
                        )[0]
                        donor_cj = donor_cluster.cj
                        donor_nj = donor_cluster.nj

                        cluster.cj = donor_cj * (1 + perturbation_size)
                        cluster.nj = donor_nj / 2
                        cluster.sj = cluster.cj * cluster.nj

                        donor_cluster.cj = donor_cj * (1 - perturbation_size)
                        donor_cluster.nj = donor_nj / 2
                        donor_cluster.sj = donor_cluster.cj * donor_cluster.nj

    return clusters

def main():
    random.seed(67) # haha six seven

    # three groups near -6, 0, and 6
    data = [random.gauss(center, 0.5) for center in [-6, 0, 6] for _ in range(40)]

    clusters = emeans_1d(data, k=3, batch_size=20, num_epochs=10, alpha=0.3)
    centers = sorted(c.cj for c in clusters)
    print("e-means centers:", [round(c, 2) for c in centers])

    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i, center in enumerate(centers):
        points = [x for x in data
                if min(range(3), key=lambda j: abs(x - centers[j])) == i]
        ax.scatter(points, [0] * len(points), color=colors[i], alpha=0.5, s=35)
        ax.scatter(center, 0, color="black", marker="x", s=180, linewidths=3,
                label="Learned center" if i == 0 else None, zorder=3)
        ax.annotate(f"{center:.2f}", (center, 0), xytext=(0, 18),
                    textcoords="offset points", ha="center", fontsize=12)

    ax.set(title="1D e-means with three groups", xlabel="Value")
    ax.set_yticks([])
    ax.set_ylim(-0.15, 0.25)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()