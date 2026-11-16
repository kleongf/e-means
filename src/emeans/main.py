import math
import random
import matplotlib.pyplot as plt
import numpy as np

from sentence_transformers import SentenceTransformer
from datasets import load_dataset, concatenate_datasets
from sentence_transformers.base.sampler import _sample_value_str
import torch

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

# n-dimensional case
def emeans(data: np.ndarray, k: int, batch_size: int, num_epochs: int, alpha: float, revival_frequency: int = 5, revival_threshold: float = 0.1, perturbation_size: float = 2**-7) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)

    rng = np.random.default_rng(seed=420)
    indices = rng.choice(len(data), size=k, replace=False)
    C = np.array(data[indices], dtype=np.float64, copy=True)  # centers: (k, d)
    N = np.full(k, 1e-10)  # counts: (k,)
    S = np.expand_dims(N, axis=1) * C  # products: (k, d)
    M = num_epochs * math.ceil(len(data) / batch_size)
    t = 0

    # Alternating relative offsets, one for each coordinate.
    perturbation = np.full(C.shape[1], perturbation_size)
    perturbation[1::2] *= -1

    for epoch in range(num_epochs):
        # at each epoch, shuffle the data and divide it into batches.
        indices = rng.permutation(len(data))
        batched_data = [indices[i:i + batch_size]
                        for i in range(0, len(data), batch_size)]

        for batch in batched_data:  # x: (batch_size, d)
            # batch contains indices; X can have fewer rows in the last batch.
            X = data[batch]
            # for each x, ||x||^2 is constant across centers. x^2 is also convex so it works out
            scores = (C * C).sum(axis=1)[None, :] - 2 * (X @ C.T)
            center_labels = scores.argmin(axis=1)

            batch_counts = np.bincount(center_labels, minlength=k)
            batch_sums = np.zeros_like(S)
            np.add.at(batch_sums, center_labels, X)

            # cosine scheduler
            alpha_t = alpha * (0.1 + 0.9 * (1 + np.cos(t * math.pi / M)) / 2)

            N = (1 - alpha_t) * N + alpha_t * batch_counts
            S = (1 - alpha_t) * S + alpha_t * batch_sums

            np.divide(S, N[:, None], out=C, where=N[:, None] > 0)  # basically c_j = s_j / n_j

            t += 1

            if t % revival_frequency == 0:
                for center_idx in range(k):
                    average_running_count = np.mean(N)
                    if N[center_idx] < revival_threshold * average_running_count:
                        # filter mask, equivalent to the condition in the 1D case
                        donors = np.flatnonzero(
                            (np.arange(k) != center_idx)
                            & (N >= revival_threshold * average_running_count)
                            & (N > 0)
                        )
                        weights = N[donors]
                        donor_cluster = rng.choice(donors, p=weights / weights.sum())

                        donor_cj = C[donor_cluster].copy()
                        donor_nj = N[donor_cluster]

                        C[center_idx] = donor_cj * (1 + perturbation)
                        N[center_idx] = donor_nj / 2
                        S[center_idx] = C[center_idx] * N[center_idx]

                        C[donor_cluster] = donor_cj * (1 - perturbation)
                        N[donor_cluster] = donor_nj / 2
                        S[donor_cluster] = C[donor_cluster] * N[donor_cluster]

    return C

def pca_eig(X, n_components):
    # center the data (subtract column means)
    X_centered = X - np.mean(X, axis=0)
    
    # compute the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # compute eigenvalues and eigenvectors (returned in ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # sort eigenvalues and eigenvectors in descending order
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    # select top k eigenvectors and project data
    components = eigenvectors[:, :n_components]
    X_reduced = np.dot(X_centered, components)
    
    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_reduced, explained_variance_ratio

def main():
    # random.seed(67) # haha six seven

    # # three groups near -6, 0, and 6
    # data = [random.gauss(center, 0.5) for center in [-6, 0, 6] for _ in range(40)]

    # clusters = emeans_1d(data, k=3, batch_size=20, num_epochs=10, alpha=0.3)
    # centers = sorted(c.cj for c in clusters)
    # print("e-means centers:", [round(c, 2) for c in centers])

    # fig, ax = plt.subplots(figsize=(10, 3))
    # colors = ["tab:blue", "tab:orange", "tab:green"]
    # for i, center in enumerate(centers):
    #     points = [x for x in data
    #             if min(range(3), key=lambda j: abs(x - centers[j])) == i]
    #     ax.scatter(points, [0] * len(points), color=colors[i], alpha=0.5, s=35)
    #     ax.scatter(center, 0, color="black", marker="x", s=180, linewidths=3,
    #             label="Learned center" if i == 0 else None, zorder=3)
    #     ax.annotate(f"{center:.2f}", (center, 0), xytext=(0, 18),
    #                 textcoords="offset points", ha="center", fontsize=12)

    # ax.set(title="1D e-means with three groups", xlabel="Value")
    # ax.set_yticks([])
    # ax.set_ylim(-0.15, 0.25)
    # ax.spines[["left", "right", "top"]].set_visible(False)
    # ax.legend(loc="upper right")
    # fig.tight_layout()
    # plt.show()

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    samples_per_class = 200
    dataset = load_dataset("fancyzhx/ag_news", split="train")
    unique_labels = dataset.unique("label")
    print(unique_labels)

    samples = [
        dataset.filter(lambda x: x["label"] == label)
            .shuffle(seed=42)
            .select(range(samples_per_class))
        for label in unique_labels
    ]
    all_samples = concatenate_datasets(samples)

    print(all_samples)

    embeddings = model.encode(
        all_samples["text"],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(embeddings.shape)
    np.save("embeddings.npy", embeddings)

    np.save("labels.npy", np.asarray(all_samples["label"]))
    all_samples.save_to_disk("embedding_samples")

    clusters = emeans(embeddings, k=len(unique_labels), batch_size=10, num_epochs=10, alpha=0.5)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # plot the clusters
    pca_clusters, explained_variance_ratio = pca_eig(clusters, 3)
    print("explained variance ratio:", explained_variance_ratio)
    # plot the pca embeddings
    ax.scatter(pca_clusters[:, 0], pca_clusters[:, 1], pca_clusters[:, 2], marker='o', s=16, color='red')

    pca_embeddings, _ = pca_eig(embeddings, 3)
    ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], pca_embeddings[:, 2], marker='o', s=4, color='blue')

    plt.show()
    
if __name__ == "__main__":
    main()