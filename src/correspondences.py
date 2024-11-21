import argparse
import torch
import numpy as np
from external.kmeans_pytorch.kmeans_pytorch import kmeans
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple
from torch.nn.functional import cosine_similarity
import faiss
import time
import faiss.contrib.torch_utils


def find_correspondences(
        descriptors1: torch.Tensor,
        descriptors2: torch.Tensor,
        num_patches: Tuple[int, int] = (16, 16),
        num_pairs: int = 10,
        use_kmeans: bool = True,
        patch_size: int = 14,
        use_best_buddies: bool = True
) -> Tuple[
    List[Tuple[float, float]],
    List[Tuple[float, float]],
]:
    """
    Find correspondences between two images using descriptors, saliency maps, and k-means clustering.

    Parameters:
        num_patches (list): The size of token in feature map
        num_pairs (int): Number of correspondence pairs to return.
        load_size (int): Size to which images are resized during preprocessing.
        layer (int): Layer from which descriptors are extracted.
        use_kmeans (bool): Whether to apply k-means clustering.
        use_best_buddies (bool): Whether to compute mutual best buddies.

    Returns:
        Tuple containing:
            - List of (y, x) coordinates from the first image.
            - List of (y, x) coordinates from the second image.
            - Preprocessed first image (PIL.Image).
            - Preprocessed second image (PIL.Image).
    """
    # Initialize timing
    start_time_corr = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Step 1: given descriptors
    descriptors1 = descriptors1
    num_patches1 = num_patches

    descriptors2 = descriptors2
    num_patches2 = num_patches

    # Step 2: Extract and threshold saliency maps
    fg_mask1, fg_mask2 = torch.ones(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=device), \
        torch.ones(num_patches2[0] * num_patches2[1], dtype=torch.bool, device=device)

    # Step 3: Compute cosine similarities between descriptors
    similarities = cosine_sim(descriptors1, descriptors2)

    # Step 4: Determine best buddies (mutual nearest neighbors)
    if use_best_buddies:
        sim_1, nn_1 = torch.max(similarities, dim=-1)
        sim_2, nn_2 = torch.max(similarities, dim=-2)

        # Assuming batch size of 1 and single channel
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=device)
        bbs_mask = nn_2[nn_1] == image_idxs

        # Apply foreground masks
        fg_mask2_new_coors = nn_2[fg_mask2]
        fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool,
                                              device=device)
        fg_mask2_mask_new_coors[fg_mask2_new_coors] = True

        bbs_mask &= fg_mask1 & fg_mask2_mask_new_coors
    else:
        bbs_mask = torch.ones(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)

    # Step 5: Extract descriptors of best buddies
    bb_descs1 = descriptors1[0, 0, bbs_mask, :]
    bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :]

    # Step 6: Apply K-Means clustering to find well-distributed correspondences
    # if use_kmeans and len(bb_descs1) >= num_pairs: # important
    if use_kmeans:
        all_keys_together = torch.cat((bb_descs1, bb_descs2), dim=1)
        n_clusters = min(num_pairs, len(all_keys_together))
        normalized = all_keys_together / torch.norm(all_keys_together, p=2, dim=1, keepdim=True)

        cluster_ids_x, _ = kmeans(
            X=normalized,
            num_clusters=n_clusters,
            distance='cosine',
            tqdm_flag=False,
            iter_limit=200,
            seed=2024,
            device=device
        )

        kmeans_labels = cluster_ids_x.detach().cpu().numpy()
        bb_topk_sims = np.full(n_clusters, -np.inf)
        bb_indices_to_show = np.full(n_clusters, -1, dtype=int)

        bb_cls_attn = sim_1[bbs_mask].cpu().numpy()

        for idx, (label, rank) in enumerate(zip(kmeans_labels, bb_cls_attn)):
            if rank > bb_topk_sims[label]:
                bb_topk_sims[label] = rank
                bb_indices_to_show[label] = idx

        # Select valid indices
        valid_indices = bb_indices_to_show != -1
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(1)[bb_indices_to_show[valid_indices]]
    else:
        # Select top num_pairs based on similarity without K-Means
        sim_scores = sim_1[bbs_mask].cpu().numpy()
        top_indices = np.argsort(sim_scores)[-num_pairs:]
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(1)[top_indices]

    # Step 7: Convert selected indices to image coordinates
    img1_indices = torch.arange(num_patches1[0] * num_patches1[1], device=device)[indices_to_show]
    img2_indices = nn_1[indices_to_show]

    img1_y = (img1_indices // num_patches1[1]).cpu().numpy()
    img1_x = (img1_indices % num_patches1[1]).cpu().numpy()
    img2_y = (img2_indices // num_patches2[1]).cpu().numpy()
    img2_x = (img2_indices % num_patches2[1]).cpu().numpy()

    points1, points2 = [], []
    stride = [patch_size, patch_size]
    for y1, x1, y2, x2 in zip(img1_y, img1_x, img2_y, img2_x):
        x1_show = (int(x1) - 1) * stride[1] + stride[1] + patch_size // 2
        y1_show = (int(y1) - 1) * stride[0] + stride[0] + patch_size // 2
        x2_show = (int(x2) - 1) * stride[1] + stride[1] + patch_size // 2
        y2_show = (int(y2) - 1) * stride[0] + stride[0] + patch_size // 2
        points1.append((y1_show, x1_show))
        points2.append((y2_show, x2_show))

    end_time_corr = time.time()
    elapsed_corr = end_time_corr - start_time_corr

    # Optional: Print timing information for debugging
    # print(f"Total Corr Time: {elapsed_corr}, Descriptor Time: {elapsed_desc}, Cosine Similarity Time: {elapsed_time_chunk_cosine}, "
    #       f"Saliency Time: {elapsed_saliency if use_saliency else 'N/A'}, K-Means Time: {elapsed_kmeans if use_kmeans else 'N/A'}, "
    #       f"Best Buddies Time: {elapsed_bb if use_best_buddies else 'N/A'}")
    torch.cuda.empty_cache()
    return points1, points2


def draw_correspondences(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]],
                         image1: Image.Image, image2: Image.Image) -> Tuple[plt.Figure, plt.Figure]:
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :return: two figures of images with marked points.
    """
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)
    fig1, ax1 = plt.subplots()
    ax1.axis('off')
    fig2, ax2 = plt.subplots()
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)
    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                               "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 8, 1
    for point1, point2, color in zip(points1, points2, colors):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)
    return fig1, fig2


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks to save GPU RAM.

    :param x: a tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
              is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
              is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y)
    """
    # Normalize the vectors along the descriptor dimension
    x_norm = torch.nn.functional.normalize(x, dim=-1)  # Bx1x(t_x)xd'
    y_norm = torch.nn.functional.normalize(y, dim=-1)  # Bx1x(t_y)xd'

    # Compute the cosine similarity using batch matrix multiplication
    cosine_sim = torch.matmul(x_norm, y_norm.transpose(-1, -2))  # Bx1x(t_x)x(t_y)

    return cosine_sim


def pairwise_sim(x: torch.Tensor, y: torch.Tensor, p=2, normalize=False) -> torch.Tensor:
    # Normalize x and y if specified
    if normalize:
        x = torch.nn.functional.normalize(x, dim=-1)
        y = torch.nn.functional.normalize(y, dim=-1)

    # Compute pairwise distances in a vectorized manner
    # x shape: [B, C, T_x, D], y shape: [B, C, T_y, D]
    # Add a dummy dimension to broadcast x and y for pairwise distance calculation
    x_expanded = x.unsqueeze(3)  # Shape: [B, C, T_x, 1, D]
    y_expanded = y.unsqueeze(2)  # Shape: [B, C, 1, T_y, D]

    # Compute pairwise distances between all tokens
    pairwise_distances = torch.cdist(x_expanded, y_expanded, p=p)  # Shape: [B, C, T_x, 1, T_y]

    # Remove the extra singleton dimension
    pairwise_distances = pairwise_distances.squeeze(3)  # Shape: [B, C, T_x, T_y]

    # Multiply by -1 to convert distance to similarity (optional)
    return pairwise_distances * (-1)


""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


@torch.jit.script
def calculate_ratio_test(dists: torch.Tensor):
    """
    Calculate weights for matches based on the ratio between kNN distances.

    Input:
        (N, P, 2) Cosine Distance between point and nearest 2 neighbors
    Output:
        (N, P, 1) Weight based on ratio; higher is more unique match
    """
    # Ratio -- close to 0 is completely unique; 1 is same feature
    # Weight -- Convert so that higher is more unique
    # clamping because some dists will be 0 (when not in the pointcloud
    dists = dists.clamp(min=1e-9)
    ratio = dists[..., 0] / dists[..., 1].clamp(min=1e-9)
    weight = 1 - ratio
    return weight


@torch.jit.script
def get_topk_matches(dists, idx, num_corres: int):
    num_corres = min(num_corres, dists.shape[-1])
    dist, idx_source = torch.topk(dists, k=num_corres, dim=-1)
    idx_target = idx[idx_source]
    return idx_source, idx_target, dist


res = faiss.StandardGpuResources()  # use a single GPU


def faiss_knn(query, target, k):
    # make sure query and target are contiguous
    query = query.contiguous()
    target = target.contiguous()

    num_elements, feat_dim = query.shape
    gpu_index = faiss.GpuIndexFlatL2(res, feat_dim)
    gpu_index.add(target)
    dist, index = gpu_index.search(query, k)
    return dist, index


def knn_points(X_f, Y_f, K=1, metric="euclidean"):
    """
    Finds the kNN according to either euclidean distance or cosine distance. This is
    tricky since PyTorch3D's fast kNN kernel does euclidean distance, however, we can
    take advantage of the relation between euclidean distance and cosine distance for
    points sampled on an n-dimension sphere.

    Using the quadratic expansion, we find that finding the kNN between two normalized
    is the same regardless of whether the metric is euclidean distance or cosine
    similiarity.

        -2 * xTy = (x - y)^2 - x^2 - y^2
        -2 * xtY = (x - y)^2 - 1 - 1
        - xTy = 0.5 * (x - y)^2 - 1

    Hence, the metric that would maximize cosine similarity is the same as that which
    would minimize the euclidean distance between the points, with the distances being
    a simple linear transformation.
    """
    assert metric in ["cosine", "euclidean"]
    if metric == "cosine":
        X_f = torch.nn.functional.normalize(X_f, dim=-1)
        Y_f = torch.nn.functional.normalize(Y_f, dim=-1)

    _, X_nn = faiss_knn(X_f, Y_f, K)

    # n_points x k x F
    X_f_nn = Y_f[X_nn]

    if metric == "euclidean":
        dists = (X_f_nn - X_f[:, None, :]).norm(p=2, dim=3)
    elif metric == "cosine":
        dists = 1 - cosine_similarity(X_f_nn, X_f[:, None, :], dim=-1)

    return dists, X_nn


def get_grid(H: int, W: int):
    # Generate a grid that's equally spaced based on image & embed size
    grid_x = torch.linspace(0.5, W - 0.5, W)
    grid_y = torch.linspace(0.5, H - 0.5, H)

    xs = grid_x.view(1, W).repeat(H, 1)
    ys = grid_y.view(H, 1).repeat(1, W)
    zs = torch.ones_like(xs)

    # Camera coordinate frame is +xyz (right, down, into-camera)
    # Dims: 3 x H x W
    grid_xyz = torch.stack((xs, ys, zs), dim=0)
    return grid_xyz


def _log_bin(x: torch.Tensor, hierarchy: int = 2, device: str = 'cpu') -> torch.Tensor:
    """
    create a log-binned descriptor.
    :param x: tensor of features. Has shape Bx1xtx(dxh).
    :param hierarchy: how many bin hierarchies to use.
    """
    B = x.shape[0]
    if x.shape[1] != 1:
        raise ValueError('log_bin function now expects features reshaped to Bx1xtx(dxh), not Bxhxtxd')
    num_patches = int(np.sqrt(x.shape[2]))
    num_bins = 1 + 8 * hierarchy

    # bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
    bin_x = x.squeeze(1)  # Bx(t-1)x(dxh)
    bin_x = bin_x.permute(0, 2, 1)  # Bx(dxh)x(t-1)
    bin_x = bin_x.reshape(B, bin_x.shape[1], num_patches, num_patches)
    # Bx(dxh)xnum_patches[0]xnum_patches[1]
    sub_desc_dim = bin_x.shape[1]

    avg_pools = []
    # compute bins of all sizes for all spatial locations.
    for k in range(0, hierarchy):
        # avg pooling with kernel 3**kx3**k
        win_size = 3 ** k
        avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
        avg_pools.append(avg_pool(bin_x))

    bin_x = torch.zeros((B, sub_desc_dim * num_bins, num_patches, num_patches)).to(device)
    for y in range(num_patches):
        for x in range(num_patches):
            part_idx = 0
            # fill all bins for a spatial location (y, x)
            for k in range(0, hierarchy):
                kernel_size = 3 ** k
                for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                    for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                        if i == y and j == x and k != 0:
                            continue
                        if 0 <= i < num_patches and 0 <= j < num_patches:
                            bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                     :, :, i, j]
                        else:  # handle padding in a more delicate way than zero padding
                            temp_i = max(0, min(i, num_patches - 1))
                            temp_j = max(0, min(j, num_patches - 1))
                            bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                     :, :, temp_i,
                                                                                                     temp_j]
                        part_idx += 1
    bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
    # Bx1x(t-1)x(dxh)
    return bin_x
