import torch
from torch import nn
from torchvision import transforms
import src.extractor as extractor
from PIL import Image
from typing import Union, List, Tuple
from src.correspondences import chunk_cosine_sim, cosine_sim, pairwise_sim
from sklearn.cluster import KMeans
import numpy as np
import time
from external.kmeans_pytorch.kmeans_pytorch import kmeans
from src.correspondences import calculate_ratio_test, knn_points, get_topk_matches, get_grid
from PIL import Image, ImageOps


class PoseViTExtractor(extractor.ViTExtractor):

    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4, model: nn.Module = None, device: str = 'cuda'):
        self.model_type = model_type
        self.stride = stride
        self.model = model
        self.device = device
        super().__init__(model_type=self.model_type, stride=self.stride, model=self.model, device=self.device)

        self.prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        self.prep_mask = transforms.Compose([
            transforms.ToTensor(),
        ])

    def center_padding(self, image, patch_size):
        w, h = image.size  # Get the width and height of the PIL image
        diff_h = h % patch_size
        diff_w = w % patch_size

        # If the image dimensions are already divisible by patch_size, no padding is needed
        if diff_h == 0 and diff_w == 0:
            return image

        # Calculate padding needed to make the height and width divisible by patch_size
        pad_h = patch_size - diff_h if diff_h != 0 else 0
        pad_w = patch_size - diff_w if diff_w != 0 else 0

        # Divide the padding into top/bottom and left/right
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Pad the image using ImageOps.expand
        padded_image = ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)

        return padded_image

    def preprocess(self, img: Image.Image,
                   load_size: Union[int, Tuple[int, int]] = None) -> Tuple[torch.Tensor, Image.Image, int]:
        scale_factor = 1
        if load_size is not None:
            width, height = img.size  # img has to be quadratic
            img = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(img)
            scale_factor = img.size[0] / width

        prep_img = self.prep(img)[None, ...]

        return prep_img, img, scale_factor

    def preprocess_mask(self, img: Image.Image,
                        load_size: Union[int, Tuple[int, int]] = None) -> Tuple[torch.Tensor, Image.Image, int]:

        scale_factor = 1
        if load_size is not None:
            width, height = img.size  # img has to be quadratic
            img = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(img)
            scale_factor = img.size[0] / width

        prep_img = self.prep_mask(img)

        return prep_img, img, scale_factor

    # Overwrite functionality of _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]
    # to extract multiple facets and layers in one turn

    def _extract_multi_features(self, batch: torch.Tensor, layers: List[int] = [9, 11], facet: str = 'key') -> List[
        torch.Tensor]:
        B, C, H, W = batch.shape
        self._feats = []
        # for (layer,fac) in zip(layers,facet):
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def extract_multi_descriptors(self, batch: torch.Tensor, layers: List[int] = [9, 11], facet: str = 'key',
                                  bin: List[bool] = [True, False],
                                  include_cls: List[bool] = [False, False]) -> torch.Tensor:

        assert facet in ['key', 'query', 'value', 'token'], f"""{facet} is not a supported facet for descriptors. 
                                                        choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_multi_features(batch, layers, facet)
        descs = []
        for i, x in enumerate(self._feats):
            if facet[i] == 'token':
                x.unsqueeze_(dim=1)  # Bx1xtxd
            if not include_cls[i]:
                x = x[:, :, 1:, :]  # remove cls token
            else:
                assert not bin[
                    i], "bin = True and include_cls = True are not supported together, set one of them False."
            if not bin:
                desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
            else:
                desc = self._log_bin(x)
            descs.append(desc)
        return descs

    def find_correspondences_weight(
            self,
            pil_img1: Image.Image,
            pil_img2: Image.Image,
            num_pairs: int = 10,
            load_size: Union[int, Tuple[int, int]] = 224,
            layer: int = 9,
            facet: str = 'key',
            bin: bool = True,
            thresh: float = 0.05,
            use_kmeans: bool = False,
            use_saliency: bool = False,
            ratio_test: bool = True,
            bidirectional: bool = True) -> Tuple[
        List[Tuple[float, float]],
        List[Tuple[float, float]],
        Image.Image,
        Image.Image
    ]:
        """
        Find correspondences between two images using descriptors, optional saliency maps,
        ratio test, bidirectional matching, and optional k-means clustering.

        Parameters:
            pil_img1 (Image.Image): First input image.
            pil_img2 (Image.Image): Second input image.
            num_pairs (int): Number of correspondence pairs to return.
            load_size (int or Tuple[int, int]): Size to which images are resized during preprocessing.
            layer (int): Layer from which descriptors are extracted.
            facet (str): Facet or aspect of descriptors to use.
            bin (bool): Whether to apply binning to descriptors.
            thresh (float): Threshold for saliency maps to distinguish foreground from background.
            use_kmeans (bool): Whether to apply k-means clustering.
            use_saliency (bool): Whether to use saliency maps.
            ratio_test (bool): Whether to apply the ratio test.
            bidirectional (bool): Whether to perform bidirectional matching.

        Returns:
            Tuple containing:
                - List of (y, x) coordinates from the first image.
                - List of (y, x) coordinates from the second image.
                - Preprocessed first image (PIL.Image).
                - Preprocessed second image (PIL.Image).
        """
        start_time_corr = time.time()

        # Step 1: Preprocess images and extract descriptors
        image1_batch, image1_pil, _ = self.preprocess(pil_img1, load_size)
        descriptors1 = self.extract_descriptors(image1_batch.to(self.device), layer, facet, bin)
        num_patches1 = self.num_patches

        image2_batch, image2_pil, _ = self.preprocess(pil_img2, load_size)
        descriptors2 = self.extract_descriptors(image2_batch.to(self.device), layer, facet, bin)
        num_patches2 = self.num_patches

        # Step 2: Optional saliency map extraction and masking
        if use_saliency:
            saliency_map1, saliency_map2 = (
                self.extract_saliency_maps(image1_batch.to(self.device))[0],
                self.extract_saliency_maps(image2_batch.to(self.device))[0]
            )
            fg_mask1, fg_mask2 = saliency_map1 > thresh, saliency_map2 > thresh
            fg_mask1_flat, fg_mask2_flat = fg_mask1.view(-1), fg_mask2.view(-1)
        else:
            fg_mask1_flat = fg_mask2_flat = None

        # Step 3: Compute k-Nearest Neighbors (k=2 for ratio test)
        K = 2
        dists_1, idx_1 = knn_points(descriptors1.squeeze(), descriptors2.squeeze(), K, metric="cosine")
        idx_1 = idx_1[..., 0]  # Select the closest neighbor

        # Step 4: Apply ratio test or use similarity scores directly
        if ratio_test:
            weights_1 = calculate_ratio_test(dists_1)
        else:
            weights_1 = 1 - dists_1[:, 0]  # Convert cosine distance to similarity

        # Step 5: Bidirectional matching
        if bidirectional:
            dists_2, idx_2 = knn_points(descriptors2.squeeze(), descriptors1.squeeze(), K, metric="cosine")
            idx_2 = idx_2[..., 0]  # Select the closest neighbor

            weights_2 = calculate_ratio_test(dists_2) if ratio_test else 1 - dists_2[:, 0]

            # Get topK matches in both directions
            m12_idx1, m12_idx2, m12_dist = get_topk_matches(weights_1, idx_1, num_pairs // 2)
            m21_idx2, m21_idx1, m21_dist = get_topk_matches(weights_2, idx_2, num_pairs // 2)

            # Concatenate correspondences and weights
            all_idx1 = torch.cat((m12_idx1, m21_idx1), dim=-1)
            all_idx2 = torch.cat((m12_idx2, m21_idx2), dim=-1)
            all_dist = torch.cat((m12_dist, m21_dist), dim=-1)
        else:
            # Get topK matches in one direction
            all_idx1, all_idx2, all_dist = get_topk_matches(weights_1, idx_1, num_pairs)

        # Step 6: Apply saliency masks to filter out background matches
        if use_saliency:
            valid_corr_mask1 = fg_mask1_flat[all_idx1]
            valid_corr_mask2 = fg_mask2_flat[all_idx2]
            valid_corr_mask = valid_corr_mask1 & valid_corr_mask2  # Only keep foreground correspondences

            # Filter correspondences based on masks
            all_idx1 = all_idx1[valid_corr_mask]
            all_idx2 = all_idx2[valid_corr_mask]
            all_dist = all_dist[valid_corr_mask]

        # Step 7: Extract descriptors of matched pairs
        bb_descs1 = descriptors1[0, 0, all_idx1, :]  # Descriptors from image1
        bb_descs2 = descriptors2[0, 0, all_idx2, :]  # Descriptors from image2
        all_keys_together = torch.cat((bb_descs1, bb_descs2), dim=1)  # Concatenate descriptors

        # Step 8: Optional K-Means clustering to select well-distributed correspondences
        if use_kmeans and len(bb_descs1) >= num_pairs:
            n_clusters = min(num_pairs, len(all_keys_together))
            normalized = all_keys_together / torch.norm(all_keys_together, p=2, dim=1, keepdim=True)

            # Perform K-Means clustering
            cluster_ids_x, _ = kmeans(
                X=normalized,
                num_clusters=n_clusters,
                distance='cosine',
                tqdm_flag=False,
                iter_limit=200,
                seed=2024,
                device=self.device
            )
            kmeans_labels = cluster_ids_x.detach().cpu().numpy()

            # Initialize arrays to store top correspondences per cluster
            bb_topk_sims = np.full(n_clusters, -np.inf)
            bb_indices_to_show = np.full(n_clusters, -1, dtype=int)

            # Determine ranking scores based on ratio test or similarity
            ranks = all_dist.cpu().numpy() if not ratio_test else (1 - all_dist)

            # Assign the highest-ranked pair to each cluster
            for i, (label, rank) in enumerate(zip(kmeans_labels, ranks)):
                if rank > bb_topk_sims[label]:
                    bb_topk_sims[label] = rank
                    bb_indices_to_show[label] = i

            # Filter out invalid indices
            valid_indices = bb_indices_to_show != -1
            selected_indices = bb_indices_to_show[valid_indices].astype(int)

            # Retrieve selected correspondences
            indices_to_show = torch.tensor(selected_indices, device=self.device)
            indices_to_show = all_idx1[indices_to_show]
            corresponding_idx2 = all_idx2[indices_to_show]

        else:
            # Step 9: Select top num_pairs based on similarity without K-Means
            sim_scores = all_dist.cpu().numpy()
            top_indices = np.argsort(sim_scores)[-num_pairs:]  # Indices of top num_pairs based on similarity

            indices_to_show = all_idx1[top_indices]
            corresponding_idx2 = all_idx2[top_indices]

        # Step 10: Convert selected indices to image coordinates
        img1_indices_to_show = indices_to_show.cpu().numpy()
        img2_indices_to_show = corresponding_idx2.cpu().numpy()

        img1_y = img1_indices_to_show // num_patches1[1]
        img1_x = img1_indices_to_show % num_patches1[1]
        img2_y = img2_indices_to_show // num_patches2[1]
        img2_x = img2_indices_to_show % num_patches2[1]

        # Step 11: Map descriptor indices to actual image coordinates
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y, img1_x, img2_y, img2_x):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        # Step 12: Return the correspondence points and preprocessed images
        return points1, points2, image1_pil, image2_pil

    def find_correspondences_fastkmeans(self, pil_img1, pil_img2, num_pairs: int = 10,
                                        load_size: Union[int, Tuple[int, int]] = 224,
                                        layer: int = 9, facet: str = 'key', bin: bool = True,
                                        thresh: float = 0.05) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:

        start_time_corr = time.time()

        start_time_desc = time.time()
        image1_batch, image1_pil, scale_factor = self.preprocess(pil_img1, load_size)
        descriptors1 = self.extract_descriptors(image1_batch.to(self.device), layer, facet, bin)
        num_patches1, load_size1 = self.num_patches, self.load_size
        image2_batch, image2_pil, scale_factor = self.preprocess(pil_img2, load_size)
        descriptors2 = self.extract_descriptors(image2_batch.to(self.device), layer, facet, bin)
        num_patches2, load_size2 = self.num_patches, self.load_size
        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        start_time_saliency = time.time()
        # extracting saliency maps for each image
        saliency_map1 = self.extract_saliency_maps(image1_batch.to(self.device))[0]
        saliency_map2 = self.extract_saliency_maps(image2_batch.to(self.device))[0]
        end_time_saliency = time.time()
        elapsed_saliencey = end_time_saliency - start_time_saliency

        # saliency_map1 = self.extract_saliency_maps(image1_batch)[0]
        # saliency_map2 = self.extract_saliency_maps(image2_batch)[0]
        # threshold saliency maps to get fg / bg masks
        fg_mask1 = saliency_map1 > thresh
        fg_mask2 = saliency_map2 > thresh

        # calculate similarity between image1 and image2 descriptors
        start_time_chunk_cosine = time.time()
        similarities = cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine

        start_time_bb = time.time()
        # calculate best buddies
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

        # remove best buddies where at least one descriptor is marked bg by saliency mask.
        fg_mask2_new_coors = nn_2[fg_mask2]
        fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=self.device)
        fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

        # applying k-means to extract k high quality well distributed correspondence pairs
        # bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
        # bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
        bb_descs1 = descriptors1[0, 0, bbs_mask, :]
        bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :]
        # apply k-means on a concatenation of a pairs descriptors.
        # all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
        all_keys_together = torch.cat((bb_descs1, bb_descs2), axis=1)
        n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
        length = torch.sqrt((all_keys_together ** 2).sum(axis=1, keepdim=True))
        normalized = all_keys_together / length

        start_time_kmeans = time.time()
        # 'euclidean'
        # cluster_ids_x, cluster_centers = kmeans(X = normalized, num_clusters=n_clusters, distance='cosine', device=self.device)
        cluster_ids_x, cluster_centers = kmeans(X=normalized,
                                                num_clusters=n_clusters,
                                                distance='cosine',
                                                tqdm_flag=False,
                                                iter_limit=200,
                                                seed=2024,
                                                device=self.device)

        kmeans_labels = cluster_ids_x.detach().cpu().numpy()

        # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
        end_time_kmeans = time.time()
        elapsed_kmeans = end_time_kmeans - start_time_kmeans

        bb_topk_sims = np.full((n_clusters), -np.inf)
        bb_indices_to_show = np.full((n_clusters), -np.inf)

        # rank pairs by their mean saliency value
        bb_cls_attn1 = saliency_map1[bbs_mask]
        bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
        bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
        ranks = bb_cls_attn

        for k in range(n_clusters):
            for i, (label, rank) in enumerate(zip(kmeans_labels, ranks)):
                if rank > bb_topk_sims[label]:
                    bb_topk_sims[label] = rank
                    bb_indices_to_show[label] = i

        # get coordinates to show
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
            bb_indices_to_show]  # close bbs
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        # coordinates in descriptor map's dimensions
        img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        end_time_bb = time.time()
        end_time_corr = time.time()
        elapsed_bb = end_time_bb - start_time_bb

        elapsed_corr = end_time_corr - start_time_corr

        # print(f"all_corr: {elapsed_corr}, desc: {elapsed_desc}, chunk cosine: {elapsed_time_chunk_cosine}, saliency: {elapsed_saliencey}, kmeans: {elapsed_kmeans}, bb: {elapsed_bb}")

        return points1, points2, image1_pil, image2_pil

    def find_correspondences(
            self,
            pil_img1: Image.Image,
            pil_img2: Image.Image,
            num_pairs: int = 10,
            load_size: Tuple[int, int] = (224, 224),
            layer: int = 9,
            facet: str = 'key',
            bin: bool = True,
            thresh: float = 0.05,
            use_kmeans: bool = True,
            use_saliency: bool = True,
            use_best_buddies: bool = True
    ) -> Tuple[
        List[Tuple[float, float]],
        List[Tuple[float, float]],
        Image.Image,
        Image.Image
    ]:
        """
        Find correspondences between two images using descriptors, saliency maps, and k-means clustering.

        Parameters:
            pil_img1 (Image.Image): First input image.
            pil_img2 (Image.Image): Second input image.
            num_pairs (int): Number of correspondence pairs to return.
            load_size (int): Size to which images are resized during preprocessing.
            layer (int): Layer from which descriptors are extracted.
            facet (str): Facet or aspect of descriptors to use.
            bin (bool): Whether to apply binning to descriptors.
            thresh (float): Threshold for saliency maps to distinguish foreground from background.
            use_kmeans (bool): Whether to apply k-means clustering.
            use_saliency (bool): Whether to use saliency maps.
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

        # Step 1: Preprocess images and extract descriptors
        image1_batch, image1_pil, _ = self.preprocess(pil_img1, load_size)
        descriptors1 = self.extract_descriptors(image1_batch.to(self.device), layer, facet, bin)
        num_patches1 = self.num_patches

        image2_batch, image2_pil, _ = self.preprocess(pil_img2, load_size)
        descriptors2 = self.extract_descriptors(image2_batch.to(self.device), layer, facet, bin)
        num_patches2 = self.num_patches

        # Step 2: Extract and threshold saliency maps
        if use_saliency:
            saliency_map1, saliency_map2 = self.extract_saliency_maps(image1_batch.to(self.device))[0], \
                self.extract_saliency_maps(image2_batch.to(self.device))[0]
            fg_mask1, fg_mask2 = saliency_map1 > thresh, saliency_map2 > thresh
        else:
            fg_mask1, fg_mask2 = torch.ones(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=self.device), \
                torch.ones(num_patches2[0] * num_patches2[1], dtype=torch.bool, device=self.device)

        # Step 3: Compute cosine similarities between descriptors
        similarities = cosine_sim(descriptors1, descriptors2)

        # Step 4: Determine best buddies (mutual nearest neighbors)
        if use_best_buddies:
            sim_1, nn_1 = torch.max(similarities, dim=-1)
            sim_2, nn_2 = torch.max(similarities, dim=-2)

            # Assuming batch size of 1 and single channel
            sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
            sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

            image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
            bbs_mask = nn_2[nn_1] == image_idxs

            # Apply foreground masks
            fg_mask2_new_coors = nn_2[fg_mask2]
            fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool,
                                                  device=self.device)
            fg_mask2_mask_new_coors[fg_mask2_new_coors] = True

            bbs_mask &= fg_mask1 & fg_mask2_mask_new_coors
        else:
            bbs_mask = torch.ones(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=self.device)
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
                device=self.device
            )

            kmeans_labels = cluster_ids_x.detach().cpu().numpy()
            bb_topk_sims = np.full(n_clusters, -np.inf)
            bb_indices_to_show = np.full(n_clusters, -1, dtype=int)

            if use_saliency:
                bb_cls_attn = ((saliency_map1[bbs_mask] + saliency_map2[nn_1[bbs_mask]]) / 2).cpu().numpy()
            else:
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
        img1_indices = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices = nn_1[indices_to_show]

        img1_y = (img1_indices // num_patches1[1]).cpu().numpy()
        img1_x = (img1_indices % num_patches1[1]).cpu().numpy()
        img2_y = (img2_indices // num_patches2[1]).cpu().numpy()
        img2_x = (img2_indices % num_patches2[1]).cpu().numpy()

        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y, img1_x, img2_y, img2_x):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        end_time_corr = time.time()
        elapsed_corr = end_time_corr - start_time_corr

        # Optional: Print timing information for debugging
        # print(f"Total Corr Time: {elapsed_corr}, Descriptor Time: {elapsed_desc}, Cosine Similarity Time: {elapsed_time_chunk_cosine}, "
        #       f"Saliency Time: {elapsed_saliency if use_saliency else 'N/A'}, K-Means Time: {elapsed_kmeans if use_kmeans else 'N/A'}, "
        #       f"Best Buddies Time: {elapsed_bb if use_best_buddies else 'N/A'}")
        torch.cuda.empty_cache()
        return points1, points2, image1_pil, image2_pil
