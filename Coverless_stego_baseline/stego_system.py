import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from config import Config

class BaselineCoverlessStego:
    def __init__(self, index_bundle, n_bits=Config.N_BITS, m_objects=Config.M_OBJECTS):
        self.index_df = index_bundle["index_df"]
        self.obj_dict = index_bundle["obj_dict"]
        self.cluster_to_images = index_bundle["cluster_to_images"]
        self.image_to_top_objects = index_bundle["image_to_top_objects"]
        self.n_bits = n_bits
        self.m_objects = m_objects

        # Pre-index available sequences per image for fast greedy multi-object matching
        self.image_seq_lookup = {}
        for img_path, objs in self.image_to_top_objects.items():
            self.image_seq_lookup[img_path] = {
                "cluster": objs[0]["cluster_membership"],
                "seq_list": [o["object_sequence"] for o in objs] # ordered 1..M
            }
            
    def generate_valid_test_message(self, num_segments=10):
        """
        Generates secret message segments strictly from available index sequences
        (matching Section IV-B2 protocol for robustness benchmarking).
        """
        available_seqs = list({
            seq for info in self.image_seq_lookup.values() for seq in info["seq_list"]
        })
        if not available_seqs:
            raise ValueError("No sequences found in index.")
        chosen_segments = [np.random.choice(available_seqs) for _ in range(num_segments)]
        return chosen_segments

    def hide_segments(self, segments: list):
        """Hides pre-validated segments directly without arbitrary random bytes failure."""
        total_segs = len(segments)
        stego_images = []
        auxiliary_info = []
        seg_ptr = 0

        while seg_ptr < total_segs:
            window = segments[seg_ptr : seg_ptr + self.m_objects]
            best_img = None
            best_match_count = 0
            best_order_mapping = None

            for img_path, info in self.image_seq_lookup.items():
                avail_seqs = info["seq_list"]
                used_obj_indices = set()
                order_array = [0] * self.m_objects
                match_count = 0

                for seg_order_idx, target_seg in enumerate(window, start=1):
                    matched = False
                    for obj_idx, obj_seq in enumerate(avail_seqs):
                        if obj_idx not in used_obj_indices and obj_seq == target_seg:
                            used_obj_indices.add(obj_idx)
                            order_array[obj_idx] = seg_order_idx
                            match_count += 1
                            matched = True
                            break
                    if not matched:
                        break

                if match_count > best_match_count:
                    best_match_count = match_count
                    best_img = img_path
                    best_order_mapping = order_array
                    if best_match_count == len(window):
                        break

            if best_match_count == 0:
                # Fallback: find any image that contains at least the single next segment
                target_seg = segments[seg_ptr]
                for img_path, info in self.image_seq_lookup.items():
                    if target_seg in info["seq_list"]:
                        best_img = img_path
                        best_match_count = 1
                        best_order_mapping = [0] * self.m_objects
                        obj_pos = info["seq_list"].index(target_seg)
                        best_order_mapping[obj_pos] = 1
                        break

            if best_img is None:
                return None, None

            cluster_id = self.image_seq_lookup[best_img]["cluster"]
            aux_tuple = tuple(best_order_mapping + [cluster_id, -1, 0])
            stego_images.append(best_img)
            auxiliary_info.append(aux_tuple)
            seg_ptr += best_match_count

        return stego_images, auxiliary_info

    
    @staticmethod
    def bytes_to_binary_segments(secret_bytes: bytes, n_bits: int):
        """Converts raw bytes to n-bit binary segments with zero-padding (Section III-B)."""
        bit_str = "".join(format(b, '08b') for b in secret_bytes)
        remainder = len(bit_str) % n_bits
        padding_zeros = (n_bits - remainder) % n_bits
        if padding_zeros > 0:
            bit_str += "0" * padding_zeros
            
        segments = [bit_str[i:i + n_bits] for i in range(0, len(bit_str), n_bits)]
        return segments, padding_zeros

    def hide_message(self, secret_bytes: bytes):
        """
        Sender Phase (Section III-B & Example 4):
        Prioritizes images that simultaneously map the most secret message segments.
        Returns:
            stego_images (list of filepaths),
            auxiliary_info (list of tuples),
            original_segments (list of n-bit strings)
        """
        segments, padding_zeros = self.bytes_to_binary_segments(secret_bytes, self.n_bits)
        total_segs = len(segments)
        
        stego_images = []
        auxiliary_info = []
        
        seg_ptr = 0
        while seg_ptr < total_segs:
            # Look at the next window of up to M_OBJECTS segments
            window = segments[seg_ptr : seg_ptr + self.m_objects]
            
            best_img = None
            best_match_count = 0
            best_order_mapping = None

            # Search dataset for an image whose top-M objects match the longest prefix of `window`
            for img_path, info in self.image_seq_lookup.items():
                avail_seqs = info["seq_list"] # length <= M
                used_obj_indices = set()
                order_array = [0] * self.m_objects
                match_count = 0

                for seg_order_idx, target_seg in enumerate(window, start=1):
                    # Check if an unused object in this image matches target_seg
                    matched = False
                    for obj_idx, obj_seq in enumerate(avail_seqs):
                        if obj_idx not in used_obj_indices and obj_seq == target_seg:
                            used_obj_indices.add(obj_idx)
                            order_array[obj_idx] = seg_order_idx
                            match_count += 1
                            matched = True
                            break
                    if not matched:
                        # Stop contiguous segment matching for this image
                        break

                if match_count > best_match_count:
                    best_match_count = match_count
                    best_img = img_path
                    best_order_mapping = order_array
                    if best_match_count == len(window):
                        break # Optimal match found for current window

            if best_match_count == 0:
                # Sequence Index Integrity Failure (cannot map segment)
                return None, None, segments, seg_ptr / total_segs

            cluster_id = self.image_seq_lookup[best_img]["cluster"]
            # Format from Example 4: (Order_1, ..., Order_M, Cluster_ID, -1, Padding_Zeros)
            aux_tuple = tuple(best_order_mapping + [cluster_id, -1, padding_zeros])
            
            stego_images.append(best_img)
            auxiliary_info.append(aux_tuple)
            seg_ptr += best_match_count

        return stego_images, auxiliary_info, segments, 1.0

    def retrieve_original_image_ssim(self, received_img_bgr: np.ndarray, cluster_id: str):
        """
        Receiver Step 1 (Section III-C & Eq. 2):
        Finds the candidate image in `cluster_id` with maximum SSIM to `received_img_bgr`.
        """
        candidates = self.cluster_to_images.get(cluster_id, [])
        if not candidates:
            raise ValueError(f"Cluster {cluster_id} not found in index.")

        # Convert received image to grayscale for fast and standard SSIM calculation
        rec_gray = cv2.cvtColor(received_img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = rec_gray.shape

        best_ssim = -1.0
        best_candidate = candidates[0]

        for cand_path in candidates:
            cand_bgr = cv2.imread(cand_path)
            if cand_bgr is None:
                continue
            cand_gray = cv2.cvtColor(cand_bgr, cv2.COLOR_BGR2GRAY)
            
            # If attack changed image dimensions (e.g., scaling), resize candidate to match
            if cand_gray.shape != (h, w):
                cand_gray = cv2.resize(cand_gray, (w, h))

            score = ssim(rec_gray, cand_gray, data_range=255)
            if score > best_ssim:
                best_ssim = score
                best_candidate = cand_path

        return best_candidate, best_ssim

    def recover_message(self, received_images_bgr: list, auxiliary_info: list):
        """
        Receiver Phase (Section III-C & Example 5):
        1. Narrow candidates by Cluster_ID and match via SSIM.
        2. Read top-M object labels of the retrieved original image.
        3. Reorder segments using the auxiliary mapping rules.
        """
        recovered_segments = []
        retrieved_paths = []

        for rec_img_bgr, aux in zip(received_images_bgr, auxiliary_info):
            order_mapping = aux[:self.m_objects]
            cluster_id = aux[self.m_objects]
            padding_zeros = aux[-1]

            # Step 1: Retrieve original unattacked image from cluster via SSIM
            retrieved_img_path, _ = self.retrieve_original_image_ssim(rec_img_bgr, cluster_id)
            retrieved_paths.append(retrieved_img_path)

            # Step 2: Obtain object labels/sequences on retrieved original image
            objs = self.image_to_top_objects[retrieved_img_path]
            
            # Step 3: Reconstruct segments according to order_mapping (1-indexed, 0 = unused)
            num_mapped = sum(1 for x in order_mapping if x > 0)
            img_segments = ["0" * self.n_bits] * num_mapped

            for obj_idx, target_pos in enumerate(order_mapping):
                if target_pos > 0 and obj_idx < len(objs):
                    img_segments[target_pos - 1] = objs[obj_idx]["object_sequence"]

            recovered_segments.extend(img_segments)

        return recovered_segments, retrieved_paths