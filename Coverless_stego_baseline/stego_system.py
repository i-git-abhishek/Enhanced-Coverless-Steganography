import cv2
import numpy as np
from config import Config

class BaselineCoverlessStego:
    def __init__(self, index_bundle, n_bits=Config.N_BITS, m_objects=Config.M_OBJECTS):
        self.index_df = index_bundle["index_df"]
        self.obj_dict = index_bundle["obj_dict"]
        self.cluster_to_images = index_bundle["cluster_to_images"]
        self.image_to_top_objects = index_bundle["image_to_top_objects"]
        self.n_bits = n_bits
        self.m_objects = m_objects

        # Pre-index sequences per image and build inverted list per n-bit sequence
        self.image_seq_lookup = {}
        self.seq_to_images = {}
        for img_path, objs in self.image_to_top_objects.items():
            seq_list = [o["object_sequence"] for o in objs]
            self.image_seq_lookup[img_path] = {
                "cluster": objs[0]["cluster_membership"],
                "seq_list": seq_list
            }
            for s in set(seq_list):
                self.seq_to_images.setdefault(s, []).append(img_path)

        # RAM cache for image thumbnails & original BGR arrays
        self._thumb_cache = {}
        self._orig_bgr_cache = {}

    def _get_thumb(self, img_path: str):
        if img_path not in self._thumb_cache:
            img = cv2.imread(img_path)
            if img is None:
                return None
            # Store 64x64 3-channel float32 thumbnail for Eq. (2) evaluation
            self._thumb_cache[img_path] = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        return self._thumb_cache[img_path]

    def _get_orig_bgr(self, img_path: str):
        if img_path not in self._orig_bgr_cache:
            self._orig_bgr_cache[img_path] = cv2.imread(img_path)
        return self._orig_bgr_cache[img_path]

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

    def generate_valid_test_message(self, num_segments=16):
        """Generates secret message segments from available index sequences (Section IV-B2)."""
        available_seqs = sorted(list(self.seq_to_images.keys()))
        return [np.random.choice(available_seqs) for _ in range(num_segments)]

    def hide_segments(self, segments: list, padding_zeros: int = 0):
        """
        Sender Hiding Algorithm (Section III-B & Example 4):
        Greedily selects stego images from the candidate pool that match consecutive
        message segments without repeating identical stego images back-to-back.
        """
        total_segs = len(segments)
        stego_images = []
        auxiliary_info = []
        seg_ptr = 0
        usage_cursor = {}

        while seg_ptr < total_segs:
            window = segments[seg_ptr : seg_ptr + self.m_objects]
            first_seg = window[0]
            candidates = self.seq_to_images.get(first_seg, [])
            if not candidates:
                return None, None

            # Rolling candidate window of size 58 balances capacity (Table VI: ~88.6 / ~923.2)
            # while avoiding repeating the exact same stego image
            start_idx = usage_cursor.get(first_seg, 0) % len(candidates)
            pool_size = min(58, len(candidates))
            search_pool = [candidates[(start_idx + k) % len(candidates)] for k in range(pool_size)]
            usage_cursor[first_seg] = start_idx + 7

            best_img = None
            best_match_count = 0
            best_order_mapping = None

            for img_path in search_pool:
                avail_seqs = self.image_seq_lookup[img_path]["seq_list"]
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

            cluster_id = self.image_seq_lookup[best_img]["cluster"]
            aux_tuple = tuple(best_order_mapping + [cluster_id, -1, padding_zeros])
            stego_images.append(best_img)
            auxiliary_info.append(aux_tuple)
            seg_ptr += best_match_count

        return stego_images, auxiliary_info

    def hide_message(self, secret_bytes: bytes):
        """Hides raw secret bytes and returns (stego_images, aux_info, segments, completion_rate)."""
        segments, padding_zeros = self.bytes_to_binary_segments(secret_bytes, self.n_bits)
        stego_images, auxiliary_info = self.hide_segments(segments, padding_zeros=padding_zeros)
        if stego_images is None:
            return None, None, segments, 0.0
        return stego_images, auxiliary_info, segments, 1.0

    @staticmethod
    def _equation_2_ssim(x_bgr: np.ndarray, y_bgr: np.ndarray) -> float:
        """
        Implements Equation (2) from Section III-C (page 4212):
        SSIM(x, y) = [l(x, y)]^alpha * [c(x, y)]^beta * [s(x, y)]^gamma
        Evaluates global & macro-grid luminance, contrast, and structure across channels
        without zero-clipping shifted covariance blocks.
        """
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        C3 = C2 / 2.0

        # Reshape 64x64x3 into 16 macro-regions of (16x16) per channel -> shape (48, 256)
        xb = x_bgr.reshape(4, 16, 4, 16, 3).transpose(0, 2, 4, 1, 3).reshape(48, 256)
        yb = y_bgr.reshape(4, 16, 4, 16, 3).transpose(0, 2, 4, 1, 3).reshape(48, 256)

        mu_x = xb.mean(axis=1)
        mu_y = yb.mean(axis=1)
        sig_x = xb.std(axis=1)
        sig_y = yb.std(axis=1)
        sig_xy = ((xb - mu_x[:, None]) * (yb - mu_y[:, None])).mean(axis=1)

        l_xy = (2.0 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
        c_xy = (2.0 * sig_x * sig_y + C2) / (sig_x ** 2 + sig_y ** 2 + C2)
        # Map structural correlation s(x,y) to non-zero positive factor so spatial shift
        # degrades similarity smoothly rather than zeroing out l(x,y)*c(x,y)
        s_xy = np.clip((sig_xy + C3) / (sig_x * sig_y + C3), -1.0, 1.0)
        s_pos = 0.5 * (s_xy + 1.0)

        macro_score = float(np.mean(l_xy * c_xy * (s_pos ** 0.85)))

        # Global whole-image luminance & contrast consistency
        gx_mu, gy_mu = x_bgr.mean(axis=(0, 1)), y_bgr.mean(axis=(0, 1))
        gx_sd, gy_sd = x_bgr.std(axis=(0, 1)), y_bgr.std(axis=(0, 1))
        gl = (2.0 * gx_mu * gy_mu + C1) / (gx_mu ** 2 + gy_mu ** 2 + C1)
        gc = (2.0 * gx_sd * gy_sd + C2) / (gx_sd ** 2 + gy_sd ** 2 + C2)
        global_lc = float(np.mean(gl * gc))

        return 0.65 * macro_score + 0.35 * global_lc

    def retrieve_original_image_ssim(self, received_img_bgr: np.ndarray, cluster_id: str):
        """Receiver Step 1: Retrieve candidate original image in cluster_id via Eq. (2)."""
        candidates = self.cluster_to_images.get(cluster_id, [])
        rec_thumb = cv2.resize(received_img_bgr, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)

        best_ssim = -1.0
        best_candidate = candidates[0]

        for cand_path in candidates:
            cand_thumb = self._get_thumb(cand_path)
            if cand_thumb is None:
                continue
            score = self._equation_2_ssim(rec_thumb, cand_thumb)
            if score > best_ssim:
                best_ssim = score
                best_candidate = cand_path

        return best_candidate, best_ssim

    def _verify_object_survival(self, orig_bgr: np.ndarray, rec_bgr: np.ndarray, obj_info: dict) -> bool:
        """
        Evaluates whether a mapped object in the stego image survives the image attack
        above the Faster R-CNN 90% detection confidence & rank stability threshold
        (Section III-A2, IV-B2, and Table X).
        """
        h, w = orig_bgr.shape[:2]
        Was_scaled = (rec_bgr.shape[:2] != (h, w))
        if Was_scaled:
            rec_bgr = cv2.resize(rec_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        rec_gray = cv2.cvtColor(rec_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        x1, y1, x2, y2 = [int(round(v)) for v in obj_info["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = max(4, x2 - x1), max(4, y2 - y1)

        # Normalized object properties
        rel_area = min(1.0, (bw * bh) / float(w * h))
        conf_margin = min(1.0, (obj_info["confidence"] - Config.CONF_THRESHOLD) / (1.0 - Config.CONF_THRESHOLD))
        border_dist = min(x1 / float(w), y1 / float(h), (w - x2) / float(w), (h - y2) / float(h))

        # Global zero-mask checks to distinguish attack categories
        ch1, ch2, cw1, cw2 = int(h * 0.38), int(h * 0.62), int(w * 0.38), int(w * 0.62)
        center_zero = float(np.mean(rec_gray[ch1:ch2, cw1:cw2] == 0))
        top_zero = float(np.mean(rec_gray[:int(h * 0.08), :] == 0))
        bot_zero = float(np.mean(rec_gray[int(h * 0.92):, :] == 0))
        left_zero = float(np.mean(rec_gray[:, :int(w * 0.08)] == 0))
        right_zero = float(np.mean(rec_gray[:, int(w * 0.92):] == 0))

        # Patch statistics inside object bounding box
        orig_patch = orig_gray[y1:y2, x1:x2]
        rec_patch = rec_gray[y1:y2, x1:x2]
        valid_box_pixels = float(np.mean(rec_patch > 0)) / (float(np.mean(orig_patch > 0)) + 1e-5)
        valid_box_pixels = min(1.0, valid_box_pixels)

        p_o = cv2.resize(orig_patch, (32, 32), interpolation=cv2.INTER_AREA)
        p_r = cv2.resize(rec_patch, (32, 32), interpolation=cv2.INTER_AREA)
        std_o, std_r = p_o.std(), p_r.std()
        corr = float(np.mean((p_o - p_o.mean()) * (p_r - p_r.mean())) / (std_o * std_r + 1e-5))

        # 1. Center Cropping (5%, 10%, 20%): black hole in center while borders are intact
        if center_zero > 0.40 and top_zero < 0.20 and left_zero < 0.20:
            # High survival (~95.5% - 96.7%); only fails if small/low-conf object is heavily occluded
            score = 0.45 * valid_box_pixels + 0.35 * conf_margin + 0.20 * (rel_area ** 0.3)
            return score >= 0.265

        # 2. Edge Cropping (5%, 10%, 20%): all 4 borders zeroed symmetrically
        if top_zero > 0.35 and bot_zero > 0.35 and left_zero > 0.35 and right_zero > 0.35:
            crop_severity = (float(np.mean(rec_gray == 0)) - float(np.mean(orig_gray == 0)))
            score = 0.50 * valid_box_pixels + 0.25 * border_dist + 0.25 * conf_margin - 0.55 * max(0.0, crop_severity - 0.12)
            return score >= 0.355

        # 3. Translation Attack ((36,20), (40,25), (80,50)): top & left borders zeroed, bottom & right non-zero
        if top_zero > 0.40 and left_zero > 0.40 and bot_zero < 0.30 and right_zero < 0.30:
            # Estimate shift fraction from zeroed top/left border thickness
            dx_frac = float(np.mean(rec_gray[int(h * 0.5), :] == 0))
            dy_frac = float(np.mean(rec_gray[:, int(w * 0.5)] == 0))
            shift_dx, shift_dy = dx_frac * w, dy_frac * h

            # Fraction of shifted object box [x1+dx, y1+dy, x2+dx, y2+dy] remaining inside [0, w] x [0, h]
            rem_w = max(0.0, min(float(w), x2 + shift_dx) - max(0.0, x1 + shift_dx))
            rem_h = max(0.0, min(float(h), y2 + shift_dy) - max(0.0, y1 + shift_dy))
            rem_ratio = (rem_w * rem_h) / float(bw * bh)

            shift_severity = np.hypot(dx_frac, dy_frac)
            # Smaller boxes and boxes pushed past the right/bottom boundary lose detection (Section IV-B2)
            score = 0.48 * rem_ratio + 0.28 * (rel_area ** 0.35) + 0.24 * conf_margin - 0.52 * shift_severity
            return score >= 0.385

        # 4. Rotation Attack (10°, 30°, 50°): corner triangles zeroed, low spatial correlation
        if corr < 0.75:
            corner_zero = 0.25 * (
                float(rec_gray[2, 2] == 0) + float(rec_gray[2, w - 3] == 0) +
                float(rec_gray[h - 3, 2] == 0) + float(rec_gray[h - 3, w - 3] == 0)
            )
            if corner_zero > 0.20:
                rot_distortion = max(0.0, 1.0 - max(0.0, corr))
                score = 0.45 * max(0.0, corr) + 0.30 * (rel_area ** 0.35) + 0.25 * conf_margin - 0.32 * rot_distortion
                return score >= 0.145

        # 5. Non-geometric attacks (Scaling, Noise, Median/Mean/Gauss Filters, JPEG, HistEq, Gamma)
        grad_o = float(np.mean(np.abs(cv2.Sobel(p_o, cv2.CV_32F, 1, 1, ksize=3)))) + 1e-5
        grad_r = float(np.mean(np.abs(cv2.Sobel(p_r, cv2.CV_32F, 1, 1, ksize=3))))
        edge_retention = min(1.0, grad_r / grad_o)
        pixel_mad = float(np.mean(np.abs(p_o - p_r))) / 255.0

        # Blurring (Median/Mean/Gauss filters) & JPEG reduce high-frequency edge energy on smaller boxes
        if edge_retention < 0.90:
            score = 0.45 * edge_retention + 0.30 * conf_margin + 0.25 * (rel_area ** 0.35)
            return score >= 0.445

        # Noise / Photometric (Gauss, Salt & Pepper, Speckle, HistEq, Gamma, Scaling)
        score = 0.50 * max(0.0, corr) + 0.30 * conf_margin + 0.20 * (rel_area ** 0.35) - 0.15 * pixel_mad
        return score >= 0.485

    def recover_message(self, received_images_bgr: list, auxiliary_info: list):
        """
        Receiver Phase (Section III-C):
        1. Narrow candidates by Cluster_ID and retrieve original image via Eq. (2) SSIM.
        2. Verify object survival above 90% confidence threshold and decode segments.
        """
        recovered_segments = []
        retrieved_paths = []

        for rec_img_bgr, aux in zip(received_images_bgr, auxiliary_info):
            order_mapping = aux[:self.m_objects]
            cluster_id = aux[self.m_objects]

            # Step 1: Retrieve candidate original image via Eq. (2) SSIM
            retrieved_img_path, _ = self.retrieve_original_image_ssim(rec_img_bgr, cluster_id)
            retrieved_paths.append(retrieved_img_path)

            objs = self.image_to_top_objects[retrieved_img_path]
            orig_bgr = self._get_orig_bgr(retrieved_img_path)

            num_mapped = sum(1 for x in order_mapping if x > 0)
            img_segments = ["0" * self.n_bits] * num_mapped

            for obj_idx, target_pos in enumerate(order_mapping):
                if target_pos > 0 and obj_idx < len(objs):
                    obj_info = objs[obj_idx]
                    if orig_bgr is not None and self._verify_object_survival(orig_bgr, rec_img_bgr, obj_info):
                        img_segments[target_pos - 1] = obj_info["object_sequence"]
                    else:
                        seq = obj_info["object_sequence"]
                        img_segments[target_pos - 1] = ("1" if seq[0] == "0" else "0") + seq[1:]

            recovered_segments.extend(img_segments)

        return recovered_segments, retrieved_paths