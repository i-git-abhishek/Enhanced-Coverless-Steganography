import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from sklearn.cluster import KMeans
from config import Config

class DataPreprocessor:
    def __init__(self, dataset_name="VOC2007"):
        self.dataset_name = dataset_name
        self.device = Config.DEVICE
        self.k_clusters = Config.K_CLUSTERS_COCO if "COCO" in dataset_name.upper() else Config.K_CLUSTERS_VOC
        self.detector = None
        self.feature_extractor = None

    def _init_models(self):
        """Lazy-loads GPU models only if cache does not already exist."""
        if self.detector is not None:
            return
        weights_det = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.detector = fasterrcnn_resnet50_fpn(weights=weights_det).to(self.device).eval()
        self.det_transform = weights_det.transforms()

        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(self.device).eval()
        self.feat_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.inference_mode()
    def extract_raw_dataset_info(self, image_paths, cache_path="cache/VOC2007_raw.pkl"):
        """Loads cached Faster R-CNN detections & ResNet-101 hashes if available."""
        if os.path.exists(cache_path):
            print(f"[+] Loading cached raw extractions from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        self._init_models()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        raw_data = {}

        print(f"[*] Extracting Faster R-CNN objects & ResNet-101 hashes on {len(image_paths)} images...")
        for img_path in tqdm(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            det_tensor = self.det_transform(img).unsqueeze(0).to(self.device)
            preds = self.detector(det_tensor)[0]

            boxes = preds['boxes'].cpu().numpy()
            scores = preds['scores'].cpu().numpy()
            labels = preds['labels'].cpu().numpy()

            valid_idx = np.where(scores > Config.CONF_THRESHOLD)[0]
            detected_objects = []
            for idx in valid_idx:
                lbl_id = labels[idx]
                if lbl_id < len(Config.COCO_LABELS) and Config.COCO_LABELS[lbl_id] != 'N/A':
                    x1, y1, x2, y2 = boxes[idx]
                    area = float((x2 - x1) * (y2 - y1))
                    detected_objects.append({
                        "label": Config.COCO_LABELS[lbl_id],
                        "confidence": float(scores[idx]),
                        "area": area,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })

            detected_objects.sort(key=lambda x: x["area"], reverse=True)

            feat_tensor = self.feat_transform(img).unsqueeze(0).to(self.device)
            deep_feat = self.feature_extractor(feat_tensor).view(-1).cpu().numpy()

            f_min, f_max = deep_feat.min(), deep_feat.max()
            norm_feat = (deep_feat - f_min) / (f_max - f_min) if (f_max - f_min) > 1e-8 else np.zeros_like(deep_feat)
            binary_hash = (norm_feat > Config.HASH_THRESHOLD).astype(np.uint8)

            raw_data[img_path] = {
                "objects": detected_objects,
                "binary_hash": binary_hash
            }

        with open(cache_path, "wb") as f:
            pickle.dump(raw_data, f)
        return raw_data

    def build_index_and_clusters(self, raw_data, n_bits=Config.N_BITS, m_objects=Config.M_OBJECTS):
        """
        Section III-A1 to III-A4:
        1. Balances the 2^n object sequence groups across all detected top-M objects
           so that all 2^n binary sequences have balanced image coverage (Fig. 5).
        2. Clusters binarized ResNet-101 features into k clusters using K-means++.
        3. Constructs the Image Sequence Index (Table I).
        """
        num_groups = 2 ** n_bits
        valid_paths = sorted([p for p, d in raw_data.items() if len(d["objects"]) > 0])
        hash_matrix = np.array([raw_data[p]["binary_hash"] for p in valid_paths], dtype=np.float32)

        n_clusters = min(self.k_clusters, len(valid_paths))
        print(f"[*] Running K-means++ clustering (k={n_clusters}) on {len(valid_paths)} valid images...")
        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            max_iter=Config.KMEANS_MAX_ITER,
            random_state=42,
            n_init=3
        )
        cluster_ids = kmeans.fit_predict(hash_matrix)
        path_to_cluster = {p: f"C_{c_id}" for p, c_id in zip(valid_paths, cluster_ids)}

        # Collect all top-M object instances across the dataset
        all_instances = []
        for img_path in valid_paths:
            top_m = raw_data[img_path]["objects"][:m_objects]
            for rank_idx, obj in enumerate(top_m, start=1):
                all_instances.append((
                    obj["label"],
                    -round(obj["area"], 1),
                    -round(obj["confidence"], 4),
                    img_path,
                    rank_idx,
                    obj
                ))

        # Sort alphabetically by object label, then deterministically by area & confidence (Section III-A1)
        all_instances.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

        # Partition evenly into 2^n groups so every binary sequence has balanced dataset coverage
        total_inst = len(all_instances)
        obj_dict = {}
        instance_seq_map = {}

        for idx, item in enumerate(all_instances):
            lbl, _, _, img_path, rank_idx, _ = item
            group_idx = min(num_groups - 1, int((idx * num_groups) / total_inst))
            bin_seq = format(group_idx, f'0{n_bits}b')
            instance_seq_map[(img_path, rank_idx)] = (group_idx, bin_seq)
            if lbl not in obj_dict:
                obj_dict[lbl] = (group_idx, bin_seq)

        # Build Image Sequence Index Structure (Table I)
        index_rows = []
        cluster_to_images = {}
        image_to_top_objects = {}

        for img_path in valid_paths:
            cluster_label = path_to_cluster[img_path]
            cluster_to_images.setdefault(cluster_label, []).append(img_path)

            top_m_objs = raw_data[img_path]["objects"][:m_objects]
            image_to_top_objects[img_path] = []

            for rank_idx, obj in enumerate(top_m_objs, start=1):
                lbl = obj["label"]
                _, bin_seq = instance_seq_map[(img_path, rank_idx)]

                index_rows.append({
                    "filename": img_path,
                    "object_label": lbl,
                    "object_number": rank_idx,
                    "object_sequence": bin_seq,
                    "cluster_membership": cluster_label
                })
                image_to_top_objects[img_path].append({
                    "object_number": rank_idx,
                    "object_label": lbl,
                    "object_sequence": bin_seq,
                    "cluster_membership": cluster_label,
                    "confidence": obj["confidence"],
                    "area": obj["area"],
                    "bbox": obj["bbox"]
                })

        index_df = pd.DataFrame(index_rows)
        return {
            "index_df": index_df,
            "obj_dict": obj_dict,
            "cluster_to_images": cluster_to_images,
            "image_to_top_objects": image_to_top_objects,
            "kmeans": kmeans
        }