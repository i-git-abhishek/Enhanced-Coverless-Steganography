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
        
        # 1. Load Pretrained Faster R-CNN for Object Detection
        weights_det = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.detector = fasterrcnn_resnet50_fpn(weights=weights_det).to(self.device).eval()
        self.det_transform = weights_det.transforms()
        
        # 2. Load Pretrained ResNet-101 for 2048-D Deep Feature Extraction (Section III-A3)
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # Remove the final classification fc layer to get 2048-D pooled features
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(self.device).eval()
        self.feat_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_raw_dataset_info(self, image_paths, cache_path="cache/raw_extractions.pkl"):
        """Runs GPU inference once over the dataset and caches raw detections + binary hashes."""
        if os.path.exists(cache_path):
            print(f"[+] Loading cached raw extractions from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        raw_data = {}

        print(f"[*] Extracting Faster R-CNN objects & ResNet-101 hashes on {len(image_paths)} images...")
        for img_path in tqdm(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            # A. Faster R-CNN Object Detection
            det_tensor = self.det_transform(img).unsqueeze(0).to(self.device)
            preds = self.detector(det_tensor)[0]
            
            boxes = preds['boxes'].cpu().numpy()
            scores = preds['scores'].cpu().numpy()
            labels = preds['labels'].cpu().numpy()

            # Filter confidence > 90% (Section III-A2)
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

            # Sort objects by bounding box size from largest to smallest (Section III-A2)
            detected_objects.sort(key=lambda x: x["area"], reverse=True)

            # B. ResNet-101 Feature Extraction & Eq. (1) Binarization
            feat_tensor = self.feat_transform(img).unsqueeze(0).to(self.device)
            deep_feat = self.feature_extractor(feat_tensor).view(-1).cpu().numpy() # (2048,)
            
            # Min-max normalization to [0, 1]
            f_min, f_max = deep_feat.min(), deep_feat.max()
            if f_max - f_min > 1e-8:
                norm_feat = (deep_feat - f_min) / (f_max - f_min)
            else:
                norm_feat = np.zeros_like(deep_feat)
                
            # Equation (1): h_t = 1 if f_t > 0.5 else 0
            binary_hash = (norm_feat > Config.HASH_THRESHOLD).astype(np.uint8)

            raw_data[img_path] = {
                "objects": detected_objects,
                "binary_hash": binary_hash
            }

        with open(cache_path, "wb") as f:
            pickle.dump(raw_data, f)
        return raw_data

    def build_object_sequence_dictionary(self, raw_data, n_bits=Config.N_BITS):
        """
        Section III-A1: Groups alphabetical object labels into 2^n groups while balancing
        image frequency distribution across groups.
        """
        num_groups = 2 ** n_bits
        
        # Count label occurrences across dataset
        label_counts = {}
        for item in raw_data.values():
            for obj in item["objects"]:
                lbl = obj["label"]
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

        # Get all valid recognizable COCO labels sorted alphabetically
        all_labels = sorted(list({l for l in Config.COCO_LABELS if l not in ['__background__', 'N/A']}))
        
        # Ensure we have at least num_groups items; partition alphabetically with greedy frequency balancing
        dictionary = {}
        if len(all_labels) >= num_groups:
            # Partition `len(all_labels)` sorted labels into `num_groups` contiguous alphabetical buckets
            # merging rarer adjacent labels as shown in Fig. 5
            weights = [label_counts.get(l, 1) for l in all_labels]
            total_weight = sum(weights)
            target_weight = total_weight / num_groups
            
            group_idx = 0
            current_weight = 0
            for i, lbl in enumerate(all_labels):
                rem_labels = len(all_labels) - i
                rem_groups = num_groups - group_idx
                
                bin_seq = format(group_idx, f'0{n_bits}b')
                dictionary[lbl] = (group_idx, bin_seq)
                current_weight += weights[i]
                
                # Advance to next group if target weight reached (and enough labels remain) or forced by remaining groups
                if (current_weight >= target_weight and rem_labels > rem_groups) or (rem_labels == rem_groups):
                    group_idx = min(num_groups - 1, group_idx + 1)
                    current_weight = 0
        else:
            raise ValueError(f"Number of recognizable classes ({len(all_labels)}) is less than 2^{n_bits} ({num_groups}).")

        return dictionary

    def build_index_and_clusters(self, raw_data, n_bits=Config.N_BITS, m_objects=Config.M_OBJECTS):
        """
        Section III-A3 & III-A4: Performs K-means++ clustering and builds the Image Sequence Index (Table I).
        """
        obj_dict = self.build_object_sequence_dictionary(raw_data, n_bits=n_bits)
        
        valid_paths = [p for p, d in raw_data.items() if len(d["objects"]) > 0]
        hash_matrix = np.array([raw_data[p]["binary_hash"] for p in valid_paths], dtype=np.float32)
        
        n_clusters = min(self.k_clusters, len(valid_paths))
        print(f"[*] Running K-means++ clustering (k={n_clusters}) on {len(valid_paths)} valid images...")
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=Config.KMEANS_MAX_ITER, random_state=42, n_init=3)
        cluster_ids = kmeans.fit_predict(hash_matrix)

        # Build Table I: Image Sequence Index Structure
        index_rows = []
        cluster_to_images = {}
        image_to_top_objects = {}

        for img_path, c_id in zip(valid_paths, cluster_ids):
            cluster_label = f"C_{c_id}"
            cluster_to_images.setdefault(cluster_label, []).append(img_path)
            
            top_m_objs = raw_data[img_path]["objects"][:m_objects]
            image_to_top_objects[img_path] = []
            
            for rank_idx, obj in enumerate(top_m_objs, start=1):
                lbl = obj["label"]
                _, bin_seq = obj_dict[lbl]
                
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
                    "cluster_membership": cluster_label
                })

        index_df = pd.DataFrame(index_rows)
        return {
            "index_df": index_df,
            "obj_dict": obj_dict,
            "cluster_to_images": cluster_to_images,
            "image_to_top_objects": image_to_top_objects,
            "kmeans": kmeans
        }