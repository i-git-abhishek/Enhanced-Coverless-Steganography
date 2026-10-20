import os
import glob
import cv2
import numpy as np
import pandas as pd
from torchvision.datasets import VOCDetection
from preprocessing import DataPreprocessor
from stego_system import BaselineCoverlessStego
from attacks import ImageAttacks
from config import Config

# Exact Reference Values from Lin & Liang (IEEE TMM 2026) Table VI & Table X on VOC 2007
PAPER_TABLE_VI_VOC2007 = {10: 9.8, 100: 88.6, 1024: 923.2}

PAPER_TABLE_X_VOC2007 = {
    "Center Cropping 5%": 96.61,
    "Center Cropping 10%": 96.67,
    "Center Cropping 20%": 95.55,
    "Edge Cropping 5%": 97.16,
    "Edge Cropping 10%": 95.93,
    "Edge Cropping 20%": 84.79,
    "Rotation 10°": 85.31,
    "Rotation 30°": 66.04,
    "Rotation 50°": 41.58,
    "Translation (36, 20)": 72.29,
    "Translation (40, 25)": 62.80,
    "Translation (80, 50)": 59.35,
    "Scaling": 97.75,
    "Gauss Noise": 97.95,
    "Salt & Pepper Noise": 96.49,
    "Speckle Noise": 97.55,
    "Median Filter": 93.83,
    "Mean Filter": 93.52,
    "Gauss Filter": 95.34,
    "JPEG Compression": 93.69,
    "Histogram Equalization": 97.43,
    "Gamma Correction": 97.89,
}

def compute_segment_accuracy(orig_segments: list, rec_segments: list) -> float:
    """Equation (3): Segment recovery accuracy ACC."""
    m = len(orig_segments)
    if m == 0:
        return 0.0
    matches = sum(1 for o, r in zip(orig_segments, rec_segments) if o == r)
    return (matches / m) * 100.0

def download_or_get_voc2007(root_dir="./data"):
    """Finds PASCAL VOC 2007 JPEGImages or downloads if absent."""
    voc_img_dir = os.path.join(root_dir, "VOCdevkit", "VOC2007", "JPEGImages")
    if not os.path.exists(voc_img_dir):
        print("[*] Downloading PASCAL VOC 2007 dataset via torchvision...")
        VOCDetection(root=root_dir, year="2007", image_set="trainval", download=True)
        VOCDetection(root=root_dir, year="2007", image_set="test", download=True)
    img_paths = sorted(glob.glob(os.path.join(voc_img_dir, "*.jpg")))
    print(f"[+] Found {len(img_paths)} images in VOC 2007.")
    return img_paths

def run_baseline_experiments(image_paths, dataset_name="VOC2007", num_robustness_trials=25):
    # 1. Phase A: Preprocess & Index (Loads existing cache/VOC2007_raw.pkl automatically)
    preprocessor = DataPreprocessor(dataset_name=dataset_name)
    raw_data = preprocessor.extract_raw_dataset_info(
        image_paths, cache_path=f"cache/{dataset_name}_raw.pkl"
    )
    index_bundle = preprocessor.build_index_and_clusters(
        raw_data, n_bits=Config.N_BITS, m_objects=Config.M_OBJECTS
    )
    stego = BaselineCoverlessStego(index_bundle, n_bits=Config.N_BITS, m_objects=Config.M_OBJECTS)

    # 2. Experiment 1: Steganographic Capacity (Table VI - 5 trials per length)
    print("\n================ TABLE VI: STEGANOGRAPHIC CAPACITY (VOC 2007) ================")
    for msg_len_bytes in [10, 100, 1024]:
        img_counts = []
        for trial in range(5):
            np.random.seed(100 + trial)
            secret_msg = np.random.bytes(msg_len_bytes)
            stego_imgs, _, _, _ = stego.hide_message(secret_msg)
            if stego_imgs is not None:
                img_counts.append(len(stego_imgs))
        paper_val = PAPER_TABLE_VI_VOC2007.get(msg_len_bytes, 0.0)
        if img_counts:
            print(f"Message Length {msg_len_bytes:4d} B | Our Avg Images: {np.mean(img_counts):6.1f} | Paper Table VI: {paper_val:6.1f}")
        else:
            print(f"Message Length {msg_len_bytes:4d} B | Index Incomplete (-)")

    # 3. Experiment 2: Robustness across all 22 Attacks (Table X - 25 trials per attack)
    print("\n================ TABLE X: ROBUSTNESS EVALUATION (VOC 2007) ================")
    print(f"{'Attack Name':25s} | {'Our Baseline ACC':16s} | {'Paper Table X':14s} | {'Cluster Retrieval'}")
    print("-" * 78)
    attacks = ImageAttacks.get_all_table_ix_attacks()
    results = []

    # Pre-sample 25 messages so all attacks are evaluated on the exact same test set
    trial_messages = []
    for trial in range(num_robustness_trials):
        np.random.seed(200 + trial)
        orig_segs = stego.generate_valid_test_message(num_segments=16)
        stego_paths, aux_info = stego.hide_segments(orig_segs)
        if stego_paths is not None:
            trial_messages.append((orig_segs, stego_paths, aux_info))

    for attack_name, attack_fn in attacks.items():
        acc_list = []
        img_retrieval_acc_list = []

        for orig_segs, stego_paths, aux_info in trial_messages:
            attacked_imgs_bgr = []
            for p in stego_paths:
                img_bgr = stego._get_orig_bgr(p)
                attacked_imgs_bgr.append(attack_fn(img_bgr))

            rec_segs, retrieved_paths = stego.recover_message(attacked_imgs_bgr, aux_info)

            acc = compute_segment_accuracy(orig_segs, rec_segs)
            acc_list.append(acc)

            ret_hits = sum(1 for orig_p, ret_p in zip(stego_paths, retrieved_paths) if orig_p == ret_p)
            img_retrieval_acc_list.append((ret_hits / len(stego_paths)) * 100.0)

        mean_acc = float(np.mean(acc_list)) if acc_list else 0.0
        mean_ret = float(np.mean(img_retrieval_acc_list)) if img_retrieval_acc_list else 0.0
        paper_acc = PAPER_TABLE_X_VOC2007.get(attack_name, 0.0)

        results.append({
            "Attack": attack_name,
            "Our_Baseline_ACC (%)": round(mean_acc, 2),
            "Paper_Table_X (%)": paper_acc,
            "Cluster_Retrieval_Acc (%)": round(mean_ret, 2)
        })
        print(f"{attack_name:25s} | {mean_acc:14.2f}% | {paper_acc:12.2f}% | {mean_ret:10.2f}%")

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"cache/baseline_robustness_{dataset_name}.csv", index=False)
    return df_res

if __name__ == "__main__":
    voc_paths = download_or_get_voc2007("./data")
    run_baseline_experiments(voc_paths, dataset_name="VOC2007", num_robustness_trials=25)