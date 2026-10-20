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

def compute_segment_accuracy(orig_segments: list, rec_segments: list) -> float:
    """Equation (3): Segment recovery accuracy ACC."""
    m = len(orig_segments)
    if m == 0:
        return 0.0
    matches = sum(1 for o, r in zip(orig_segments, rec_segments) if o == r)
    return (matches / m) * 100.0

def download_or_get_voc2007(root_dir="./data"):
    """Automatically downloads PASCAL VOC 2007 if not present and returns image paths."""
    voc_img_dir = os.path.join(root_dir, "VOCdevkit", "VOC2007", "JPEGImages")
    if not os.path.exists(voc_img_dir):
        print("[*] Downloading PASCAL VOC 2007 dataset via torchvision...")
        VOCDetection(root=root_dir, year="2007", image_set="trainval", download=True)
        VOCDetection(root=root_dir, year="2007", image_set="test", download=True)
    img_paths = sorted(glob.glob(os.path.join(voc_img_dir, "*.jpg")))
    print(f"[+] Found {len(img_paths)} images in VOC 2007.")
    return img_paths

def run_baseline_experiments(image_paths, dataset_name="VOC2007", num_robustness_trials=5):
    # 1. Phase A: Preprocess & Index
    preprocessor = DataPreprocessor(dataset_name=dataset_name)
    raw_data = preprocessor.extract_raw_dataset_info(
        image_paths, cache_path=f"cache/{dataset_name}_raw.pkl"
    )
    index_bundle = preprocessor.build_index_and_clusters(
        raw_data, n_bits=Config.N_BITS, m_objects=Config.M_OBJECTS
    )
    stego = BaselineCoverlessStego(index_bundle, n_bits=Config.N_BITS, m_objects=Config.M_OBJECTS)

    # 2. Experiment 1: Steganographic Capacity (Table VI / VII / VIII)
    print("\n================ STEGANOGRAPHIC CAPACITY ================")
    for msg_len_bytes in [10, 100, 1024]:
        img_counts = []
        for trial in range(5):
            np.random.seed(100 + trial)
            secret_msg = np.random.bytes(msg_len_bytes)
            stego_imgs, _, _, completion = stego.hide_message(secret_msg)
            if stego_imgs is not None:
                img_counts.append(len(stego_imgs))
        if img_counts:
            print(f"Message Length {msg_len_bytes:4d} B | Avg Stego Images Required: {np.mean(img_counts):.1f}")
        else:
            print(f"Message Length {msg_len_bytes:4d} B | Index Incomplete (-)")

    # 3. Experiment 2: Robustness across all 22 Attacks (Table X / XI / XII)
    print("\n================ ROBUSTNESS EVALUATION ================")
    attacks = ImageAttacks.get_all_table_ix_attacks()
    results = []

    for attack_name, attack_fn in attacks.items():
        acc_list = []
        img_retrieval_acc_list = []

        for trial in range(num_robustness_trials):
            np.random.seed(200 + trial)
            # Generate valid message segments (Section IV-B2 protocol)
            orig_segs = stego.generate_valid_test_message(num_segments=10)
            stego_paths, aux_info = stego.hide_segments(orig_segs)
            if stego_paths is None:
                continue

            # Apply attack to each transmitted stego image
            attacked_imgs_bgr = []
            for p in stego_paths:
                img_bgr = cv2.imread(p)
                attacked_imgs_bgr.append(attack_fn(img_bgr))

            # Receiver recovery
            rec_segs, retrieved_paths = stego.recover_message(attacked_imgs_bgr, aux_info)
            
            # Metric 1: Equation (3) Segment Accuracy
            acc = compute_segment_accuracy(orig_segs, rec_segs)
            acc_list.append(acc)

            # Metric 2: Diagnostic Original-Image Retrieval Accuracy
            ret_hits = sum(1 for orig_p, ret_p in zip(stego_paths, retrieved_paths) if orig_p == ret_p)
            img_retrieval_acc_list.append((ret_hits / len(stego_paths)) * 100.0)

        mean_acc = np.mean(acc_list)
        mean_ret = np.mean(img_retrieval_acc_list)
        results.append({
            "Attack": attack_name,
            "Segment_ACC (%)": round(mean_acc, 2),
            "Orig_Image_Retrieval (%)": round(mean_ret, 2)
        })
        print(f"{attack_name:25s} | ACC: {mean_acc:6.2f}% | Image Retrieval Acc: {mean_ret:6.2f}%")

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"cache/baseline_robustness_{dataset_name}.csv", index=False)
    return df_res

if __name__ == "__main__":
    # Start with VOC 2007 (9,963 images). For a quick 5-minute smoke test, slice [:1500]
    voc_paths = download_or_get_voc2007("./data")
    run_baseline_experiments(voc_paths, dataset_name="VOC2007", num_robustness_trials=5)