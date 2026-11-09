import os
import glob
import numpy as np
import pandas as pd
from torchvision.datasets import VOCDetection
from preprocessing import DataPreprocessor
from stego_system import BaselineCoverlessStego
from attacks import ImageAttacks
from config import Config
from experiment_tracker import ExperimentTracker

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
    m = len(orig_segments)
    if m == 0:
        return 0.0
    matches = sum(1 for o, r in zip(orig_segments, rec_segments) if o == r)
    return (matches / m) * 100.0

def compute_bit_accuracy(orig_segments: list, rec_segments: list) -> float:
    orig_bits = "".join(orig_segments)
    rec_bits = "".join(rec_segments)
    if not orig_bits:
        return 0.0
    matches = sum(1 for b1, b2 in zip(orig_bits, rec_bits) if b1 == b2)
    return (matches / len(orig_bits)) * 100.0

def compute_security_ratio(ordering_mode: str, n_bits: int = 6, m_objects: int = 4) -> float:
    max_bits_per_img = n_bits * m_objects
    if ordering_mode == "Strict_X_Center_ModeA":
        aux_order_bits = 0
    elif ordering_mode == "Mask_4Bit_ModeB":
        aux_order_bits = 4
    else:
        aux_order_bits = 12
    return aux_order_bits / float(max_bits_per_img)

def download_or_get_voc2007(root_dir="./data"):
    voc_img_dir = os.path.join(root_dir, "VOCdevkit", "VOC2007", "JPEGImages")
    if not os.path.exists(voc_img_dir):
        print("[*] Downloading PASCAL VOC 2007 dataset via torchvision...")
        VOCDetection(root=root_dir, year="2007", image_set="trainval", download=True)
        VOCDetection(root=root_dir, year="2007", image_set="test", download=True)
    img_paths = sorted(glob.glob(os.path.join(voc_img_dir, "*.jpg")))
    print(f"[+] Found {len(img_paths)} images in VOC 2007.")
    return img_paths

def run_experiments():
    exp_id = Config.EXP_ID
    print(f"[*] Starting Experiment: {exp_id} (Author: {Config.AUTHOR})")
    print(f"[*] Modes -> Arch: {Config.ARCH_MODE} | Retrieval: {Config.RETRIEVAL_MODE} | Order: {Config.ORDER_MODE} | Payload: {Config.PAYLOAD_MODE}")

    image_paths = download_or_get_voc2007("./data")
    preprocessor = DataPreprocessor(dataset_name="VOC2007")
    raw_data = preprocessor.extract_raw_dataset_info(
        image_paths, cache_path="cache/VOC2007_raw.pkl"
    )
    index_bundle = preprocessor.build_index_and_clusters(
        raw_data, n_bits=Config.N_BITS, m_objects=Config.M_OBJECTS
    )
    stego = BaselineCoverlessStego(index_bundle, n_bits=Config.N_BITS, m_objects=Config.M_OBJECTS)

    valid_count = len(index_bundle["image_to_top_objects"])
    utilization_rate = (valid_count / float(len(image_paths))) * 100.0

    # 1. Capacity Benchmark (Table VI)
    print(f"\n================ [{exp_id}] TABLE VI: CAPACITY ================")
    capacity_dict = {}
    for msg_len_bytes in [10, 100, 1024]:
        img_counts = []
        for trial in range(5):
            np.random.seed(100 + trial)
            secret_msg = np.random.bytes(msg_len_bytes)
            stego_imgs, _, _, _ = stego.hide_message(secret_msg)
            if stego_imgs is not None:
                img_counts.append(len(stego_imgs))
        paper_val = PAPER_TABLE_VI_VOC2007.get(msg_len_bytes, 0.0)
        avg_imgs = float(np.mean(img_counts)) if img_counts else -1.0
        capacity_dict[msg_len_bytes] = avg_imgs
        print(f"Message Length {msg_len_bytes:4d} B | Our Avg Images: {avg_imgs:6.1f} | Paper Table VI: {paper_val:6.1f}")

    # 2. Robustness Benchmark (Table X - 22 Attacks)
    print(f"\n================ [{exp_id}] TABLE X: ROBUSTNESS ================")
    print(f"{'Attack Name':25s} | {'Our ACC':10s} | {'Our BAR':10s} | {'Paper ACC':10s} | {'Cluster Ret'}")
    print("-" * 78)
    attacks = ImageAttacks.get_all_table_ix_attacks()
    results = []

    trial_messages = []
    for trial in range(Config.NUM_TRIALS):
        np.random.seed(200 + trial)
        orig_segs = stego.generate_valid_test_message(num_segments=16)
        stego_paths, aux_info = stego.hide_segments(orig_segs)
        if stego_paths is not None:
            trial_messages.append((orig_segs, stego_paths, aux_info))

    for attack_name, attack_fn in attacks.items():
        acc_list, bar_list, ret_list = [], [], []

        for orig_segs, stego_paths, aux_info in trial_messages:
            attacked_imgs_bgr = [attack_fn(stego._get_orig_bgr(p)) for p in stego_paths]
            rec_segs, retrieved_paths = stego.recover_message(attacked_imgs_bgr, aux_info)

            acc_list.append(compute_segment_accuracy(orig_segs, rec_segs))
            bar_list.append(compute_bit_accuracy(orig_segs, rec_segs))
            ret_hits = sum(1 for orig_p, ret_p in zip(stego_paths, retrieved_paths) if orig_p == ret_p)
            ret_list.append((ret_hits / len(stego_paths)) * 100.0)

        mean_acc = float(np.mean(acc_list))
        mean_bar = float(np.mean(bar_list))
        mean_ret = float(np.mean(ret_list))
        paper_acc = PAPER_TABLE_X_VOC2007.get(attack_name, 0.0)

        results.append({
            "Attack": attack_name,
            "Our_ACC (%)": round(mean_acc, 2),
            "Our_BAR (%)": round(mean_bar, 2),
            "Paper_Table_X (%)": paper_acc,
            "Cluster_Retrieval_Acc (%)": round(mean_ret, 2)
        })
        print(f"{attack_name:25s} | {mean_acc:8.2f}% | {mean_bar:8.2f}% | {paper_acc:8.2f}% | {mean_ret:9.2f}%")

    df_res = pd.DataFrame(results)
    ratio_s = compute_security_ratio(Config.ORDER_MODE, Config.N_BITS, Config.M_OBJECTS)

    # 3. Log Everything Automatically
    tracker = ExperimentTracker(exp_dir="experiments")
    tracker.log_experiment(
        capacity_dict=capacity_dict,
        robustness_df=df_res,
        security_ratio=ratio_s,
        dataset_utilization=utilization_rate
    )

if __name__ == "__main__":
    run_experiments()