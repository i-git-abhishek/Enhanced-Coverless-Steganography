import os
import json
import shutil
import subprocess
from datetime import datetime
import pandas as pd
from config import Config

class ExperimentTracker:
    def __init__(self, exp_dir="experiments"):
        self.exp_dir = exp_dir
        self.runs_dir = os.path.join(exp_dir, "runs")
        self.master_csv = os.path.join(exp_dir, "master_ablation_log.csv")
        self.dashboard_md = os.path.join(exp_dir, "RESULTS_DASHBOARD.md")
        os.makedirs(self.runs_dir, exist_ok=True)

    @staticmethod
    def _get_git_commit():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
        except Exception:
            return "uncommitted"

    def log_experiment(
        self,
        capacity_dict: dict,
        robustness_df: pd.DataFrame,
        security_ratio: float,
        dataset_utilization: float
    ):
        exp_id = Config.EXP_ID
        this_run_dir = os.path.join(self.runs_dir, exp_id)
        os.makedirs(this_run_dir, exist_ok=True)

        # 1. Snapshot the exact Python files used for this run
        for fname in ["config.py", "preprocessing.py", "stego_system.py", "evaluate.py"]:
            if os.path.exists(fname):
                shutil.copy2(fname, os.path.join(this_run_dir, fname))

        # 2. Save uncommitted line-by-line code changes (git diff)
        try:
            diff_txt = subprocess.check_output(["git", "diff"], stderr=subprocess.DEVNULL).decode("utf-8")
            with open(os.path.join(this_run_dir, "code_changes.diff"), "w") as f:
                f.write(diff_txt if diff_txt else "Clean working tree (matches git commit).")
        except Exception:
            pass

        # 3. Save detailed 22-attack CSV for this run
        run_csv_path = os.path.join(this_run_dir, "attacks.csv")
        robustness_df.to_csv(run_csv_path, index=False)

        # 4. Compute category averages
        geo_keywords = ("Cropping", "Rotation", "Translation")
        is_geo = robustness_df["Attack"].apply(lambda x: any(k in x for k in geo_keywords))
        geo_avg_acc = float(robustness_df.loc[is_geo, "Our_ACC (%)"].mean())
        non_geo_avg_acc = float(robustness_df.loc[~is_geo, "Our_ACC (%)"].mean())
        overall_avg_acc = float(robustness_df["Our_ACC (%)"].mean())
        overall_ret_acc = float(robustness_df["Cluster_Retrieval_Acc (%)"].mean())

        trans_avg_acc = float(robustness_df[robustness_df["Attack"].str.contains("Translation")]["Our_ACC (%)"].mean())
        rot_avg_acc = float(robustness_df[robustness_df["Attack"].str.contains("Rotation")]["Our_ACC (%)"].mean())

        meta_record = {
            "Exp_ID": exp_id,
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Author": Config.AUTHOR,
            "Git_Commit": self._get_git_commit(),
            "Arch": Config.ARCH_MODE,
            "Retrieval": Config.RETRIEVAL_MODE,
            "Stability": Config.STABILITY_FILTER,
            "Order_Mode": Config.ORDER_MODE,
            "Payload": Config.PAYLOAD_MODE,
            "Det_KMeans": Config.DET_KMEANS,
            "Cap_10B": round(capacity_dict.get(10, -1.0), 1),
            "Cap_100B": round(capacity_dict.get(100, -1.0), 1),
            "Cap_1024B": round(capacity_dict.get(1024, -1.0), 1),
            "Ratio_s": round(security_ratio, 3),
            "Util_Rate (%)": round(dataset_utilization, 2),
            "Trans_Avg (%)": round(trans_avg_acc, 2),
            "Rot_Avg (%)": round(rot_avg_acc, 2),
            "Geo_Avg (%)": round(geo_avg_acc, 2),
            "NonGeo_Avg (%)": round(non_geo_avg_acc, 2),
            "Overall_ACC (%)": round(overall_avg_acc, 2),
            "Cluster_Ret (%)": round(overall_ret_acc, 2),
            "Notes": Config.NOTES
        }

        with open(os.path.join(this_run_dir, "meta.json"), "w") as f:
            json.dump(meta_record, f, indent=4)

        # 5. Update Master CSV Log
        if os.path.exists(self.master_csv) and os.path.getsize(self.master_csv) > 0:
            master_df = pd.read_csv(self.master_csv)
            master_df = master_df[master_df["Exp_ID"] != exp_id]
            master_df = pd.concat([master_df, pd.DataFrame([meta_record])], ignore_index=True)
        else:
            master_df = pd.DataFrame([meta_record])

        master_df.to_csv(self.master_csv, index=False)

        # 6. Rebuild RESULTS_DASHBOARD.md
        self._build_markdown_dashboard(master_df)
        print(f"\n[+] Saved code snapshot & attack table to: {this_run_dir}/")
        print(f"[+] Updated Master Log: {self.master_csv}")
        print(f"[+] Updated Dashboard:  {self.dashboard_md}")

    def _build_markdown_dashboard(self, master_df: pd.DataFrame):
        lines = [
            "# Group 21 — Coverless Steganography Experiment Dashboard",
            f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
            "## 1. High-Level Ablation Summary",
            ""
        ]

        summary_cols = [
            "Exp_ID", "Author", "Arch", "Retrieval", "Stability",
            "Order_Mode", "Payload", "Cap_100B", "Ratio_s",
            "Trans_Avg (%)", "Rot_Avg (%)", "Geo_Avg (%)", "Overall_ACC (%)", "Notes"
        ]
        avail_cols = [c for c in summary_cols if c in master_df.columns]
        lines.append(master_df[avail_cols].to_markdown(index=False))
        lines.append("\n---\n")

        lines.append("## 2. Full 22-Attack Robustness Matrix (Table X Comparison)\n")
        merged_attacks = None

        for exp_id in master_df["Exp_ID"]:
            run_csv = os.path.join(self.runs_dir, exp_id, "attacks.csv")
            if not os.path.exists(run_csv):
                continue
            df_run = pd.read_csv(run_csv)
            if merged_attacks is None:
                merged_attacks = df_run[["Attack", "Paper_Table_X (%)"]].copy()
            merged_attacks[f"{exp_id} ACC (%)"] = df_run["Our_ACC (%)"]
            merged_attacks[f"{exp_id} Ret (%)"] = df_run["Cluster_Retrieval_Acc (%)"]

        if merged_attacks is not None:
            lines.append(merged_attacks.to_markdown(index=False))

        with open(self.dashboard_md, "w") as f:
            f.write("\n".join(lines) + "\n")