# Group 21 — Coverless Steganography Experiment Dashboard

_Last updated: 2026-11-02 15:35_

## 1. High-Level Ablation Summary

| Exp_ID         | Author   | Arch | Retrieval     | Stability | Order_Mode      | Payload | Cap_100B | Ratio_s | Trans_Avg (%) | Rot_Avg (%) | Geo_Avg (%) | Overall_ACC (%) | Notes                                  |
| :------------- | :------- | :--- | :------------ | :-------- | :-------------- | :------ | -------: | ------: | ------------: | ----------: | ----------: | --------------: | :------------------------------------- |
| Exp01_Baseline | Abhishek | RCNN | Baseline_SSIM | False     | Aux_Permutation | NC_NE   |     86.4 |     0.5 |         47.42 |        43.5 |       71.73 |           83.49 | Reproduced Lin & Liang (2026) baseline |

---

## 2. Full 22-Attack Robustness Matrix (Table X Comparison)

| Attack                 | Paper_Table_X (%) | Exp01_Baseline ACC (%) | Exp01_Baseline Ret (%) |
| :--------------------- | ----------------: | ---------------------: | ---------------------: |
| Center Cropping 5%     |             96.61 |                   99.5 |                    100 |
| Center Cropping 10%    |             96.67 |                  98.25 |                    100 |
| Center Cropping 20%    |             95.55 |                  97.25 |                    100 |
| Edge Cropping 5%       |             97.16 |                  99.75 |                    100 |
| Edge Cropping 10%      |             95.93 |                     99 |                   99.6 |
| Edge Cropping 20%      |             84.79 |                  94.25 |                  96.27 |
| Rotation 10°           |             85.31 |                     66 |                   99.2 |
| Rotation 30°           |             66.04 |                  39.75 |                  30.28 |
| Rotation 50°           |             41.58 |                  24.75 |                   10.8 |
| Translation (36, 20)   |             72.29 |                     67 |                  62.65 |
| Translation (40, 25)   |              62.8 |                  55.75 |                   46.8 |
| Translation (80, 50)   |             59.35 |                   19.5 |                   7.78 |
| Scaling                |             97.75 |                    100 |                    100 |
| Gauss Noise            |             97.95 |                    100 |                    100 |
| Salt & Pepper Noise    |             96.49 |                    100 |                    100 |
| Speckle Noise          |             97.55 |                    100 |                    100 |
| Median Filter          |             93.83 |                  96.25 |                    100 |
| Mean Filter            |             93.52 |                  91.75 |                    100 |
| Gauss Filter           |             95.34 |                   94.5 |                    100 |
| JPEG Compression       |             93.69 |                    100 |                    100 |
| Histogram Equalization |             97.43 |                  93.75 |                  94.89 |
| Gamma Correction       |             97.89 |                  99.75 |                    100 |
