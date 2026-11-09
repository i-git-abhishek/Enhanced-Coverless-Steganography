# Baseline manual

This manual takes you from the zip file to saved baseline results on Kaggle. It also shows how to change n and M, how to run other datasets, and how to change code safely.

**Follow sections 3 to 6 in order for your first run.** Kaggle's menu names can change. If a button looks different, look for the closest match.

## Contents

1. What "recreating the baseline" means
2. What is in the repo
3. Put the code in GitHub (once)
4. Set up the Kaggle notebook (once)
5. Run the VOC 2007 baseline
6. Save the results to GitHub
7. Run n = 3, 4, 5, 6 and M = 3, 4, 5, 6
8. Run VOC 2012 and COCO 2014
9. Reuse the cache and save GPU time
10. Change the code safely
11. Read the results
12. Where the paper is unclear
13. Fix common problems
14. Command reference

---

## 1. What "recreating the baseline" means

A successful run means two things:

1. **Every automatic check passes.** The code then matches the method as described and its results are traceable to a commit.
2. **The results are compared honestly with the paper.** The dashboard shows both sets of numbers and the difference.

**Do not expect an exact match, and never change the code just to get closer to a paper number.** The paper leaves out some details, such as its trained detector and the SSIM image size. A remaining gap is a finding to report (section 12).

Exp01 is invalid and must not be reported. See `legacy/README.md`.

---

## 2. What is in the repo

| Path | What it does |
| --- | --- |
| `run.py` | Starts everything |
| `configs/` | Default settings per dataset |
| `stego/mapping.py` | Dictionary, top-M objects, hash, clusters, index (paper III-A) |
| `stego/hiding.py` | Message segments, image selection, auxiliary data (III-B) |
| `stego/recovery.py` | Eq. 2 retrieval and decoding (III-C) |
| `stego/attacks.py` | Table IX attacks and the no-attack control |
| `stego/evaluate.py` | Capacity and robustness experiments |
| `stego/extract.py` | The GPU step: Faster R-CNN and ResNet-101 |
| `stego/paper_refs.py` | Paper values used for comparison only |
| `stego/tracking.py` | Run folders and dashboard |
| `tests/` | Automated tests, including the paper's Examples 2 to 5 |
| `notebooks/kaggle_baseline.ipynb` | The notebook you run on Kaggle |
| `experiments/runs/` | Saved results. These are committed to Git |

---

## 3. Put the code in GitHub (once)

You need Git installed on your computer. Run these in a terminal. They work in Windows PowerShell, macOS and Linux.

**Step 1. Get the repo and go into it.** Skip the `git clone` line if you already have it, and make sure your work is committed.

```bash
git clone https://github.com/i-git-abhishek/Enhanced-Coverless-Steganography.git
cd Enhanced-Coverless-Steganography
git status
```

`git status` should say the working tree is clean.

**Step 2. Mark the old state and create the new branch.**

```bash
git tag exp01-invalid
git push origin exp01-invalid
git checkout -b rewrite-baseline
```

**Step 3. Move the old code into `legacy/`.**

```bash
mkdir legacy
git mv Coverless_stego_baseline legacy/Coverless_stego_baseline_invalid
```

**Step 4. Extract the zip.** Extract `Enhanced-Coverless-Steganography-rewrite.zip` into this same folder, and allow it to replace `README.md`, `requirements.txt` and `.gitignore`. `run.py` should now sit directly in the repo's main folder.

**Step 5. Commit and push.**

```bash
git add -A
git status
git commit -m "Rewrite baseline to follow the paper and add manual"
git push -u origin rewrite-baseline
```

Before committing, check that `git status` shows no dataset images or large files. If one appears, run `git restore --staged PATH`, then add that path to `.gitignore`. Your repo now has a `rewrite-baseline` branch, and `main` is unchanged.

**Optional local test (no GPU needed).** This install is also needed if you ever run `python run.py dashboard` on your computer:

```bash
python -m venv venv
# Windows PowerShell, if activation is blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
# Windows: venv\Scripts\activate
# macOS or Linux: source venv/bin/activate
pip install -r requirements.txt
python run.py smoke
python -m pytest tests -q
```

`python run.py smoke` must end with `SMOKE TEST PASSED`, and all tests must pass. The tests can take a few minutes.

---

## 4. Set up the Kaggle notebook (once)

**Step 1. Create the notebook.** Sign in at kaggle.com, then choose **Create → New Notebook**. Choose **File → Import Notebook** and upload `notebooks/kaggle_baseline.ipynb`.

**Step 2. Set the session options** in the right sidebar:

- **Accelerator:** GPU T4 x2. Avoid P100, because recent PyTorch versions may not support it.
- **Internet:** On. It is needed for the code and model weights. Kaggle may first ask you to verify your phone number.

**Step 3. Add the dataset.** Choose **Add Input** and search for `zaraks/pascal-voc-2007` ([page](https://www.kaggle.com/datasets/zaraks/pascal-voc-2007)). The notebook checks that all 9,963 images are present. If that dataset is incomplete or unavailable, try another VOC 2007 dataset that includes both trainval and test.

**Step 4. Private repo only.** Your repo is currently public, so skip this step unless you make it private.

1. On GitHub, go to **Settings → Developer settings → Personal access tokens → Fine-grained tokens** and create a token. Limit it to this repo with **Contents: Read-only**.
2. In the Kaggle notebook, choose **Add-ons → Secrets**. Add a secret named exactly `GITHUB_TOKEN`, paste the token, and attach it to this notebook.

Never paste a token directly into a notebook cell.

---

## 5. Run the VOC 2007 baseline

**Step 1. Check the settings cell.** It should read:

```python
REPO = "i-git-abhishek/Enhanced-Coverless-Steganography"
BRANCH = "rewrite-baseline"
CONFIG = "configs/voc2007.yaml"
RUN_EXTRACTION = True
RUN_BASELINE = True
SWEEP_N = []
SWEEP_M = []
```

**Step 2. Start the run.** Choose **Save Version → Save & Run All (Commit)**. You can then close the browser, because the run continues on Kaggle.

My rough, untimed estimate is 30 to 60 minutes in total. The longest part is the detector pass over all 9,963 images.

**Step 3. Check the log.** Open the finished version and read the log. You should see these lines in order:

| Log line | Meaning | If it is missing or wrong |
| --- | --- | --- |
| `SMOKE TEST PASSED` | The code works on made-up data | Section 13 |
| `PASS: found all 9963 images` | The dataset is complete | Section 13 |
| `[extract] device: cuda` | The GPU is on | Turn on the GPU in step 2 of section 4 |
| `[preview] ...` lines with object names | The detector works | Section 13 |
| `"agreement_percent"` near 100 | Stored detections match a new pass | Section 11 |
| `[check] PASS` on every check | The system is correct | Section 13 |
| `[run] Status: OFFICIAL` | The run can be reported | Section 13 |

---

## 6. Save the results to GitHub

Kaggle deletes session files, so do this after every run you want to keep.

**Step 1.** In the finished notebook version, open the **Output** tab and download `results_bundle.tar.gz`.

**Step 2.** Put the file in your repo's main folder on your computer, then run:

```bash
git checkout rewrite-baseline
git pull
tar -xzf results_bundle.tar.gz
git add experiments
git commit -m "Add VOC 2007 baseline results"
git push
```

`tar` is included in Windows 10 and later. The file itself is ignored by Git, so you can delete it afterwards.

**Step 3.** Open `experiments/RESULTS_DASHBOARD.md` on GitHub to see all results.

If you and a teammate both add results and Git reports a conflict in `RESULTS_DASHBOARD.md`, keep the run folders from both sides and rebuild it:

```bash
python run.py dashboard
git add experiments
git commit -m "Rebuild dashboard"
```

Do not delete or edit a run folder. To repeat a run, start another one. Each run gets its own folder automatically.

---

## 7. Run n = 3, 4, 5, 6 and M = 3, 4, 5, 6

Change only the settings cell, then choose **Save & Run All** again.

**n sweep, keeping M at its config value of 4:**

```python
RUN_BASELINE = False
SWEEP_N = [3, 4, 5, 6]
SWEEP_M = []
```

**M sweep, keeping n at its config value of 6:**

```python
RUN_BASELINE = False
SWEEP_N = []
SWEEP_M = [3, 4, 5, 6]
```

You can fill both lists in one run. Each value creates a separate run folder. Afterwards, follow section 6 to save the results.

**One specific setting,** such as n = 4 only:

```python
RUN_BASELINE = False
SWEEP_N = [4]
```

**Command-line equivalents:**

```bash
python run.py sweep --config configs/voc2007.yaml --param n --values 3 4 5 6
python run.py sweep --config configs/voc2007.yaml --param M --values 3 4 5 6
python run.py baseline --config configs/voc2007.yaml --set n=4 --set M=5
```

**Which settings have paper values:**

| Dataset | n | M | Paper tables |
| --- | --- | --- | --- |
| VOC 2007 | 6 | 4 | VI, X |
| VOC 2012 | 6 | 4 | VII, XI |
| COCO 2014 | 6 | 4 | VIII, XII |
| COCO 2014 | 3, 4, 5 | 4 | XIII, XIV, XV |
| COCO 2014 | 6 | 3, 5, 6 | XVI, XVII |

**The paper ran its n and M sweeps on COCO 2014 only.** A sweep on VOC is valid and useful, and it runs much faster, but the dashboard will show `-` for paper values because none exist.

n can only go as high as the labels allow: there must be at least 2^n detected labels. n = 7 stops with a clear error because the detector has only 80 labels. This limit applies at n = 6 too. If a dataset has fewer than 64 detected labels, n = 6 will stop with the same error.

---

## 8. Run VOC 2012 and COCO 2014

**VOC 2012 (11,540 images):**

1. Add a VOC 2012 dataset, such as `huanghanchina/pascal-voc-2012` ([page](https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012)). It must include `VOC2012/ImageSets/Main/trainval.txt`.
2. Set `CONFIG = "configs/voc2012.yaml"`.
3. Run, then save the results as in section 6.

**COCO 2014 (164,062 images):** Do this last.

1. Add a COCO 2014 dataset with `train2014`, `val2014` and `test2014` folders.
2. Set `CONFIG = "configs/coco2014.yaml"`.
3. Check Kaggle's current session limit first. The detector pass will likely take several hours. A new notebook version starts empty and only reuses a *finished* cache, so a detector pass cut off by the limit starts again from zero.

**If a COCO dataset has no test images,** the count check stops the run. You can continue using train and val only by adding `expected_images: 123287` to the config, but this must be reported as a deviation. Commit and push that config change first, then run the notebook.

`configs/coco2014.yaml` uses MiniBatch k-means because exact k-means may not finish in one session at this size. This is a deliberate, reported deviation (section 12).

---

## 9. Reuse the cache and save GPU time

The detector results can be reused for every n and M. A sweep within one notebook run already shares them. To reuse them in later notebook versions:

1. In a finished notebook version, open the **Output** tab and create a new private dataset from the output.
2. Attach that dataset to the notebook with **Add Input**.
3. Keep the same image dataset attached as well.

The code finds a finished cache automatically under `/kaggle/input`. `extract` then says `Finished extraction already exists`.

To run from the cache without a GPU, set `RUN_EXTRACTION = False` and turn the accelerator off. Keep Internet on so that the code can be downloaded.

Each cache key includes the image list and the model names. If you change what `stego/extract.py` computes, raise `FORMAT_VERSION` in `stego/cache.py` so that an old cache isn't reused. `conf_threshold` can be changed without another GPU pass, because detections above 0.05 are stored, up to the detector's limit of 100 per image.

---

## 10. Change the code safely

**To change only a setting,** use section 7, or use `--set`:

```bash
python run.py baseline --config configs/voc2007.yaml --set ssim_mode=windowed
```

Available settings are the fields in `stego/config.py`. An unknown name stops with an error, so a run can't carry a label that did nothing.

**To change code or add an experiment:**

1. Create a branch: `git checkout -b exp-short-name`
2. Make the change. A new option must go in `stego/config.py` and must actually change behaviour.
3. Add a test in `tests/`.
4. Run `python run.py smoke` and `python -m pytest tests -q`.
5. Commit and push the branch.
6. Set `BRANCH = "exp-short-name"` in the notebook and run it.
7. Save the results as in section 6, but check out your experiment branch instead of `rewrite-baseline`.

**Rules that keep results valid:**

- Never add a constant chosen to reach a paper value.
- Never edit `paper_refs.py` to fit a result. Only correct it if it differs from the paper.
- Never edit `summary.json`, `attacks.csv` or the dashboard by hand.
- Compare one change at a time against the official baseline, using the same seed.
- Report all runs that test a hypothesis, including those that did not help.
- Uncommitted code is refused. `--allow-dirty` is for quick tests only, and those runs are marked NOT official.

---

## 11. Read the results

Each run folder contains:

| File | Contents |
| --- | --- |
| `summary.json` | Main numbers and official status |
| `attacks.csv` | Per-attack accuracy, 95% interval, retrieval accuracy, paper value, difference |
| `capacity.csv` | Per message: images, bits per image, label use and auxiliary bits per stego image |
| `segments.csv.gz` | Raw result for every segment and attack |
| `dictionary.csv` | The label-to-bits dictionary used |
| `checks.json` | Each check and its result |
| `config.json`, `environment.json`, `source.tar.gz` | Exact settings, versions and source code |

**Checks. All must pass for a run to be official:**

| Check | Meaning |
| --- | --- |
| `dictionary_complete` | All 2^n groups are used, and each label has one group |
| `dictionary_alphabetical` | Groups follow alphabetical label order, as in Fig. 5 |
| `index_complete` | All capacity messages were fully hidden |
| `bytes_round_trip` | Decoding from the index returns the exact original bytes |
| `bytes_round_trip_with_retrieval` | The same is true when going through real retrieval |
| `all_attacks_present` | All 22 attacks and the control ran |
| `no_attack_acc_100` | Accuracy is 100% with no attack |
| `no_attack_retrieval_100` | Retrieval is 100% with no attack |
| `image_count` | All images were used |

**`verify-detections` agreement.** The paper re-runs the detector on the retrieved image, while this code reads stored results for speed. These are equivalent only if the agreement is 100%. If it is lower, keep the output, report it, and tell your supervisor, because it may help explain the paper's Table XVII pattern.

**How to read the numbers:**

- **ACC and retrieval:** ACC should be close to retrieval, or somewhat above it when neighbouring images share labels. ACC far below retrieval would indicate a bug.
- **Capacity:** This code searches every image, so it may need fewer images than the paper. Report the difference. Do not limit the search to match.
- **M trend:** The paper's accuracy falls sharply from M = 3 to M = 6, even under weak attacks. Retrieval does not depend on M, so this code may not show the same fall. That result would be important, so report it.
- **Intervals:** With 25 messages, differences that fit within the 95% intervals should not be claimed as real.
- **Auxiliary bits:** `aux_bits_tight` and `aux_bits_fixed` exclude padding and the marker, so they are minimums.

---

## 12. Where the paper is unclear

Report this table in your methods section. Where a setting exists, you can test the alternative.

| Topic | Paper | This code | Setting |
| --- | --- | --- | --- |
| Detector | "Well-trained," Table IV lists training values, no training data named | Pretrained torchvision Faster R-CNN R50-FPN, COCO_V1 | None |
| Dictionary balance | Evenly distribute images per group, grouped alphabetically | Neighbouring labels only, minimising the sum of squared image counts | None |
| Image counts for balance | Images available for each segment, not exact | Images where the label is among the top-M objects, so the dictionary can differ between M values. Labels never in the top M are left out | `conf_threshold`, `M` |
| ResNet-101 input | Not stated | IMAGENET1K_V1, likely the weights available in the paper's Python 3.6 setup (my inference). Resized to 224×224, avgpool output | None |
| Min-max normalisation | Not stated per image or per dataset | Per image | None |
| Clustering | Text says k-means++, Table V describes averaging random groups | scikit-learn k-means++, one initialisation, 100 iterations | `cluster_algorithm`, `cluster_seed` |
| Clustered images | "Each image in the dataset" | All images, including those with no objects. These can be retrieved but not selected | None |
| Eq. 2 | Global formula shown, cites Wang et al. | One global window on grayscale, C3 = C2/2, exponents 1 | `ssim_mode` |
| SSIM size | Not stated | 256×256 | `ssim_size` |
| Stego selection | "Sequentially searches," prefers most segments | Searches all images. Breaks ties at random, preferring unused images | None |
| Receiver detector | Re-run on retrieved image | Stored results, checked by `verify-detections` | None |
| Robustness message | 25 tests, length not stated | 16 random segments | `robustness_segments`, `robustness_trials` |
| 1 KB | Text says 1000 B, tables imply 1024 B | 1024 B | `capacity_bytes` |
| Rotation | Angle only | About the centre, same image size, no scaling to fit, black fill | None |
| Scaling | "3 times" | Bilinear enlargement. The receiver resizes it for comparison | None |
| Noise σ | Number only | Applied to pixel values on a 0 to 1 scale | None |
| Gauss filter | Size 3×3 only | OpenCV derives sigma from the size | None |
| Cropping | Ratio only | Black fill. Ratio is the area removed | None |
| Translation | "Ratio: (36, 20)" | Pixels, black fill | None |
| Salt and pepper | Lists "σ = 0.001" | Density of 0.001 of pixels | None |
| Gamma | "Factor: 0.8" | Output = input^0.8. Use 1.25 for the opposite reading | `gamma_exponent` |
| Histogram equalization | No details | On the luminance channel | None |
| COCO clustering | Not stated | MiniBatch k-means | `cluster_algorithm` |

---

## 13. Fix common problems

| Message | What to do |
| --- | --- |
| `Clone failed` | Check `REPO` and `BRANCH`, turn Internet on, and confirm the branch is pushed. For a private repo, complete step 4 of section 4 |
| `Found N images ... expected 9963` | The dataset is incomplete, often trainval only. Add a complete one. If you see more than expected, check for extra attached data |
| `[extract] WARNING: no GPU found` | Set the accelerator to GPU T4 x2, then run again |
| Model weights fail to download | Turn Internet on |
| `No finished extraction found` | Run `extract`, or keep `RUN_EXTRACTION = True`. A cache only applies to exactly the same image list |
| `n=7 needs 128 groups` | Use a smaller n. The same error can apply at n = 6 if too few labels are detected. Report it |
| `The code has uncommitted changes` | Commit the change. On Kaggle, do not edit files in the clone |
| `Unknown setting(s)` | Check the name against `stego/config.py` |
| `no_attack_retrieval_100` fails | Filter `segments.csv.gz` to `No attack`, then find rows where `retrieval_hit` is False. Compare `stego_image` and `retrieved_image`. Identical images in the dataset can tie. If that is the cause, report it and do not change the check |
| `no_attack_acc_100` fails with retrieval at 100% | This is a bug. Stop and do not use the run |
| `bytes_round_trip` fails | This is a bug. Stop and do not use the run |
| `index_complete` fails | This should not happen, because every group has a usable image. Treat it as a bug and keep the log |
| `SMOKE TEST FAILED` | The code or environment has a problem. Keep the log, and do not continue |
| The notebook still runs old code in an interactive session | Use **Save & Run All**, which starts clean, or restart the session and delete the old clone |
| Kaggle stops at its time limit | Run fewer settings in each version, and attach the cache dataset from an earlier finished run |

---

## 14. Command reference

Every command is run from the repo's main folder.

| Command | Purpose |
| --- | --- |
| `python run.py smoke` | Test the full pipeline on made-up data |
| `python -m pytest tests -q` | Run all automated tests |
| `python run.py check-data --config configs/voc2007.yaml` | Check the dataset |
| `python run.py extract --config configs/voc2007.yaml --limit 4` | Preview the detector on 4 images and save nothing |
| `python run.py extract --config configs/voc2007.yaml` | Run or resume the full GPU step |
| `python run.py verify-detections --config configs/voc2007.yaml --sample 100` | Compare stored and new detections |
| `python run.py baseline --config configs/voc2007.yaml` | Run capacity and robustness once |
| `python run.py baseline --config configs/voc2007.yaml --set n=4 --set M=5` | Run once with other settings |
| `python run.py baseline --config configs/voc2007.yaml --set seed=1 --label repeat` | Repeat with another seed and a label in the folder name |
| `python run.py sweep --config configs/voc2007.yaml --param n --values 3 4 5 6` | Run several values of one setting |
| `python run.py dashboard` | Rebuild the dashboard |
| `python run.py bundle --output results_bundle.tar.gz` | Pack results for download |

Add `--data-root FOLDER` if the images are not in `/kaggle/input` (Kaggle) or `data/` (your computer).
