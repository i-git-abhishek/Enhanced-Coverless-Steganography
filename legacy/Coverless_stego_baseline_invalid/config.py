import torch

class Config:
    # ==================== EXPERIMENT CONTROL PANEL ====================
    EXP_ID = "Exp01_Baseline"
    AUTHOR = "Abhishek"
    NOTES  = "Reproduced Lin & Liang (2026) baseline"

    # 1. Architecture: "RCNN", "RCNN_LSTM", "RCNN_GRU", "YOLO_TRANSFORMER"
    ARCH_MODE = "RCNN"

    # 2. Receiver Alignment: "Baseline_SSIM", "TRA_PhaseCorr", "Cosine_ResNet"
    RETRIEVAL_MODE = "Baseline_SSIM"

    # 3. Object Selection: False (Standard Area) or True (Multi-shift Stability Score)
    STABILITY_FILTER = False

    # 4. Auxiliary Data Mode: "Aux_Permutation", "Strict_X_Center_ModeA", "Mask_4Bit_ModeB"
    ORDER_MODE = "Aux_Permutation"

    # 5. Payload Preprocessing: "NC_NE", "Only_C", "Only_E", "Both_C_and_E"
    PAYLOAD_MODE = "NC_NE"

    # 6. Clustering Seed: False (Standard K-Means++) or True (Deterministic Sorted-Hash C0)
    DET_KMEANS = False

    # Number of robustness trials per attack
    NUM_TRIALS = 25
    # ==================================================================

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Steganography Parameters (Section III & IV-C)
    N_BITS = 6                 # Secret segment length n
    M_OBJECTS = 4              # Top-M largest objects per image
    CONF_THRESHOLD = 0.90      # 90% confidence threshold for Faster R-CNN (Section III-A2)
    
    # Clustering Parameters (Table V)
    FEATURE_DIM = 2048         # ResNet-101 output feature length
    HASH_THRESHOLD = 0.5       # Binarization threshold in Eq. (1)
    K_CLUSTERS_VOC = 256       # k=256 for PASCAL VOC 2007 / 2012
    K_CLUSTERS_COCO = 1024     # k=1024 for MS COCO 2014
    KMEANS_MAX_ITER = 100      # Max iterations for K-means++
    
    # Standard 80 COCO Recognizable Object Labels (Alphabetical for Section III-A1)
    COCO_LABELS = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]