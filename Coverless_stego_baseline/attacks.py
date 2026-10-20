import cv2
import numpy as np

class ImageAttacks:
    """Implements all geometric and non-geometric attacks from Table IX."""

    @staticmethod
    def center_cropping(img: np.ndarray, ratio: float) -> np.ndarray:
        """Masks out a centered square/rectangle of area `ratio` (5%, 10%, 20%)."""
        out = img.copy()
        h, w = out.shape[:2]
        ch, cw = int(h * np.sqrt(ratio)), int(w * np.sqrt(ratio))
        y1, x1 = (h - ch) // 2, (w - cw) // 2
        out[y1:y1+ch, x1:x1+cw] = 0
        return out

    @staticmethod
    def edge_cropping(img: np.ndarray, ratio: float) -> np.ndarray:
        """Crops/zeros out image borders totaling `ratio` (5%, 10%, 20%)."""
        out = img.copy()
        h, w = out.shape[:2]
        keep_ratio = np.sqrt(1.0 - ratio)
        kh, kw = int(h * keep_ratio), int(w * keep_ratio)
        y1, x1 = (h - kh) // 2, (w - kw) // 2
        mask = np.zeros_like(out)
        mask[y1:y1+kh, x1:x1+kw] = out[y1:y1+kh, x1:x1+kw]
        return mask

    @staticmethod
    def rotation(img: np.ndarray, angle: float) -> np.ndarray:
        """Rotates image by `angle` degrees (10, 30, 50)."""
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

    @staticmethod
    def translation(img: np.ndarray, shift_xy: tuple) -> np.ndarray:
        """Shifts image by (dx, dy) pixels: (36, 20), (40, 25), (80, 50)."""
        dx, dy = shift_xy
        h, w = img.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

    @staticmethod
    def scaling(img: np.ndarray, scale: float = 3.0) -> np.ndarray:
        """Scales image by 3x (Table IX)."""
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def gauss_noise(img: np.ndarray, mean=0.0, std=0.001) -> np.ndarray:
        norm = img.astype(np.float32) / 255.0
        noise = np.random.normal(mean, std, norm.shape).astype(np.float32)
        return np.clip((norm + noise) * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def salt_and_pepper_noise(img: np.ndarray, density=0.001) -> np.ndarray:
        out = img.copy()
        num_pixels = int(density * img.shape[0] * img.shape[1])
        # Salt
        ys = np.random.randint(0, img.shape[0], num_pixels // 2)
        xs = np.random.randint(0, img.shape[1], num_pixels // 2)
        out[ys, xs] = 255
        # Pepper
        yp = np.random.randint(0, img.shape[0], num_pixels // 2)
        xp = np.random.randint(0, img.shape[1], num_pixels // 2)
        out[yp, xp] = 0
        return out

    @staticmethod
    def speckle_noise(img: np.ndarray, mean=0.0, std=0.01) -> np.ndarray:
        norm = img.astype(np.float32) / 255.0
        noise = np.random.normal(mean, std, norm.shape).astype(np.float32)
        return np.clip((norm + norm * noise) * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def median_filter(img: np.ndarray, ksize=3) -> np.ndarray:
        return cv2.medianBlur(img, ksize)

    @staticmethod
    def mean_filter(img: np.ndarray, ksize=3) -> np.ndarray:
        return cv2.blur(img, (ksize, ksize))

    @staticmethod
    def gauss_filter(img: np.ndarray, ksize=3) -> np.ndarray:
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    @staticmethod
    def jpeg_compression(img: np.ndarray, quality=90) -> np.ndarray:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    @staticmethod
    def histogram_equalization(img: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def gamma_correction(img: np.ndarray, gamma=0.8) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    @classmethod
    def get_all_table_ix_attacks(cls):
        """Returns dictionary of all 22 attack configurations from Table IX."""
        return {
            "Center Cropping 5%": lambda x: cls.center_cropping(x, 0.05),
            "Center Cropping 10%": lambda x: cls.center_cropping(x, 0.10),
            "Center Cropping 20%": lambda x: cls.center_cropping(x, 0.20),
            "Edge Cropping 5%": lambda x: cls.edge_cropping(x, 0.05),
            "Edge Cropping 10%": lambda x: cls.edge_cropping(x, 0.10),
            "Edge Cropping 20%": lambda x: cls.edge_cropping(x, 0.20),
            "Rotation 10°": lambda x: cls.rotation(x, 10),
            "Rotation 30°": lambda x: cls.rotation(x, 30),
            "Rotation 50°": lambda x: cls.rotation(x, 50),
            "Translation (36, 20)": lambda x: cls.translation(x, (36, 20)),
            "Translation (40, 25)": lambda x: cls.translation(x, (40, 25)),
            "Translation (80, 50)": lambda x: cls.translation(x, (80, 50)),
            "Scaling": lambda x: cls.scaling(x, 3.0),
            "Gauss Noise": lambda x: cls.gauss_noise(x, 0.0, 0.001),
            "Salt & Pepper Noise": lambda x: cls.salt_and_pepper_noise(x, 0.001),
            "Speckle Noise": lambda x: cls.speckle_noise(x, 0.0, 0.01),
            "Median Filter": lambda x: cls.median_filter(x, 3),
            "Mean Filter": lambda x: cls.mean_filter(x, 3),
            "Gauss Filter": lambda x: cls.gauss_filter(x, 3),
            "JPEG Compression": lambda x: cls.jpeg_compression(x, 90),
            "Histogram Equalization": lambda x: cls.histogram_equalization(x),
            "Gamma Correction": lambda x: cls.gamma_correction(x, 0.8),
        }