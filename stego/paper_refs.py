"""Reference values copied from Lin & Liang. They are only compared with, never used to tune."""

from __future__ import annotations

from .attacks import TABLE_IX_ORDER


def _row(values: list[float]) -> dict[str, float]:
    if len(values) != len(TABLE_IX_ORDER):
        raise ValueError("A robustness row needs exactly 22 values")
    return dict(zip(TABLE_IX_ORDER, values))


# Attack order: center 5/10/20, edge 5/10/20, rotation 10/30/50, translation x3, scaling,
# Gauss, salt and pepper, speckle, median, mean, Gauss filter, JPEG, histogram, gamma.
TABLE_X_VOC2007 = _row([96.61, 96.67, 95.55, 97.16, 95.93, 84.79, 85.31, 66.04, 41.58, 72.29, 62.80, 59.35,
                        97.75, 97.95, 96.49, 97.55, 93.83, 93.52, 95.34, 93.69, 97.43, 97.89])
TABLE_XI_VOC2012 = _row([96.87, 95.39, 92.74, 93.21, 92.48, 82.59, 83.99, 64.69, 30.47, 64.48, 62.36, 52.42,
                         95.81, 95.87, 94.65, 95.66, 92.62, 92.82, 92.41, 98.44, 93.05, 94.75])
TABLE_XII_COCO = _row([98.57, 96.93, 94.51, 95.87, 93.01, 82.12, 83.41, 64.35, 30.44, 68.12, 67.32, 51.90,
                       94.84, 98.11, 96.50, 98.14, 94.44, 93.84, 94.95, 99.60, 93.39, 98.53])
TABLE_XV_N3 = _row([96.36, 94.59, 90.21, 92.90, 88.26, 81.18, 80.69, 70.05, 47.71, 71.68, 67.95, 58.91,
                    90.29, 97.22, 93.72, 95.99, 89.19, 92.36, 93.49, 99.52, 93.12, 96.05])
TABLE_XV_N4 = _row([97.32, 96.35, 94.91, 97.75, 94.11, 90.26, 82.29, 68.96, 48.92, 69.90, 70.24, 57.19,
                    95.52, 96.43, 95.86, 96.22, 94.45, 94.42, 95.00, 98.36, 96.48, 94.59])
TABLE_XV_N5 = _row([99.52, 99.32, 97.91, 97.09, 96.65, 87.07, 87.55, 70.21, 41.53, 71.17, 74.09, 56.03,
                    97.54, 99.25, 97.12, 99.32, 93.84, 98.49, 98.62, 99.80, 96.34, 99.17])
TABLE_XVII_M3 = _row([99.60, 99.80, 98.57, 98.03, 96.80, 85.33, 88.43, 67.95, 48.51, 66.99, 60.88, 45.29,
                      96.92, 99.80, 99.20, 99.47, 98.37, 98.57, 98.71, 100.00, 99.17, 99.39])
TABLE_XVII_M5 = _row([82.91, 82.01, 80.65, 82.08, 80.73, 73.66, 77.21, 64.07, 36.33, 66.73, 63.80, 59.61,
                      81.53, 83.31, 82.21, 83.15, 80.94, 81.70, 82.52, 83.65, 82.36, 83.09])
TABLE_XVII_M6 = _row([73.60, 73.39, 71.19, 73.52, 72.46, 66.58, 67.59, 58.27, 33.91, 63.65, 61.35, 54.47,
                      71.93, 74.12, 72.89, 74.25, 73.60, 73.18, 73.83, 74.43, 72.68, 74.25])

# Images needed for 10 B, 100 B and 1 KB.
_BYTES = (10, 100, 1024)
TABLE_VI_VOC2007 = dict(zip(_BYTES, (9.8, 88.6, 923.2)))
TABLE_VII_VOC2012 = dict(zip(_BYTES, (8.4, 79.0, 798.4)))
TABLE_VIII_COCO = dict(zip(_BYTES, (7.2, 65.2, 652.0)))
TABLE_XIII_N3 = dict(zip(_BYTES, (7.0, 67.0, 683.0)))
TABLE_XIII_N4 = dict(zip(_BYTES, (5.2, 50.8, 517.4)))
TABLE_XIII_N5 = dict(zip(_BYTES, (5.6, 55.4, 569.6)))
TABLE_XVI_M3 = dict(zip(_BYTES, (7.2, 69.4, 697.6)))
TABLE_XVI_M5 = dict(zip(_BYTES, (6.8, 61.6, 618.8)))
TABLE_XVI_M6 = dict(zip(_BYTES, (6.8, 60.2, 605.2)))

# Table XIV: label utilisation (%), COCO, M = 4.
TABLE_XIV_UTILISATION = {
    3: dict(zip(_BYTES, (96.43, 98.83, 99.89))),
    4: dict(zip(_BYTES, (95.83, 98.65, 98.95))),
    5: dict(zip(_BYTES, (75.90, 72.71, 76.36))),
    6: dict(zip(_BYTES, (52.67, 55.77, 55.39))),
}

_REFERENCES = {
    ("voc2007", 6, 4): ("Tables VI and X", TABLE_VI_VOC2007, TABLE_X_VOC2007),
    ("voc2012", 6, 4): ("Tables VII and XI", TABLE_VII_VOC2012, TABLE_XI_VOC2012),
    ("coco2014", 6, 4): ("Tables VIII and XII", TABLE_VIII_COCO, TABLE_XII_COCO),
    ("coco2014", 3, 4): ("Tables XIII and XV", TABLE_XIII_N3, TABLE_XV_N3),
    ("coco2014", 4, 4): ("Tables XIII and XV", TABLE_XIII_N4, TABLE_XV_N4),
    ("coco2014", 5, 4): ("Tables XIII and XV", TABLE_XIII_N5, TABLE_XV_N5),
    ("coco2014", 6, 3): ("Tables XVI and XVII", TABLE_XVI_M3, TABLE_XVII_M3),
    ("coco2014", 6, 5): ("Tables XVI and XVII", TABLE_XVI_M5, TABLE_XVII_M5),
    ("coco2014", 6, 6): ("Tables XVI and XVII", TABLE_XVI_M6, TABLE_XVII_M6),
}


def lookup(dataset: str, n: int, m: int):
    """Returns (table names, capacity, robustness), or None if the paper has no such setting."""
    return _REFERENCES.get((dataset, n, m))


def lookup_utilisation(dataset: str, n: int, m: int) -> dict | None:
    if dataset == "coco2014" and m == 4:
        return TABLE_XIV_UTILISATION.get(n)
    return None
