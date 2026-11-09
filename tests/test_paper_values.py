"""Guards the typed paper values against the averages printed in the paper's text."""
import pytest

from stego import paper_refs as p
from stego.attacks import TABLE_IX_ORDER, build_attacks


def mean(row):
    return sum(row.values()) / len(row)


@pytest.mark.parametrize(
    "row,printed",
    [
        (p.TABLE_XII_COCO, 85.86),  # text below Table XII; also n=6 in Table XV
        (p.TABLE_XV_N3, 85.52),
        (p.TABLE_XV_N4, 87.52),
        (p.TABLE_XV_N5, 88.98),
    ],
)
def test_averages_match_paper_text(row, printed):
    assert mean(row) == pytest.approx(printed, abs=0.02)


def test_all_rows_have_each_attack():
    for row in (p.TABLE_X_VOC2007, p.TABLE_XI_VOC2012, p.TABLE_XII_COCO, p.TABLE_XVII_M3, p.TABLE_XVII_M5, p.TABLE_XVII_M6):
        assert list(row) == list(TABLE_IX_ORDER)
    assert [name for name in build_attacks() if name != "No attack"] == list(TABLE_IX_ORDER)


def test_lookup():
    assert p.lookup("voc2007", 6, 4)[1][100] == 88.6
    assert p.lookup("coco2014", 4, 4)[1][1024] == 517.4
    assert p.lookup("coco2014", 6, 6)[2]["JPEG Compression"] == 74.43
    assert p.lookup("voc2007", 3, 4) is None
