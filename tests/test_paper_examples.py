"""Checks the code against the worked examples in Section III of the paper."""
import numpy as np

from stego.hiding import AuxInfo, bytes_to_segments, hide_segments, segments_to_bytes
from stego.mapping import build_dictionary, build_index, segment_to_bits, top_objects
from stego.recovery import decode_image


FIG5 = {"person": 33, "refrigerator": 36, "cat": 10}


def obj(label, area, score=0.95):
    return {"label": label, "score": score, "box": [0.0, 0.0, float(area), 1.0]}


def example_index(monkeypatch=None):
    """Fig. 7: person, refrigerator, cat, cat, largest first, plus a smaller fifth object."""
    import stego.mapping as mapping

    objects = [[obj("cat", 30), obj("person", 100), obj("cat", 40), obj("refrigerator", 80), obj("cup", 5)]]
    original = mapping.build_dictionary
    mapping.build_dictionary = lambda counts, n: {**FIG5, "cup": 0}
    try:
        return build_index(["test2014_46352.jpg"], objects, np.array([453]), 6, 4, 0.9)
    finally:
        mapping.build_dictionary = original


def test_example_2_top_objects_are_largest_four():
    index = example_index()
    assert index.labels[0] == ("person", "refrigerator", "cat", "cat")


def test_example_3_table_i_row():
    rows = list(example_index().table_rows())
    assert rows[0] == {
        "filename": "test2014_46352.jpg",
        "object_label": "person",
        "object_number": 1,
        "object_sequence": "100001",
        "cluster": "C453",
    }
    assert [row["object_sequence"] for row in rows] == ["100001", "100100", "001010", "001010"]


def test_example_4_segments_and_auxiliary_tuple():
    segments = [0b001010, 0b100001, 0b100100]
    index = example_index()
    result = hide_segments(segments, index, np.random.default_rng(0))
    assert result.complete and len(result.stego) == 1
    assert result.stego[0].aux.as_tuple() == (2, 3, 1, 0, "C453", -1, 0)


def test_example_5_recovery():
    index = example_index()
    aux = AuxInfo((2, 3, 1, 0), 453, 0)
    assert decode_image(0, aux, index) == [0b001010, 0b100001, 0b100100]


def test_confidence_must_be_above_threshold():
    objects = [obj("a", 10, 0.90), obj("b", 5, 0.9001)]
    assert [o["label"] for o in top_objects(objects, 0.9, 4)] == ["b"]


def test_padding_round_trip_for_each_n():
    message = bytes(range(251)) + b"\xff\x00\x80"
    for n in range(1, 9):
        segments, padding = bytes_to_segments(message, n)
        assert all(0 <= s < 2**n for s in segments)
        assert padding == (n - len(message) * 8 % n) % n
        assert segments_to_bytes(segments, n, padding) == message


def test_lost_segment_counts_as_zero_bits():
    assert segments_to_bytes([None, 1], 4, 0) == b"\x01"


def test_segment_bits():
    assert segment_to_bits(10, 6) == "001010"


def test_dictionary_keeps_common_label_alone_and_merges_rare_neighbours():
    counts = {"wine glass": 2, "zebra": 3, "person": 1000, "apple": 5, "airplane": 6, "cat": 200}
    dictionary = build_dictionary(counts, 2)
    assert set(dictionary.values()) == {0, 1, 2, 3}
    assert dictionary["person"] not in {dictionary[k] for k in counts if k != "person"}
    assert dictionary["wine glass"] == dictionary["zebra"] == 3
    labels = sorted(counts)
    groups = [dictionary[label] for label in labels]
    assert groups == sorted(groups)


def test_dictionary_refuses_too_few_labels():
    import pytest

    from stego.mapping import MappingError

    with pytest.raises(MappingError):
        build_dictionary({"a": 1, "b": 1, "c": 1}, 2)
