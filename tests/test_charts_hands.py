"""Slice C — 169 preflop hand-class taxonomy (impl doc §3 Slice C).

The canonical labelling + combo counts + concrete card enumeration shared by the
equity-matrix generator and the jam/fold solver.
"""

from pokerlab.charts import hands


def test_there_are_169_unique_classes():
    assert len(hands.HAND_CLASSES) == 169
    assert len(set(hands.HAND_CLASSES)) == 169


def test_class_breakdown_13_pairs_78_suited_78_offsuit():
    pairs = [h for h in hands.HAND_CLASSES if hands.is_pair(h)]
    suited = [h for h in hands.HAND_CLASSES if h.endswith("s")]
    offsuit = [h for h in hands.HAND_CLASSES if h.endswith("o")]
    assert len(pairs) == 13
    assert len(suited) == 78
    assert len(offsuit) == 78


def test_index_round_trip():
    for i, label in enumerate(hands.HAND_CLASSES):
        assert hands.HAND_INDEX[label] == i


def test_known_labels_present():
    for label in ("AA", "AKs", "AKo", "22", "72o", "T9s", "A2s"):
        assert label in hands.HAND_INDEX


def test_combo_counts():
    assert hands.combos("AA") == 6
    assert hands.combos("AKs") == 4
    assert hands.combos("AKo") == 12
    # total combos across all classes == C(52,2) = 1326
    assert sum(hands.combos(h) for h in hands.HAND_CLASSES) == 1326


def test_ranks_high_first():
    assert hands.hand_ranks("AKs") == (14, 13)
    assert hands.hand_ranks("72o") == (7, 2)
    assert hands.hand_ranks("22") == (2, 2)


def test_card_combos_match_counts_and_are_class_consistent():
    for label in ("AA", "AKs", "AKo", "T9s", "22"):
        combos = hands.card_combos(label)
        assert len(combos) == hands.combos(label)
        hi, lo = hands.hand_ranks(label)
        for c1, c2 in combos:
            assert c1 != c2
            r1, r2 = c1 // 4, c2 // 4  # rank index (0..12)
            s1, s2 = c1 % 4, c2 % 4
            ranks = {r1 + 2, r2 + 2}
            assert ranks == {hi, lo} if hi != lo else (r1 == r2)
            if hands.is_pair(label):
                assert r1 == r2 and s1 != s2
            elif label.endswith("s"):
                assert s1 == s2
            else:  # offsuit
                assert s1 != s2


def test_card_combos_use_distinct_real_cards():
    # every concrete card id is in 0..51 and the two cards differ
    for label in hands.HAND_CLASSES:
        for c1, c2 in hands.card_combos(label):
            assert 0 <= c1 < 52 and 0 <= c2 < 52 and c1 != c2
