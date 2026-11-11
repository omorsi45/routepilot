from routepilot_train.data import selection


def test_selection_matches_the_ruling():
    assert len(selection.KEPT) == 11
    assert selection.KEPT["mmlupro"] == "test_3000"
    assert selection.MIXED_EXTRA == {"mmlupro": "test_1000"}
    assert len(selection.FLAGSHIP_MODELS) == 12
    assert len(set(selection.FLAGSHIP_MODELS)) == 12
    assert selection.NON_MODELS == ("openrouter",)
    assert selection.SUCCESS_THRESHOLD == 0.5
    assert selection.DATASET_BYTES == 1283503080


def test_every_kept_dataset_has_one_domain():
    assert set(selection.DOMAINS) == set(selection.KEPT)
    assert set(selection.DOMAINS.values()) == {"math", "code", "knowledge", "reasoning", "writing"}


def test_classify_roles():
    assert selection.classify("simpleqa", "test") == ("canonical", "")
    assert selection.classify("mmlupro", "test_1000") == ("mixed_extra", "")
    role, reason = selection.classify("hle", "subset_500")
    assert role == "excluded" and "hle:test" in reason
    role, reason = selection.classify("arenahard", "test")
    assert role == "excluded" and "duplicate" in reason
    assert selection.classify("math500", "test")[0] == "excluded"
    assert selection.classify("hle", "other") == (
        "excluded",
        "split 'other' is not the selected split of hle",
    )
    assert selection.classify("nope", "test") == ("excluded", "dataset is not in the selection")


def test_excluded_never_overlaps_the_canonical_splits():
    for dataset, split in selection.KEPT.items():
        assert f"{dataset}:{split}" not in selection.EXCLUDED
        assert dataset not in selection.EXCLUDED
