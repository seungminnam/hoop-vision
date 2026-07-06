import numpy as np

from hoopvision.teams import TeamAssigner, kmeans_two, torso_crop

DARK = [40.0, 128.0, 128.0]  # LAB-ish: low L
BRIGHT = [220.0, 128.0, 128.0]  # high L


def _noisy(base, n, seed):
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, 4.0, size=(n, 3))


def test_kmeans_two_separates_clusters_and_orders_by_darkness():
    features = np.vstack([_noisy(DARK, 20, 1), _noisy(BRIGHT, 20, 2)]).astype(np.float32)
    labels = kmeans_two(features)
    assert set(labels[:20]) == {0}  # darker cluster is always team 0
    assert set(labels[20:]) == {1}


def test_kmeans_two_handles_tiny_input():
    assert kmeans_two(np.empty((0, 3), dtype=np.float32)).tolist() == []
    assert kmeans_two(np.array([DARK], dtype=np.float32)).tolist() == [0]


def test_assigner_majority_vote_smooths_outliers():
    assigner = TeamAssigner()
    # Track 1 is dark except one noisy bright observation; track 2 is bright.
    for f in _noisy(DARK, 9, 3):
        assigner.observe_feature(1, f)
    assigner.observe_feature(1, np.array(BRIGHT))
    for f in _noisy(BRIGHT, 10, 4):
        assigner.observe_feature(2, f)
    assigner.fit()
    assert assigner.team_of(1) == 0
    assert assigner.team_of(2) == 1
    assert assigner.team_of(99) is None


def test_assigner_empty_fit():
    assigner = TeamAssigner()
    assert assigner.fit() == {}


def test_torso_crop_extracts_upper_center_region():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    frame[60:100, 110:140] = (0, 0, 255)  # red torso area
    crop = torso_crop(frame, (100.0, 50.0, 150.0, 180.0))
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop.mean() > 0  # overlaps the painted torso


def test_torso_crop_degenerate_box_returns_none():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert torso_crop(frame, (10.0, 10.0, 11.0, 11.0)) is None
