"""Tests for the pure identity logic (src/hoopvision/identity.py)."""

from hoopvision.identity import (
    NumberRead,
    TrackBox,
    confirm_numbers,
    identify,
    ios,
    match_reads_to_tracks,
    merge_by_number,
)


def test_ios_number_inside_player_is_one():
    player = (0.0, 0.0, 100.0, 200.0)
    number = (40.0, 20.0, 60.0, 40.0)  # fully inside
    assert ios(number, player) == 1.0  # inter == smaller (number) area


def test_ios_disjoint_is_zero():
    assert ios((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_ios_partial_overlap():
    a = (0.0, 0.0, 10.0, 10.0)  # area 100
    b = (5.0, 0.0, 25.0, 10.0)  # area 200; intersection 5x10=50
    assert ios(a, b) == 0.5  # 50 / min(100, 200)


def test_match_assigns_read_to_enclosing_track():
    reads = [NumberRead(0, (40, 20, 60, 40), "23")]
    tracks = [
        TrackBox(0, 1, (0, 0, 100, 200)),  # encloses the number
        TrackBox(0, 2, (300, 0, 400, 200)),  # far away
    ]
    votes = match_reads_to_tracks(reads, tracks)
    assert votes == {1: [(0, "23")]}


def test_match_drops_read_below_ios_threshold():
    reads = [NumberRead(0, (90, 0, 130, 40), "23")]  # half outside track 1
    tracks = [TrackBox(0, 1, (0, 0, 100, 200))]
    assert match_reads_to_tracks(reads, tracks, min_ios=0.9) == {}


def test_match_respects_frame():
    reads = [NumberRead(5, (40, 20, 60, 40), "23")]
    tracks = [TrackBox(0, 1, (0, 0, 100, 200))]  # same box, wrong frame
    assert match_reads_to_tracks(reads, tracks) == {}


def test_confirm_requires_min_votes():
    votes = {1: [(0, "23"), (1, "23")]}  # only 2 reads
    assert confirm_numbers(votes, min_votes=3) == {}


def test_confirm_majority_wins():
    votes = {1: [(0, "23"), (1, "23"), (2, "23"), (3, "8")]}
    assert confirm_numbers(votes, min_votes=3) == {1: "23"}


def test_confirm_rejects_no_plurality():
    # 2x "23", 2x "8": winner has 2 votes but only 50% and below min_votes=3
    votes = {1: [(0, "23"), (1, "8"), (2, "23"), (3, "8")]}
    assert confirm_numbers(votes, min_votes=3, min_fraction=0.5) == {}


def test_merge_same_number_disjoint():
    confirmed = {1: "23", 2: "23"}
    spans = {1: (0, 10), 2: (15, 25)}  # disjoint
    remap = merge_by_number(confirmed, spans)
    assert remap[1] == remap[2] == 1


def test_merge_keeps_overlapping_same_number_separate():
    confirmed = {1: "23", 2: "23"}
    spans = {1: (0, 20), 2: (10, 30)}  # overlap -> two players, misread
    remap = merge_by_number(confirmed, spans)
    assert remap[1] != remap[2]


def test_merge_different_numbers_never_join():
    confirmed = {1: "23", 2: "8"}
    spans = {1: (0, 10), 2: (15, 25)}
    remap = merge_by_number(confirmed, spans)
    assert remap[1] != remap[2]


def test_merge_chains_three_disjoint_fragments():
    confirmed = {1: "23", 2: "23", 3: "23"}
    spans = {1: (0, 10), 2: (12, 20), 3: (22, 30)}
    remap = merge_by_number(confirmed, spans)
    assert remap[1] == remap[2] == remap[3] == 1


def test_merge_unconfirmed_keeps_own_id():
    confirmed = {1: "23"}
    spans = {1: (0, 10), 2: (12, 20)}  # track 2 anonymous
    remap = merge_by_number(confirmed, spans)
    assert remap[2] == 2


def test_identify_end_to_end():
    # track 1 (frames 0-2) and track 2 (frames 10-12) are the same #23 player,
    # fragmented; both read #23 three times; disjoint in time -> merge.
    number_box = (40, 20, 60, 40)
    player_a = (0, 0, 100, 200)
    reads = [NumberRead(f, number_box, "23") for f in (0, 1, 2, 10, 11, 12)]
    tracks = [TrackBox(f, 1, player_a) for f in (0, 1, 2)]
    tracks += [TrackBox(f, 2, player_a) for f in (10, 11, 12)]
    spans = {1: (0, 2), 2: (10, 12)}
    remap, numbers = identify(reads, tracks, spans)
    assert remap[1] == remap[2]
    assert numbers[remap[1]] == "23"


def test_identify_empty():
    remap, numbers = identify([], [], {})
    assert remap == {} and numbers == {}
