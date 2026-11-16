import pytest

from motioniq.video import (
    build_manifest,
    create_synthetic_video,
    extract_frames,
    inspect_video,
    split_by_participant,
)


def test_synthetic_video_inspection_and_frames(tmp_path):
    video = create_synthetic_video(tmp_path / "sample.mp4", frames=12, fps=12)
    metadata = inspect_video(video)
    assert metadata["frame_count"] == 12
    assert metadata["fps"] == 12
    frames = extract_frames(video, tmp_path / "frames", every_n_frames=3)
    assert len(frames) == 4
    rows = build_manifest(tmp_path, tmp_path / "manifest.json")
    assert any(row.get("frame_count") == 12 for row in rows)


def test_participant_split_is_disjoint_and_reproducible():
    rows = [{"participant_id": str(index), "recording": index} for index in range(20)]
    first = split_by_participant(rows, seed=7)
    second = split_by_participant(rows, seed=7)
    assert first == second
    sets = [{row["participant_id"] for row in first[split]} for split in first]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    with pytest.raises(ValueError):
        split_by_participant([{"recording": "missing participant"}])
