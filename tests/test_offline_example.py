from examples.offline_workflow import run


def test_all_stages_end_to_end_with_fakes(tmp_path):
    summary = run(tmp_path)
    assert summary["demonstration_only"]
    assert summary["synthesis"]["generated"] == 1
    assert summary["governance"]["pass"] == 1
    assert summary["review"]["accept"] == 1
