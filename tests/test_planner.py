from msdl.client import RepoFile
from msdl.planner import build_download_plan


def test_plan_counts_complete_and_partial_files(tmp_path):
    (tmp_path / "done.bin").write_bytes(b"12345")
    (tmp_path / "partial.bin.part").write_bytes(b"12")
    files = [RepoFile("done.bin", 5), RepoFile("partial.bin", 10)]

    plan = build_download_plan("owner/model", files, tmp_path, import_modelscope_cache=False)

    assert plan.complete_count == 1
    assert plan.pending_count == 1
    assert plan.total_bytes == 15
    assert plan.local_bytes == 7
    assert plan.remaining_bytes == 8


def test_plan_imports_modelscope_cache_partial_to_local_dir(tmp_path):
    cache = tmp_path / "cache"
    local_dir = tmp_path / "local"
    cached_file = cache / "owner" / "model" / "weights.bin"
    cached_file.parent.mkdir(parents=True)
    cached_file.write_bytes(b"123")
    files = [RepoFile("weights.bin", 10)]

    plan = build_download_plan("owner/model", files, local_dir, import_modelscope_cache=True, cache_dir=cache)

    assert plan.imported_partials == 1
    assert (local_dir / "weights.bin.part").read_bytes() == b"123"
    assert plan.local_bytes == 3
    assert plan.remaining_bytes == 7
