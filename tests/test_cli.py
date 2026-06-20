from pathlib import Path

from rich.progress import Progress

from msdl.cli import ProgressTasks, download_one, local_downloaded_size, parse_args
from msdl.client import RepoFile


def test_parse_hf_style_file_download():
    args = parse_args(["download", "Qwen/Qwen2.5-0.5B", "config.json", "--local-dir", "out"])

    assert args.command == "download"
    assert args.model_id == "Qwen/Qwen2.5-0.5B"
    assert args.filename == "config.json"
    assert args.local_dir == "out"


def test_parse_hf_style_snapshot_download():
    args = parse_args(["download", "Qwen/Qwen2.5-0.5B", "--include", "*.json"])

    assert args.command == "download"
    assert args.filename is None
    assert args.include == ["*.json"]


def test_parse_legacy_file_download():
    args = parse_args(["Qwen/Qwen2.5-0.5B", "--file", "config.json", "-o", "out"])

    assert args.command == "legacy"
    assert args.file == "config.json"
    assert args.output == "out"


def test_download_one_uses_requested_destination():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def download_file(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    args = parse_args(["download", "Qwen/Qwen2.5-0.5B", "nested/config.json", "--local-dir", "out"])
    client = FakeClient()

    rc = download_one(client, args, args.filename, Path(args.local_dir) / args.filename)

    assert rc == 0
    assert client.calls[0][0][2] == Path("out/nested/config.json")


def test_local_downloaded_size_counts_complete_file(tmp_path):
    target = tmp_path / "nested" / "config.json"
    target.parent.mkdir()
    target.write_bytes(b"12345")

    size = local_downloaded_size(tmp_path, RepoFile("nested/config.json", 5))

    assert size == 5


def test_local_downloaded_size_counts_part_file(tmp_path):
    part = tmp_path / "model.bin.part"
    part.write_bytes(b"123")

    assert local_downloaded_size(tmp_path, RepoFile("model.bin", 10)) == 3


def test_complete_files_are_not_rendered_as_individual_progress_tasks(tmp_path):
    (tmp_path / "done.bin").write_bytes(b"12345")
    progress = Progress()

    tasks = ProgressTasks(progress, total_files=1, files=[RepoFile("done.bin", 5)], output_dir=tmp_path)

    assert tasks.is_complete("done.bin")
    assert len(progress.tasks) == 1


def test_completed_active_file_keeps_row_but_marks_done():
    progress = Progress()
    tasks = ProgressTasks(progress, total_files=1, files=[RepoFile("active.bin", 5)])

    tasks.update("active.bin", 5, 5, 5)
    tasks.mark_done("active.bin")

    assert tasks.is_complete("active.bin")
    assert len(progress.tasks) == 2
    assert progress.tasks[1].visible is True
    assert progress.tasks[1].fields["done"] is True


def test_parse_ls_command():
    args = parse_args(["ls", "Qwen/Qwen2.5-0.5B"])
    assert args.command == "ls"
    assert args.model_id == "Qwen/Qwen2.5-0.5B"


def test_parse_info_command():
    args = parse_args(["info", "Qwen/Qwen2.5-0.5B"])
    assert args.command == "info"
    assert args.model_id == "Qwen/Qwen2.5-0.5B"


def test_parse_search_command():
    args = parse_args(["search", "qwen2", "--page-size", "10", "--page-number", "2"])
    assert args.command == "search"
    assert args.keyword == "qwen2"
    assert args.page_size == 10
    assert args.page_number == 2

