from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .client import RepoFile


DEFAULT_MODELSCOPE_CACHE = Path.home() / ".cache" / "modelscope"


@dataclass(frozen=True)
class FilePlan:
    repo_file: RepoFile
    target: Path
    part: Path
    complete_bytes: int
    part_bytes: int

    @property
    def local_bytes(self) -> int:
        return self.complete_bytes or self.part_bytes

    @property
    def is_complete(self) -> bool:
        return bool(self.repo_file.size and self.complete_bytes >= self.repo_file.size)

    @property
    def remaining_bytes(self) -> int | None:
        if self.repo_file.size is None:
            return None
        return max(self.repo_file.size - self.local_bytes, 0)


@dataclass(frozen=True)
class DownloadPlan:
    files: list[FilePlan]
    total_bytes: int
    local_bytes: int
    remaining_bytes: int
    unknown_size_count: int
    imported_partials: int

    @property
    def complete_count(self) -> int:
        return sum(1 for item in self.files if item.is_complete)

    @property
    def pending_count(self) -> int:
        return len(self.files) - self.complete_count


def build_download_plan(
    model_id: str,
    files: list[RepoFile],
    output_dir: Path,
    import_modelscope_cache: bool = True,
    cache_dir: Path = DEFAULT_MODELSCOPE_CACHE,
) -> DownloadPlan:
    if import_modelscope_cache:
        imported = import_partials_from_modelscope_cache(model_id, files, output_dir, cache_dir)
    else:
        imported = 0

    plans = [build_file_plan(repo_file, output_dir) for repo_file in files]
    total_bytes = sum(item.repo_file.size or 0 for item in plans)
    local_bytes = sum(item.local_bytes for item in plans if item.repo_file.size is not None)
    remaining_bytes = sum(item.remaining_bytes or 0 for item in plans)
    unknown_size_count = sum(1 for item in plans if item.repo_file.size is None)
    return DownloadPlan(plans, total_bytes, local_bytes, remaining_bytes, unknown_size_count, imported)


def build_file_plan(repo_file: RepoFile, output_dir: Path) -> FilePlan:
    target = output_dir / repo_file.path
    part = target.with_name(target.name + ".part")
    complete_bytes = sized_path_bytes(target, repo_file.size)
    part_bytes = 0 if complete_bytes else sized_path_bytes(part, repo_file.size)
    return FilePlan(repo_file, target, part, complete_bytes, part_bytes)


def sized_path_bytes(path: Path, expected_size: int | None) -> int:
    if not path.exists() or not path.is_file():
        return 0
    size = path.stat().st_size
    return min(size, expected_size) if expected_size else size


def import_partials_from_modelscope_cache(
    model_id: str,
    files: list[RepoFile],
    output_dir: Path,
    cache_dir: Path = DEFAULT_MODELSCOPE_CACHE,
) -> int:
    if not cache_dir.exists():
        return 0
    imported = 0
    for repo_file in files:
        target = output_dir / repo_file.path
        part = target.with_name(target.name + ".part")
        if target.exists() or part.exists():
            continue
        candidate = find_cached_partial(cache_dir, model_id, repo_file)
        if candidate is None:
            continue
        part.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate, part)
        imported += 1
    return imported


def find_cached_partial(cache_dir: Path, model_id: str, repo_file: RepoFile) -> Path | None:
    suffix_parts = [
        Path(model_id) / repo_file.path,
        Path(model_id.replace("/", os.sep)) / repo_file.path,
        Path(repo_file.path),
    ]
    best: Path | None = None
    best_size = 0
    for path in cache_dir.rglob(Path(repo_file.path).name):
        if not path.is_file():
            continue
        normalized = Path(*path.parts)
        if not any(str(normalized).endswith(str(suffix)) for suffix in suffix_parts):
            continue
        size = path.stat().st_size
        if repo_file.size and size >= repo_file.size:
            continue
        if size > best_size:
            best = path
            best_size = size
    return best


def check_disk_space(output_dir: Path, required_bytes: int) -> tuple[bool, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_dir).free
    return free_bytes >= required_bytes, free_bytes
