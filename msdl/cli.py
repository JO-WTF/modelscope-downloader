from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    ProgressColumn,
    Progress,
    TaskID,
    TextColumn,
)
from rich.text import Text
from rich.table import Table

from datetime import datetime
from .client import RepoFile, filter_files
from . import __version__
from .client import DEFAULT_ENDPOINT, ModelScopeClient, ModelScopeError, SearchResultModel
from .planner import FilePlan, build_download_plan, check_disk_space


LOGO = r"""
 _   .-')                _ .-') _     ('-.             .-')                              _ (`-.    ('-.
( '.( OO )_             ( (  OO) )  _(  OO)           ( OO ).                           ( (OO  ) _(  OO)
 ,--.   ,--.).-'),-----. \     .'_ (,------.,--.     (_)---\_)   .-----.  .-'),-----.  _.`     \(,------.
 |   `.'   |( OO'  .-.  ',`'--..._) |  .---'|  |.-') /    _ |   '  .--./ ( OO'  .-.  '(__...--'' |  .---'
 |         |/   |  | |  ||  |  \  ' |  |    |  | OO )\  :` `.   |  |('-. /   |  | |  | |  /  | | |  |
 |  |'.'|  |\_) |  |\|  ||  |   ' |(|  '--. |  |`-' | '..`''.) /_) |OO  )\_) |  |\|  | |  |_.' |(|  '--.
 |  |   |  |  \ |  | |  ||  |   / : |  .--'(|  '---.'.-._)   \ ||  |`-'|   \ |  | |  | |  .___.' |  .--'
 |  |   |  |   `'  '-'  '|  '--'  / |  `---.|      | \       /(_'  '--'\    `'  '-'  ' |  |      |  `---.
 `--'   `--'     `-----' `-------'  `------'`------'  `-----'    `-----'      `-----'  `--'      `------'
"""

console = Console()
_resize_lock = Lock()
_resize_requested = False


def request_terminal_redraw(signum, frame) -> None:  # type: ignore[no-untyped-def]
    global _resize_requested
    with _resize_lock:
        _resize_requested = True


if hasattr(signal, "SIGWINCH"):
    signal.signal(signal.SIGWINCH, request_terminal_redraw)


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-r", "--revision", default="master", help="Model revision/branch. Defaults to master.")
    parser.add_argument("--endpoint", default=os.getenv("MODELSCOPE_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--token", default=os.getenv("MODELSCOPE_TOKEN"), help="Access token or MODELSCOPE_TOKEN.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    parser.add_argument("--no-verify-ssl", action="store_true", help="Disable TLS certificate verification.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel downloads for repository mode.")
    parser.add_argument("--retries", type=int, default=5, help="Retries per file before skipping it.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned downloads and disk usage, then exit.")
    parser.add_argument("--simple-progress", action="store_true", help="Use non-live progress output for fragile terminals.")
    parser.add_argument(
        "--no-import-modelscope-cache",
        action="store_true",
        help="Do not import partial files from the official ModelScope cache.",
    )
    parser.add_argument(
        "--modelscope-cache-dir",
        default=str(Path.home() / ".cache" / "modelscope"),
        help="Official ModelScope cache directory to scan for partial downloads.",
    )


def add_filter_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Glob pattern to include for repository downloads. Can be repeated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Glob pattern to exclude for repository downloads. Can be repeated.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ms",
        description="Download ModelScope model files with optional SSL verification bypass.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    download = subparsers.add_parser(
        "download",
        help="Download a file or a full ModelScope repository.",
        description="Download a ModelScope file or repository, similar to `hf download`.",
    )
    download.add_argument("model_id", help="ModelScope model id, for example Qwen/Qwen2.5-0.5B")
    download.add_argument(
        "filename",
        nargs="?",
        help="Optional file path inside the repository. Omit it to download the repository.",
    )
    download.add_argument(
        "--local-dir",
        "-d",
        default=".",
        help="Directory to save the file or repository. Defaults to current directory.",
    )
    download.add_argument(
        "--local-dir-use-symlinks",
        choices=["auto", "true", "false"],
        default=None,
        help="Accepted for Hugging Face CLI muscle memory; ignored because this downloader writes real files.",
    )
    add_common_options(download)
    add_filter_options(download)

    ls_parser = subparsers.add_parser(
        "ls",
        help="List files in a ModelScope repository.",
    )
    ls_parser.add_argument("model_id", help="ModelScope model id")
    add_common_options(ls_parser)
    add_filter_options(ls_parser)

    info_parser = subparsers.add_parser(
        "info",
        help="Show metadata for a ModelScope repository.",
    )
    info_parser.add_argument("model_id", help="ModelScope model id")
    add_common_options(info_parser)

    search_parser = subparsers.add_parser(
        "search",
        help="Search models on ModelScope by keyword.",
    )
    search_parser.add_argument("keyword", help="Keyword to search for")
    search_parser.add_argument(
        "--page-size",
        type=int,
        default=20,
        help="Number of results to display. Defaults to 20.",
    )
    search_parser.add_argument(
        "--page-number",
        type=int,
        default=1,
        help="Page number to display. Defaults to 1.",
    )
    add_common_options(search_parser)

    legacy = argparse.ArgumentParser(add_help=False)
    legacy.add_argument("model_id", help=argparse.SUPPRESS)
    legacy.add_argument("-o", "--output", default=".", help=argparse.SUPPRESS)
    add_common_options(legacy)
    mode = legacy.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file", help=argparse.SUPPRESS)
    mode.add_argument("--snapshot", action="store_true", help=argparse.SUPPRESS)
    add_filter_options(legacy)
    parser.set_defaults(_legacy_parser=legacy)
    return parser


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    if argv and argv[0] not in ("download", "ls", "info", "search") and not argv[0].startswith("-"):
        args = parser.get_default("_legacy_parser").parse_args(argv)
        args.command = "legacy"
        return args
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        raise SystemExit(2)
    return args


def download_one(
    client: ModelScopeClient,
    args: argparse.Namespace,
    file_path: str,
    output: Path,
    progress_output_dir: Path | None = None,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    repo_file = resolve_single_repo_file(client, args, file_path)
    plan = build_download_plan(
        args.model_id,
        [repo_file],
        progress_output_dir or output.parent,
        import_modelscope_cache=not args.no_import_modelscope_cache,
        cache_dir=Path(args.modelscope_cache_dir),
    )
    print_plan(plan, progress_output_dir or output.parent, args.dry_run)
    ok, free_bytes = check_disk_space(output.parent, plan.remaining_bytes)
    if not ok:
        raise ModelScopeError(
            f"Not enough disk space in {output.parent}: need {format_bytes(plan.remaining_bytes)}, "
            f"free {format_bytes(free_bytes)}."
        )
    if args.dry_run:
        return 0
    if plan.files[0].is_complete:
        console.print(f"[cyan]skipped[/cyan] complete file: {output}")
        return 0
    if args.simple_progress:
        path = download_with_retries(client, args, file_path, output, NullProgressTasks())
        console.print(f"[green]saved[/green]: {path}")
        return 0
    with create_progress() as progress:
        tasks = ProgressTasks(
            progress,
            total_files=1,
            file_plans=plan.files,
            output_dir=progress_output_dir,
        )
        download_with_retries(client, args, file_path, output, tasks)
        tasks.mark_done(file_path)
    console.print(f"[green]saved[/green]: {output}")
    return 0


def download_repo(client: ModelScopeClient, args: argparse.Namespace, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = filter_files(client.list_files(args.model_id, args.revision), args.include, args.exclude)
    if not files:
        raise ModelScopeError("No files matched the requested filters.")
    plan = build_download_plan(
        args.model_id,
        files,
        output_dir,
        import_modelscope_cache=not args.no_import_modelscope_cache,
        cache_dir=Path(args.modelscope_cache_dir),
    )
    print_plan(plan, output_dir, args.dry_run)
    ok, free_bytes = check_disk_space(output_dir, plan.remaining_bytes)
    if not ok:
        raise ModelScopeError(
            f"Not enough disk space in {output_dir}: need {format_bytes(plan.remaining_bytes)}, "
            f"free {format_bytes(free_bytes)}."
        )
    if args.dry_run:
        return 0

    console.print(
        Panel.fit(
            f"[bold]ModelScope download[/bold]\n"
            f"repo: [cyan]{args.model_id}[/cyan]\n"
            f"files: [cyan]{len(files)}[/cyan]   workers: [cyan]{max(args.max_workers, 1)}[/cyan]\n"
            f"remaining: [cyan]{format_bytes(plan.remaining_bytes)}[/cyan]",
            border_style="cyan",
        )
    )
    if args.simple_progress:
        result = download_files_parallel(client, args, output_dir, [item for item in plan.files if not item.is_complete], NullProgressTasks())
        console.print(f"[green]downloaded[/green] {len(result.written)} files")
        if result.failures:
            print_failures(result.failures)
            return 1
        console.print(f"[green]done[/green]: {output_dir}")
        return 0

    with create_progress() as progress:
        tasks = ProgressTasks(
            progress,
            total_files=len(files),
            file_plans=plan.files,
            output_dir=output_dir,
        )
        pending = [item for item in plan.files if not tasks.is_complete(item.repo_file.path)]
        result = download_files_parallel(client, args, output_dir, pending, tasks)
        skipped = [item.target for item in plan.files if tasks.is_complete(item.repo_file.path)]
    console.print(f"[green]downloaded[/green] {len(result.written)} files, [cyan]skipped[/cyan] {len(skipped)} complete files")
    if result.failures:
        print_failures(result.failures)
        return 1
    console.print(f"[green]done[/green]: {output_dir}")
    return 0


def create_progress() -> Progress:
    return ResizeAwareProgress(
        TextColumn("{task.fields[label]}", justify="left"),
        BarColumn(bar_width=None),
        TextColumn("{task.percentage:>6.1f}%", justify="right"),
        DownloadColumn(binary_units=True),
        CleanTransferSpeedColumn(),
        CleanTimeRemainingColumn(),
        console=console,
        expand=True,
        refresh_per_second=8,
    )


class ResizeAwareProgress(Progress):
    def refresh(self) -> None:
        global _resize_requested
        should_clear = False
        with _resize_lock:
            if _resize_requested:
                should_clear = True
                _resize_requested = False
        if should_clear:
            self.console.clear()
        super().refresh()


class ProgressTasks:
    def __init__(
        self,
        progress: Progress,
        total_files: int,
        files: list[RepoFile] | None = None,
        file_plans: list[FilePlan] | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.progress = progress
        self.lock = Lock()
        self.file_tasks: dict[str, TaskID] = {}
        self.file_totals: dict[str, int] = {}
        self.file_completed: dict[str, int] = {}
        self.complete_files: set[str] = set()
        self.overall_total = 0
        self.overall_completed = 0
        self.overall_task = progress.add_task(
            "TOTAL",
            total=None,
            name=self._overall_name(0, total_files),
            label=self._overall_label(0, total_files),
        )
        self.total_files = total_files
        repo_files = [plan.repo_file for plan in file_plans] if file_plans else (files or [])
        plan_by_path = {plan.repo_file.path: plan for plan in file_plans or []}
        for repo_file in repo_files:
            if repo_file.size:
                self.file_totals[repo_file.path] = repo_file.size
                self.overall_total += repo_file.size
            file_plan = plan_by_path.get(repo_file.path)
            local_completed = file_plan.local_bytes if file_plan else local_downloaded_size(output_dir, repo_file)
            is_complete = bool(repo_file.size and local_completed >= repo_file.size)
            if local_completed:
                self.file_completed[repo_file.path] = local_completed
                self.overall_completed += local_completed
                if not is_complete:
                    task_id = self.progress.add_task(
                        repo_file.path,
                        total=repo_file.size,
                        completed=local_completed,
                        name=repo_file.path,
                        label=self._file_label(repo_file.path),
                        done=False,
                    )
                    self.file_tasks[repo_file.path] = task_id
            if is_complete:
                self.complete_files.add(repo_file.path)
        if self.overall_total:
            self.progress.update(self.overall_task, total=self.overall_total, completed=self.overall_completed)
        self._refresh_overall_name()

    def is_complete(self, file_path: str) -> bool:
        return file_path in self.complete_files

    def update(self, file_path: str, downloaded: int, total: int | None, advance: int) -> None:
        with self.lock:
            task_id = self.file_tasks.get(file_path)
            if task_id is None:
                task_id = self.progress.add_task(
                    file_path,
                    total=total,
                    name=file_path,
                    label=self._file_label(file_path),
                    done=False,
                )
                self.file_tasks[file_path] = task_id

            if total and file_path not in self.file_totals:
                self.file_totals[file_path] = total
                self.overall_total += total
                self.progress.update(self.overall_task, total=self.overall_total)

            previous = self.file_completed.get(file_path, 0)
            delta = downloaded - previous
            self.file_completed[file_path] = downloaded
            if delta:
                self.overall_completed += delta
            self.progress.update(task_id, total=total, completed=downloaded)
            self.progress.update(self.overall_task, completed=self.overall_completed)

    def mark_done(self, file_path: str) -> None:
        with self.lock:
            self.complete_files.add(file_path)
            task_id = self.file_tasks.get(file_path)
            if task_id is not None:
                self.progress.update(task_id, done=True)
            self._refresh_overall_name()

    def _refresh_overall_name(self) -> None:
        self.progress.update(
            self.overall_task,
            name=self._overall_name(len(self.complete_files), self.total_files),
            label=self._overall_label(len(self.complete_files), self.total_files),
        )

    @staticmethod
    def _overall_name(done: int, total: int) -> str:
        remaining = max(total - done, 0)
        return f"━━ TOTAL ━━ files {done:>4}/{total:<4} left {remaining:<4}"

    @classmethod
    def _overall_label(cls, done: int, total: int) -> str:
        return f"[bold magenta]{cls._overall_name(done, total):<44.44}[/bold magenta]"

    @staticmethod
    def _file_label(file_path: str) -> str:
        return f"[cyan]{file_path:<44.44}[/cyan]"


class CleanTransferSpeedColumn(ProgressColumn):
    def render(self, task) -> Text:  # type: ignore[no-untyped-def]
        if task.fields.get("done"):
            return Text(" " * 14)
        speed = task.speed
        if speed is None:
            return Text(" " * 14)
        return Text(f"{format_bytes(int(speed))}/s".rjust(14))


class CleanTimeRemainingColumn(ProgressColumn):
    def render(self, task) -> Text:  # type: ignore[no-untyped-def]
        if task.fields.get("done"):
            return Text(" " * 8)
        remaining = task.time_remaining
        if remaining is None:
            return Text(" " * 8)
        return Text(format_duration(int(remaining)).rjust(8))


class NullProgressTasks:
    def update(self, file_path: str, downloaded: int, total: int | None, advance: int) -> None:
        return None

    def mark_done(self, file_path: str) -> None:
        console.print(f"[green]done[/green] {file_path}")


def local_downloaded_size(output_dir: Path | None, repo_file: RepoFile) -> int:
    if output_dir is None:
        return 0
    target = output_dir / repo_file.path
    part = target.with_name(target.name + ".part")
    if repo_file.size and target.exists():
        return min(target.stat().st_size, repo_file.size)
    if part.exists():
        return min(part.stat().st_size, repo_file.size) if repo_file.size else part.stat().st_size
    return 0


def download_files_parallel(
    client: ModelScopeClient,
    args: argparse.Namespace,
    output_dir: Path,
    files: list[FilePlan],
    tasks: ProgressTasks,
) -> DownloadResult:
    max_workers = max(args.max_workers, 1)
    result = DownloadResult()
    if max_workers == 1:
        for file_plan in files:
            download_plan_item(client, args, file_plan, tasks, result)
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_plan_item, client, args, file_plan, tasks, None) for file_plan in files]
        for future in as_completed(futures):
            item_result = future.result()
            result.written.extend(item_result.written)
            result.failures.extend(item_result.failures)
    return result


@dataclass
class DownloadFailure:
    file_path: str
    error: str


@dataclass
class DownloadResult:
    written: list[Path]
    failures: list[DownloadFailure]

    def __init__(self) -> None:
        self.written = []
        self.failures = []


def download_plan_item(
    client: ModelScopeClient,
    args: argparse.Namespace,
    file_plan: FilePlan,
    tasks: ProgressTasks,
    result: DownloadResult | None,
) -> DownloadResult:
    local_result = result or DownloadResult()
    try:
        path = download_with_retries(client, args, file_plan.repo_file.path, file_plan.target, tasks)
    except Exception as exc:  # noqa: BLE001
        local_result.failures.append(DownloadFailure(file_plan.repo_file.path, str(exc)))
        return local_result
    tasks.mark_done(file_plan.repo_file.path)
    local_result.written.append(path)
    return local_result


def download_with_retries(
    client: ModelScopeClient,
    args: argparse.Namespace,
    file_path: str,
    output: Path,
    tasks: ProgressTasks,
) -> Path:
    attempts = max(args.retries, 0) + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return client.download_file(
                args.model_id,
                file_path,
                output,
                revision=args.revision,
                progress_callback=tasks.update,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(min(2 ** (attempt - 1), 10))
    raise ModelScopeError(f"failed after {attempts} attempts: {last_error}")


def resolve_single_repo_file(client: ModelScopeClient, args: argparse.Namespace, file_path: str) -> RepoFile:
    try:
        files = client.list_files(args.model_id, args.revision)
    except (AttributeError, ModelScopeError):
        return RepoFile(file_path, None)
    for repo_file in files:
        if repo_file.path == file_path:
            return repo_file
    return RepoFile(file_path, None)


def print_plan(plan, output_dir: Path, dry_run: bool) -> None:
    table = Table(title="Download plan" + (" (dry run)" if dry_run else ""))
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value", justify="right")
    table.add_row("output", str(output_dir))
    table.add_row("files", str(len(plan.files)))
    table.add_row("complete files", str(plan.complete_count))
    table.add_row("pending files", str(plan.pending_count))
    table.add_row("total size", format_bytes(plan.total_bytes) if plan.total_bytes else "unknown")
    table.add_row("local bytes", format_bytes(plan.local_bytes))
    table.add_row("remaining", format_bytes(plan.remaining_bytes))
    table.add_row("unknown size files", str(plan.unknown_size_count))
    table.add_row("imported partials", str(plan.imported_partials))
    console.print(table)


def print_failures(failures: list[DownloadFailure]) -> None:
    table = Table(title="Failed files")
    table.add_column("file", style="red")
    table.add_column("error")
    for failure in failures:
        table.add_row(failure.file_path, failure.error)
    console.print(table)


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{value} B"


def format_duration(seconds: int) -> str:
    hours, remainder = divmod(max(seconds, 0), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def command_ls(client: ModelScopeClient, args: argparse.Namespace) -> int:
    files = filter_files(client.list_files(args.model_id, args.revision), args.include, args.exclude)
    if not files:
        console.print("[yellow]No files found.[/yellow]")
        return 0

    table = Table(title=f"Files in {args.model_id} ({args.revision})", box=None)
    table.add_column("Path", style="cyan")
    table.add_column("Size", justify="right", style="green")

    total_size = 0
    for f in files:
        size_str = format_bytes(f.size) if f.size is not None else "unknown"
        table.add_row(f.path, size_str)
        if f.size:
            total_size += f.size

    console.print(table)
    console.print(f"\nTotal: {len(files)} files, {format_bytes(total_size)}")
def command_info(client: ModelScopeClient, args: argparse.Namespace) -> int:
    try:
        info = client.get_model_info(args.model_id)
    except ModelScopeError as exc:
        console.print(f"[red]Error fetching metadata:[/red] {exc}")
        return 1

    table = Table(title=f"Model Info: {info.name}", show_header=False, box=None)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")

    table.add_row("Description", info.description[:200] + ("..." if len(info.description) > 200 else ""))
    table.add_row("Tasks", ", ".join(info.tasks) if info.tasks else "None")
    table.add_row("Stars", str(info.stars))
    table.add_row("Storage Size", format_bytes(info.storage_size))
    table.add_row("Created At", info.created_at)
    table.add_row("Modified At", info.modified_at)

    console.print(table)
    return 0


def command_search(client: ModelScopeClient, args: argparse.Namespace) -> int:
    try:
        results = client.search_models(
            keyword=args.keyword,
            page_size=args.page_size,
            page_number=args.page_number,
        )
    except ModelScopeError as exc:
        console.print(f"[red]Error searching models:[/red] {exc}")
        return 1

    if not results:
        console.print(f"[yellow]No models found matching '{args.keyword}'.[/yellow]")
        return 0

    table = Table(title=f"Search Results for '{args.keyword}' (Page {args.page_number})", box=None)
    table.add_column("Model ID", style="cyan")
    table.add_column("Task", style="magenta")
    table.add_column("Stars", justify="right", style="yellow")
    table.add_column("Downloads", justify="right", style="green")
    table.add_column("Size", justify="right", style="blue")
    table.add_column("Last Updated", style="dim")

    for m in results:
        size_str = format_bytes(m.storage_size) if m.storage_size is not None else "unknown"
        updated_str = (
            datetime.fromtimestamp(m.last_modified).strftime("%Y-%m-%d")
            if m.last_modified
            else "unknown"
        )
        task_str = m.task if m.task else "-"
        table.add_row(
            m.model_id,
            task_str,
            str(m.stars),
            str(m.downloads),
            size_str,
            updated_str,
        )

    console.print(table)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    
    # 不影响需要纯文本处理的命令输出，仅在输出是终端时打印 Logo。
    # 官方客户端是直接 print 的，导致窄屏幕下会换行乱码。
    # 我们这里做个优化：只有当终端宽度足够（>= 105）时才打印，保持优雅。
    if sys.stdout.isatty() and not args.simple_progress:
        if console.width >= 105:
            console.print(LOGO, style="bold blue", highlight=False, soft_wrap=True)

    client = ModelScopeClient(
        endpoint=args.endpoint,
        token=args.token,
        verify_ssl=not args.no_verify_ssl,
        timeout=args.timeout,
    )

    try:
        if args.command == "download":
            local_dir = Path(args.local_dir)
            if args.filename:
                return download_one(client, args, args.filename, local_dir / args.filename, local_dir)
            return download_repo(client, args, local_dir)

        if args.command == "ls":
            return command_ls(client, args)

        if args.command == "info":
            return command_info(client, args)

        if args.command == "search":
            return command_search(client, args)

        if args.command == "legacy" and args.file:
            output = Path(args.output)
            if output.exists() and output.is_dir():
                output = output / Path(args.file).name
            elif str(args.output).endswith(("/", "\\")):
                output.mkdir(parents=True, exist_ok=True)
                output = output / Path(args.file).name
            return download_one(client, args, args.file, output, output.parent)

        if args.command == "legacy":
            return download_repo(client, args, Path(args.output))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except ModelScopeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
