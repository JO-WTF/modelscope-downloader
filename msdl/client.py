from __future__ import annotations

import fnmatch
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote

import requests
import urllib3


DEFAULT_ENDPOINT = "https://www.modelscope.cn"
ProgressCallback = Callable[[str, int, int | None, int], None]


class ModelScopeError(RuntimeError):
    """Raised when ModelScope requests cannot be completed."""


@dataclass(frozen=True)
class RepoFile:
    path: str
    size: int | None = None


class ModelScopeClient:
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        token: str | None = None,
        verify_ssl: bool = True,
        timeout: int = 30,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.session = requests.Session()
        self.headers: dict[str, str] = {}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
            self.session.headers.update(self.headers)
        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def file_url(self, model_id: str, file_path: str, revision: str = "master") -> str:
        model = quote(model_id.strip("/"), safe="/")
        rev = quote(revision, safe="")
        path = quote(file_path.strip("/"), safe="/")
        return f"{self.endpoint}/models/{model}/resolve/{rev}/{path}"

    def list_files(self, model_id: str, revision: str = "master") -> list[RepoFile]:
        errors: list[str] = []
        for url, params in self._list_file_candidates(model_id, revision):
            try:
                resp = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    headers={"Accept": "application/json"},
                )
            except requests.RequestException as exc:
                errors.append(f"{url}: {exc}")
                continue
            if resp.status_code == 404:
                errors.append(f"{resp.url}: 404")
                continue
            if not resp.ok:
                errors.append(f"{resp.url}: HTTP {resp.status_code} {resp.text[:200]}")
                continue
            try:
                files = self._parse_file_list(resp.json())
            except (ValueError, TypeError) as exc:
                errors.append(f"{resp.url}: {exc}")
                continue
            if files:
                return files
        detail = "\n".join(errors) if errors else "no API candidates were tried"
        raise ModelScopeError(
            "Could not list repository files from ModelScope. "
            "Use --file for a single file download, or update the API candidate list.\n"
            f"{detail}"
        )

    def download_file(
        self,
        model_id: str,
        file_path: str,
        destination: Path,
        revision: str = "master",
        chunk_size: int = 1024 * 1024,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_name(destination.name + ".part")
        headers: dict[str, str] = {}
        existing = temp_path.stat().st_size if temp_path.exists() else 0
        if existing:
            headers["Range"] = f"bytes={existing}-"

        url = self.file_url(model_id, file_path, revision)
        request_headers = {**self.headers, **headers}
        session = requests.Session()
        with session.get(
            url,
            stream=True,
            timeout=self.timeout,
            verify=self.verify_ssl,
            headers=request_headers,
        ) as resp:
            if resp.status_code == 416:
                temp_path.replace(destination)
                if progress_callback:
                    progress_callback(file_path, existing, existing, 0)
                return destination
            if existing and resp.status_code != 206:
                existing = 0
            if not resp.ok:
                raise ModelScopeError(f"Download failed for {file_path}: HTTP {resp.status_code} {resp.text[:300]}")

            mode = "ab" if existing and resp.status_code == 206 else "wb"
            total = _content_length(resp, existing)
            downloaded = existing
            if progress_callback:
                progress_callback(file_path, downloaded, total, 0)
            with temp_path.open(mode + "") as fh:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(file_path, downloaded, total, len(chunk))

        if total and temp_path.stat().st_size != total:
            raise ModelScopeError(
                f"Integrity check failed for {file_path}: expected {total} bytes, "
                f"got {temp_path.stat().st_size} bytes"
            )
        os.replace(temp_path, destination)
        return destination

    def download_snapshot(
        self,
        model_id: str,
        output_dir: Path,
        revision: str = "master",
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        max_workers: int = 4,
        progress_callback: ProgressCallback | None = None,
    ) -> list[Path]:
        files = filter_files(self.list_files(model_id, revision), include, exclude)
        if max_workers <= 1:
            return [
                self.download_file(
                    model_id,
                    repo_file.path,
                    output_dir / repo_file.path,
                    revision,
                    progress_callback=progress_callback,
                )
                for repo_file in files
            ]

        written: list[Path] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.download_file,
                    model_id,
                    repo_file.path,
                    output_dir / repo_file.path,
                    revision,
                    progress_callback=progress_callback,
                )
                for repo_file in files
            ]
            for future in as_completed(futures):
                written.append(future.result())
        return written

    def _list_file_candidates(self, model_id: str, revision: str) -> Iterable[tuple[str, dict[str, str]]]:
        model = model_id.strip("/")
        quoted_model = quote(model, safe="/")
        parts = model.split("/", 1)
        candidates = [
            (
                f"{self.endpoint}/api/v1/models/{quoted_model}/repo/files",
                {"Revision": revision, "Recursive": "true"},
            ),
            (
                f"{self.endpoint}/api/v1/models/{quoted_model}/repo",
                {"Revision": revision, "FilePath": ""},
            ),
        ]
        if len(parts) == 2:
            owner, name = (quote(p, safe="") for p in parts)
            candidates.extend(
                [
                    (
                        f"{self.endpoint}/api/v1/models/{owner}/{name}/repo/files",
                        {"Revision": revision, "Recursive": "true"},
                    ),
                    (
                        f"{self.endpoint}/api/v1/models/{owner}/{name}/repo",
                        {"Revision": revision, "FilePath": ""},
                    ),
                ]
            )
        return candidates

    def _parse_file_list(self, payload: Any) -> list[RepoFile]:
        items = _find_file_items(payload)
        files: list[RepoFile] = []
        for item in items:
            path = _path_from_item(item)
            if not path:
                continue
            kind = str(item.get("type") or item.get("Type") or item.get("kind") or "").lower()
            if kind in {"tree", "dir", "directory", "folder"}:
                continue
            size = item.get("size") or item.get("Size") or item.get("file_size")
            files.append(RepoFile(path=path, size=int(size) if isinstance(size, int) else None))
        return sorted({f.path: f for f in files}.values(), key=lambda f: f.path)


def filter_files(
    files: Iterable[RepoFile],
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[RepoFile]:
    selected = []
    for repo_file in files:
        if include and not any(fnmatch.fnmatch(repo_file.path, pattern) for pattern in include):
            continue
        if exclude and any(fnmatch.fnmatch(repo_file.path, pattern) for pattern in exclude):
            continue
        selected.append(repo_file)
    return selected


def _find_file_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        raise TypeError("unexpected JSON response")
    for key in ("files", "Files", "data", "Data", "items", "Items", "repoFiles"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            nested = _find_file_items(value)
            if nested:
                return nested
    return []


def _path_from_item(item: dict[str, Any]) -> str | None:
    for key in ("path", "Path", "name", "Name", "filePath", "FilePath", "file_name"):
        value = item.get(key)
        if isinstance(value, str) and value and value not in {".", "/"}:
            return value.strip("/")
    return None


def _content_length(resp: requests.Response, existing: int) -> int | None:
    range_header = resp.headers.get("Content-Range")
    if range_header and "/" in range_header:
        tail = range_header.rsplit("/", 1)[-1]
        if tail.isdigit():
            return int(tail)
    length = resp.headers.get("Content-Length")
    if length and length.isdigit():
        return int(length) + existing
    return None
