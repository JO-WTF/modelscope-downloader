from msdl.client import ModelScopeClient, RepoFile, filter_files


def test_file_url_quotes_path_but_keeps_model_separator():
    client = ModelScopeClient()

    url = client.file_url("Qwen/Qwen2.5-0.5B", "dir/file name.json", "main")

    assert url == "https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B/resolve/main/dir/file%20name.json"


def test_filter_files_include_and_exclude():
    files = [
        RepoFile("config.json"),
        RepoFile("model.safetensors"),
        RepoFile("model.bin"),
        RepoFile("README.md"),
    ]

    selected = filter_files(files, include=["*.json", "*.safetensors", "*.bin"], exclude=["*.bin"])

    assert [file.path for file in selected] == ["config.json", "model.safetensors"]


def test_parse_file_list_from_nested_payload():
    client = ModelScopeClient()
    payload = {
        "Data": {
            "Files": [
                {"Path": "config.json", "Type": "blob", "Size": 10},
                {"Path": "weights", "Type": "tree"},
                {"Path": "weights/a.safetensors", "Type": "file"},
            ]
        }
    }

    files = client._parse_file_list(payload)

    assert files == [RepoFile("config.json", 10), RepoFile("weights/a.safetensors", None)]
