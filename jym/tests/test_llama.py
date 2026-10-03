import huggingface_hub

from jym import llama


def test_hub_models_download_once(tmp_path, monkeypatch):
    monkeypatch.setenv("JYM_MODELS", str(tmp_path))
    downloads = []

    def download(repo, filename, revision, local_dir):
        downloads.append((repo, filename, revision))
        local_dir.mkdir(parents=True)
        (local_dir / filename).write_bytes(b"GGUF")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    model = llama.Gguf(name="model", file="model-Q8_0.gguf", repo="owner/repo", revision="0123456789abcdef")
    assert model.fetch() == model.fetch() == tmp_path / "owner" / "repo" / "model-Q8_0.gguf"
    assert downloads == [("owner/repo", "model-Q8_0.gguf", "0123456789abcdef")]
    assert model.identity() == "owner/repo@01234567/model-Q8_0.gguf"  # known without the file
