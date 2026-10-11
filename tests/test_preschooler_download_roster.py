from types import SimpleNamespace

from scripts.benchmarks.preschooler_download_roster import missing_bytes, selected_files


def test_ministral_snapshot_keeps_tekken_tokenizer():
    info = SimpleNamespace(id="mistralai/Ministral-3-3B-Instruct-2512",
                           siblings=[SimpleNamespace(rfilename=name, size=1)
                                     for name in ("model.safetensors", "tokenizer.json",
                                                  "tekken.json", "README.md")])
    assert {entry.rfilename for entry in selected_files(info, "model.safetensors")} == {
        "model.safetensors", "tokenizer.json", "tekken.json", "README.md"}


def test_existing_manifest_does_not_hide_missing_tokenizer(tmp_path):
    files = [SimpleNamespace(rfilename="model.safetensors", size=3),
             SimpleNamespace(rfilename="tekken.json", size=5)]
    (tmp_path / "model.safetensors").write_bytes(b"abc")
    (tmp_path / "snapshot-manifest.json").write_text("{}")
    assert missing_bytes(tmp_path, files) == 5
    (tmp_path / "tekken.json").write_bytes(b"short")
    assert missing_bytes(tmp_path, files) == 0
