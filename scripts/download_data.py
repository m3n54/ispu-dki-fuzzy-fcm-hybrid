"""Download genuine public ISPU DKI Jakarta dataset from public open-data mirrors."""
from __future__ import annotations

import io
import json
from pathlib import Path
import requests
import pandas as pd


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

CANDIDATE_URLS = [
    (
        "dhiboen_project_ispu",
        "https://raw.githubusercontent.com/dhiboen/Project/main/ISPU.csv",
    ),
]

GITHUB_REPO_CANDIDATES = [
    "aryapangestu/Spada_Klasfikasi-Kualitas-Udara",
    "Rendika7/NLP_ISPU_DKI_JAKARTA_2021",
]


def probe_url(url: str, timeout: int = 20) -> tuple[bool, bytes]:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code == 200 and len(resp.content) > 500:
        return True, resp.content
    return False, b""


def discover_github_csvs(repo: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for branch in ("main", "master"):
        api_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
        try:
            r = requests.get(api_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                tree = r.json().get("tree", [])
                for item in tree:
                    p = item.get("path", "")
                    if p.lower().endswith(".csv"):
                        raw = f"https://raw.githubusercontent.com/{repo}/{branch}/{p}"
                        found.append((f"{repo.replace('/', '_')}_{Path(p).name}", raw))
                if found:
                    break
        except Exception:
            continue
    return found


def search_github_ispu_repos() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    try:
        r = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": "ISPU DKI Jakarta csv", "sort": "stars", "per_page": 8},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code == 200:
            for item in r.json().get("items", []):
                found.extend(discover_github_csvs(item["full_name"]))
    except Exception:
        pass
    return found


def inspect_csv_bytes(content: bytes) -> dict:
    for sep in (",", ";"):
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep)
            if df.shape[1] >= 5:
                return {
                    "rows": int(df.shape[0]),
                    "cols": list(df.columns),
                    "sep": sep,
                    "head": df.head(3).to_dict(orient="records"),
                }
        except Exception:
            continue
    return {"rows": 0, "cols": []}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    candidates = list(CANDIDATE_URLS)
    for repo in GITHUB_REPO_CANDIDATES:
        candidates.extend(discover_github_csvs(repo))

    downloaded = []
    for name, url in candidates:
        try:
            ok, content = probe_url(url)
            if ok:
                info = inspect_csv_bytes(content)
                print(f"[OK] {name} <- {url} | rows={info['rows']} cols={info['cols']}")
                if info["rows"] >= 100:
                    out_path = RAW_DIR / (name if name.endswith(".csv") else f"{name}.csv")
                    out_path.write_bytes(content)
                    downloaded.append({"name": name, "url": url, **info})
        except Exception as exc:
            print(f"[ERR] {name}: {exc}")

    if not downloaded:
        print("Searching additional GitHub repositories...")
        for name, url in search_github_ispu_repos():
            try:
                ok, content = probe_url(url)
                if ok:
                    info = inspect_csv_bytes(content)
                    print(f"[OK] {name} <- {url} | rows={info['rows']} cols={info['cols']}")
                    if info["rows"] >= 100:
                        out_path = RAW_DIR / (name if name.endswith(".csv") else f"{name}.csv")
                        out_path.write_bytes(content)
                        downloaded.append({"name": name, "url": url, **info})
            except Exception as exc:
                print(f"[ERR] {name}: {exc}")

    meta_path = RAW_DIR / "download_manifest.json"
    meta_path.write_text(json.dumps(downloaded, indent=2, default=str), encoding="utf-8")
    print(f"Saved {len(downloaded)} dataset file(s) to {RAW_DIR}")


if __name__ == "__main__":
    main()
