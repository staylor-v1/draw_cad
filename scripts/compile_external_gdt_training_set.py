"""Compile provenance-rich external GD&T drawing/reference assets.

The script intentionally favors sources with usable redistribution terms:
NIST public information/public-domain materials and Wikimedia Commons files
with explicit per-file licenses. Third-party leads are recorded in the dataset
README for manual permission review rather than downloaded blindly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import cairosvg
except Exception:  # pragma: no cover - optional but expected in local env
    cairosvg = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "training_data/gdt_external"
USER_AGENT = "draw-cad-gdt-training-compiler/1.0 (+https://github.com)"

NIST_PMI_DRAWINGS = [
    ("nist_pmi_ctc_01", "https://www.nist.gov/document/nist-cad-model-ctc-01", "Complex Test Case 1"),
    ("nist_pmi_ctc_02", "https://www.nist.gov/document/nist-cad-model-ctc-02", "Complex Test Case 2"),
    ("nist_pmi_ctc_03", "https://www.nist.gov/document/nist-cad-model-ctc-03", "Complex Test Case 3"),
    ("nist_pmi_ctc_04", "https://www.nist.gov/document/nist-cad-model-ctc-04", "Complex Test Case 4"),
    ("nist_pmi_ctc_05", "https://www.nist.gov/document/nist-cad-model-ctc-05", "Complex Test Case 5"),
    ("nist_pmi_ftc_06", "https://www.nist.gov/document/nist-cad-model-ftc-06", "Fully Toleranced Test Case 6"),
    ("nist_pmi_ftc_07", "https://www.nist.gov/document/nist-cad-model-ftc-07", "Fully Toleranced Test Case 7"),
    ("nist_pmi_ftc_08", "https://www.nist.gov/document/nist-cad-model-ftc-08", "Fully Toleranced Test Case 8"),
    ("nist_pmi_ftc_09", "https://www.nist.gov/document/nist-cad-model-ftc-09", "Fully Toleranced Test Case 9"),
    ("nist_pmi_ftc_10", "https://www.nist.gov/document/nist-cad-model-ftc-10", "Fully Toleranced Test Case 10"),
    ("nist_pmi_ftc_11", "https://www.nist.gov/document/nist-cad-model-ftc-11", "Fully Toleranced Test Case 11"),
]

NIST_AM_DRAWINGS = [
    (
        "nist_am_test_artifact_engineering_drawing",
        "https://www.nist.gov/document/engineeringdrawingpdf",
        "NIST additive manufacturing test artifact engineering drawing",
    ),
]

COMMONS_CATEGORIES = [
    "Category:Geometric dimensioning and tolerancing",
    "Category:Geometric tolerancing",
]

REVIEW_ONLY_SOURCES = [
    {
        "name": "University of Illinois ME170 Engineering Drawing Notes",
        "url": "https://courses.grainger.illinois.edu/me170/fa2019/Engineering_Drawing_Notes_B.pdf",
        "reason": "Useful GD&T teaching drawings, but redistribution/license terms were not explicit in the search result.",
    },
    {
        "name": "GD&T Basics feature control frame/datum examples",
        "url": "https://www.gdandtbasics.com/feature-control-frame/",
        "reason": "High-quality examples for visual review; use as design reference unless permission is granted.",
    },
    {
        "name": "KEYENCE GD&T overview examples",
        "url": "https://www.keyence.com/ss/products/measure-sys/gd-and-t/basic/tolerance-entry-frame.jsp",
        "reason": "High-quality commercial reference examples; not downloaded into training data without permission.",
    },
    {
        "name": "NASA internship final report GD&T discussion",
        "url": "https://ntrs.nasa.gov/api/citations/20180002822/downloads/20180002822.pdf",
        "reason": "Public technical report with GD&T context, but it does not appear to be a dense drawing corpus.",
    },
]


@dataclass
class AssetRecord:
    id: str
    group: str
    role: str
    title: str
    source_url: str
    page_url: str
    license: str
    license_url: str
    local_path: str
    rendered_paths: list[str]
    sha256: str
    bytes: int
    notes: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile external GD&T training assets.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "raw/nist_pmi").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw/nist_am").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw/wikimedia_commons").mkdir(parents=True, exist_ok=True)
    (output_dir / "rendered/nist_pmi").mkdir(parents=True, exist_ok=True)
    (output_dir / "rendered/nist_am").mkdir(parents=True, exist_ok=True)
    (output_dir / "rendered/wikimedia_commons").mkdir(parents=True, exist_ok=True)

    records: list[AssetRecord] = []
    for asset_id, url, title in NIST_PMI_DRAWINGS:
        records.append(
            download_pdf_asset(
                asset_id=asset_id,
                group="nist_pmi",
                role="full_drawing",
                title=title,
                url=url,
                output_dir=output_dir,
                dpi=args.dpi,
                notes=[
                    "NIST MBE PMI validation/conformance test-case drawing",
                    "Good source for feature control frames, datum features, datum targets, dimensions, and title blocks",
                ],
            )
        )

    for asset_id, url, title in NIST_AM_DRAWINGS:
        records.append(
            download_pdf_asset(
                asset_id=asset_id,
                group="nist_am",
                role="full_drawing",
                title=title,
                url=url,
                output_dir=output_dir,
                dpi=args.dpi,
                notes=[
                    "NIST public-domain additive-manufacturing test artifact drawing",
                    "Useful for full-sheet drawing segmentation with GD&T markings",
                ],
            )
        )

    records.extend(download_commons_assets(output_dir))
    manifest = {
        "name": "external_gdt_training_set",
        "createdBy": "scripts/compile_external_gdt_training_set.py",
        "sourcePolicy": {
            "included": "NIST public information/public-domain assets and Wikimedia Commons files with explicit license metadata.",
            "excluded": "Commercial/blog/university examples without clear redistribution terms are listed for review only.",
        },
        "counts": {
            "assets": len(records),
            "renderedImages": sum(len(record.rendered_paths) for record in records),
            "nistPdfs": sum(1 for record in records if record.group.startswith("nist")),
            "commonsFiles": sum(1 for record in records if record.group == "wikimedia_commons"),
        },
        "assets": [asdict(record) for record in records],
        "reviewOnlySources": REVIEW_ONLY_SOURCES,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(render_readme(manifest), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))


def download_pdf_asset(
    *,
    asset_id: str,
    group: str,
    role: str,
    title: str,
    url: str,
    output_dir: Path,
    dpi: int,
    notes: list[str],
) -> AssetRecord:
    raw_dir = output_dir / "raw" / group
    rendered_dir = output_dir / "rendered" / group
    local_path = raw_dir / f"{asset_id}.pdf"
    download(url, local_path)
    rendered_paths = render_pdf(local_path, rendered_dir / asset_id, dpi)
    return AssetRecord(
        id=asset_id,
        group=group,
        role=role,
        title=title,
        source_url=url,
        page_url=url,
        license="NIST public information/public domain unless marked otherwise",
        license_url="https://www.nist.gov/copyrights-disclaimers",
        local_path=rel(local_path),
        rendered_paths=[rel(path) for path in rendered_paths],
        sha256=sha256(local_path),
        bytes=local_path.stat().st_size,
        notes=notes,
    )


def download_commons_assets(output_dir: Path) -> list[AssetRecord]:
    titles: list[str] = []
    for category in COMMONS_CATEGORIES:
        titles.extend(commons_category_titles(category))
    unique_titles = sorted(set(titles))
    records: list[AssetRecord] = []
    for title, info in commons_file_infos(unique_titles).items():
        try:
            original_url = info["url"]
            filename = safe_name(title.removeprefix("File:"))
            suffix = Path(urllib.parse.urlparse(original_url).path).suffix or Path(filename).suffix
            if suffix:
                filename = f"{Path(filename).stem}{suffix.lower()}"
            local_path = output_dir / "raw/wikimedia_commons" / filename
            download(original_url, local_path)
            rendered_paths = render_commons_asset(local_path, output_dir / "rendered/wikimedia_commons")
            metadata = info.get("extmetadata", {})
            license_short = clean_metadata(metadata.get("LicenseShortName", {}).get("value", "unknown"))
            license_url = clean_metadata(metadata.get("LicenseUrl", {}).get("value", ""))
            description = clean_metadata(metadata.get("ObjectName", {}).get("value", title))
            records.append(
                AssetRecord(
                    id=f"commons_{Path(filename).stem}",
                    group="wikimedia_commons",
                    role=classify_commons_role(title),
                    title=description or title,
                    source_url=original_url,
                    page_url=f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                    license=license_short,
                    license_url=license_url,
                    local_path=rel(local_path),
                    rendered_paths=[rel(path) for path in rendered_paths],
                    sha256=sha256(local_path),
                    bytes=local_path.stat().st_size,
                    notes=[
                        "Wikimedia Commons file from GD&T/geometric tolerancing categories",
                        "Use per-file license metadata in manifest before redistribution outside this project",
                    ],
                )
            )
        except Exception as exc:
            print(f"warning: skipped {title}: {exc}", file=sys.stderr)
    return records


def commons_category_titles(category: str) -> list[str]:
    titles: list[str] = []
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "file",
        "cmlimit": "500",
    }
    while True:
        data = request_json("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params))
        titles.extend(item["title"] for item in data["query"]["categorymembers"])
        if "continue" not in data:
            break
        params.update(data["continue"])
    return titles


def commons_file_infos(titles: list[str]) -> dict[str, dict[str, Any]]:
    infos: dict[str, dict[str, Any]] = {}
    for index in range(0, len(titles), 25):
        batch = titles[index : index + 25]
        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime",
        }
        data = request_json("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params))
        for page in data["query"]["pages"].values():
            title = page.get("title")
            if title and page.get("imageinfo"):
                infos[title] = page["imageinfo"][0]
        time.sleep(0.8)
    return infos


def request_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with open_url(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with open_url(req, timeout=120) as response:
        data = response.read()
    path.write_bytes(data)
    time.sleep(0.25)


def open_url(req: urllib.request.Request, *, timeout: int):
    for attempt in range(5):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable retry loop")


def render_pdf(pdf_path: Path, output_prefix: Path, dpi: int) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(output_prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return sorted(output_prefix.parent.glob(f"{output_prefix.name}-*.png"))


def render_commons_asset(local_path: Path, rendered_dir: Path) -> list[Path]:
    rendered_dir.mkdir(parents=True, exist_ok=True)
    suffix = local_path.suffix.lower()
    if suffix == ".svg" and cairosvg is not None:
        target = rendered_dir / f"{local_path.stem}.png"
        if not target.exists():
            cairosvg.svg2png(url=str(local_path), write_to=str(target), output_width=1400)
        return [target]
    if suffix in {".png", ".jpg", ".jpeg"}:
        target = rendered_dir / local_path.name
        if not target.exists():
            shutil.copy2(local_path, target)
        return [target]
    return []


def classify_commons_role(title: str) -> str:
    lowered = title.lower()
    if "gdt" in lowered or "gd&t" in lowered or "tolerancing" in lowered or "toleranz" in lowered:
        if any(token in lowered for token in ("flatness", "position", "perpendicularity", "profile", "runout", "circularity")):
            return "symbol_reference"
        return "callout_reference"
    if "cotation gps" in lowered or "technical drawing" in lowered or "schneckenwelle" in lowered:
        return "drawing_reference"
    return "reference"


def render_readme(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    review = "\n".join(
        f"- [{item['name']}]({item['url']}): {item['reason']}" for item in manifest["reviewOnlySources"]
    )
    return textwrap.dedent(
        f"""\
        # External GD&T Training Set

        This directory is a provenance-rich seed set for GD&T detection and segmentation work.
        It was compiled by `scripts/compile_external_gdt_training_set.py`.

        ## Contents

        - Assets: {counts['assets']}
        - Rendered images: {counts['renderedImages']}
        - NIST PDF drawings: {counts['nistPdfs']}
        - Wikimedia Commons files: {counts['commonsFiles']}

        ## Source Policy

        Included assets come from NIST public information/public-domain pages or Wikimedia
        Commons files with explicit per-file license metadata. The manifest records the
        source URL, local path, license label, license URL, SHA-256, and rendered derivatives
        for every item.

        This set is not yet hand-labelled for YOLO. It is ready for teacher review,
        auto-proposal, and annotation. Full-sheet NIST pages are especially valuable for
        dense feature-control-frame, datum, basic-dimension, and title-block examples.
        Wikimedia assets are best used as symbol/reference material and synthetic
        augmentation seeds.

        ## Review-Only Leads

        These sources looked useful during search, but were not downloaded because the
        redistribution/training-data terms were not clear enough for an included corpus:

        {review}

        ## Rebuild

        ```bash
        .venv/bin/python scripts/compile_external_gdt_training_set.py --clean
        ```
        """
    ).lstrip()


def clean_metadata(value: str) -> str:
    return " ".join(str(value).replace("<span class=\"licensetpl_short\">", "").replace("</span>", "").split())


def safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in name).strip("_")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
