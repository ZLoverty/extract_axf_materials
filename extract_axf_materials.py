"""
AXF material extractor.

Usage
-----
1. Extract a single AXF file:
   python extract_axf_materials.py "C:\\path\\to\\sample.axf"

2. Extract every AXF file in a folder:
   python extract_axf_materials.py "C:\\path\\to\\folder"

3. Override the texture size threshold:
   python extract_axf_materials.py "C:\\path\\to\\folder" --size-threshold 128

Outputs
-------
- BaseColor.png or BaseColor_preview.png
- Normal.png
- Height.png
- Roughness.png or Roughness_scalar.png
- color_info.json

Notes
-----
- Spectral color textures with 31 channels are exported as preview RGB images.
- Scalar roughness values are exported as a 1x1 grayscale PNG.
- Output files are written to a folder next to each AXF file, using the AXF stem as folder name.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def normalize_preview(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = np.where(finite, arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    max_val = float(arr.max())
    if max_val <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = np.clip(arr / max_val, 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def coerce_image_layout(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4, 31):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def spectral_to_rgb(spectral: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(spectral, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)

    if arr.size >= 31:
        blue = float(arr[0:10].mean())
        green = float(arr[10:20].mean())
        red = float(arr[20:31].mean())
    elif arr.size >= 3:
        blue, green, red = map(float, arr[:3])
    else:
        v = float(arr.mean())
        return (v, v, v)

    scale = max(red, green, blue, 1e-8)
    return (red / scale, green / scale, blue / scale)


def infer_tag(dataset_path: str) -> str:
    path = dataset_path.lower()
    if not path.endswith("/data"):
        return "Unknown"
    if "normal/data" in path:
        return "Normal"
    if "height/data" in path or "displacementfilter/height/data" in path or "displacement/data" in path:
        return "Height"
    if "roughness/data" in path or "gloss/data" in path or "lobes/data" in path:
        return "Roughness"
    if "diffusemodel/color_1/data" in path or "diffusemodel/color/data" in path:
        return "BaseColor"
    return "Unknown"


def to_exportable_image(data: np.ndarray, semantic_tag: str) -> tuple[np.ndarray, bool]:
    arr = coerce_image_layout(data)
    preview_only = False

    if arr.ndim == 3 and arr.shape[-1] == 31:
        preview_only = True
        arr = arr[..., 10:13]

    if arr.dtype == np.uint8:
        out = arr
    elif semantic_tag == "Normal":
        arr = np.asarray(arr, dtype=np.float32)
        if arr.min() >= -1.01 and arr.max() <= 1.01:
            out = np.clip((arr * 0.5 + 0.5) * 255.0, 0, 255).round().astype(np.uint8)
        elif arr.min() >= 0.0 and arr.max() <= 1.0:
            out = np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
        else:
            out = normalize_preview(arr)
            preview_only = True
    elif semantic_tag in {"Height", "Roughness", "BaseColor"}:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.min() >= 0.0 and arr.max() <= 1.0:
            out = np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
        else:
            out = normalize_preview(arr)
            preview_only = True
    else:
        out = normalize_preview(arr)
        preview_only = True

    if out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]

    return out, preview_only


def save_scalar_png(value: float, out_path: Path) -> None:
    pixel = np.clip(round(float(value) * 255.0), 0, 255)
    Image.fromarray(np.array([[pixel]], dtype=np.uint8)).save(out_path)


def extract_material_assets(axf_file: str | Path, size_threshold: int = 200) -> dict:
    f_path = Path(axf_file)
    out_dir = f_path.parent / f_path.stem
    out_dir.mkdir(exist_ok=True)

    color_info = {
        "source_file": str(f_path),
        "resource_roots": [],
        "base_color_rgb": None,
        "base_color_rgb_255": None,
        "maps": {},
        "scalars": {},
        "notes": [],
    }

    exported: dict[str, dict] = {}

    with h5py.File(f_path, "r") as f:
        def process_node(name: str, obj) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if not np.issubdtype(obj.dtype, np.number):
                return
            if "Measurements" in name:
                return

            semantic_tag = infer_tag(name)
            if semantic_tag == "Unknown":
                return

            shape = obj.shape
            color_info["resource_roots"].append(name)
            data = obj[:]

            if obj.ndim >= 2 and any(dim >= size_threshold for dim in shape):
                image_data, preview_only = to_exportable_image(data, semantic_tag)
                current_area = int(image_data.shape[0] * image_data.shape[1])
                previous_area = exported.get(semantic_tag, {}).get("area", -1)

                if current_area > previous_area:
                    suffix = "_preview" if preview_only else ""
                    file_name = f"{semantic_tag}{suffix}.png"
                    Image.fromarray(image_data).save(out_dir / file_name)
                    exported[semantic_tag] = {"file": file_name, "area": current_area}
                    color_info["maps"][semantic_tag] = {
                        "file": file_name,
                        "dataset": name,
                        "shape": list(image_data.shape),
                        "preview_only": preview_only,
                    }

                    if semantic_tag == "BaseColor":
                        mean_rgb = np.asarray(image_data, dtype=np.float32)
                        if mean_rgb.ndim == 2:
                            rgb = [float(mean_rgb.mean() / 255.0)] * 3
                        else:
                            rgb = (mean_rgb[..., :3].reshape(-1, 3).mean(axis=0) / 255.0).tolist()
                        color_info["base_color_rgb"] = [round(float(c), 6) for c in rgb]
                        color_info["base_color_rgb_255"] = [int(round(float(c) * 255.0)) for c in rgb]

            elif semantic_tag == "BaseColor" and np.asarray(data).ndim == 1 and color_info["base_color_rgb"] is None:
                rgb = spectral_to_rgb(data)
                color_info["base_color_rgb"] = [round(float(c), 6) for c in rgb]
                color_info["base_color_rgb_255"] = [int(round(float(c) * 255.0)) for c in rgb]
                color_info["notes"].append(f"BaseColor derived from spectral dataset: {name}")

            elif semantic_tag == "Roughness" and np.asarray(data).size == 1 and "Roughness" not in color_info["maps"]:
                scalar = float(np.asarray(data).reshape(-1)[0])
                if 0.0 <= scalar <= 1.0:
                    file_name = "Roughness_scalar.png"
                    save_scalar_png(scalar, out_dir / file_name)
                    color_info["maps"]["Roughness"] = {
                        "file": file_name,
                        "dataset": name,
                        "shape": [1, 1],
                        "preview_only": False,
                    }
                    color_info["scalars"]["Roughness"] = scalar

        f.visititems(process_node)

        if color_info["base_color_rgb"] is None:
            for name, obj in f.items():
                if name.startswith("com.xrite.Materials"):
                    dataset_path = f"{name}/com.xrite.Resources/DiffuseModel/Color/Data"
                    if dataset_path in f and np.issubdtype(f[dataset_path].dtype, np.number):
                        rgb = spectral_to_rgb(f[dataset_path][:])
                        color_info["base_color_rgb"] = [round(float(c), 6) for c in rgb]
                        color_info["base_color_rgb_255"] = [int(round(float(c) * 255.0)) for c in rgb]
                        color_info["notes"].append(
                            f"BaseColor derived from fallback spectral dataset: {dataset_path}"
                        )
                        break

    color_info["resource_roots"] = sorted(set(color_info["resource_roots"]))
    (out_dir / "color_info.json").write_text(
        json.dumps(color_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"提取完成: {out_dir}")
    return color_info


def iter_axf_files(target: str | Path) -> list[Path]:
    path = Path(target)
    if path.is_file():
        return [path]
    return sorted(path.glob("*.axf"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract material maps and color info from AXF files.",
        epilog=(
            "Examples:\n"
            "  python extract_axf_materials.py \"C:\\path\\to\\sample.axf\"\n"
            "  python extract_axf_materials.py \"C:\\path\\to\\folder\"\n"
            "  python extract_axf_materials.py \"C:\\path\\to\\folder\" --size-threshold 128"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("target", help="Path to an .axf file or a folder containing .axf files.")
    parser.add_argument(
        "--size-threshold",
        type=int,
        default=128,
        help="Minimum width/height threshold for exporting 2D texture datasets.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    files = iter_axf_files(args.target)
    if not files:
        raise SystemExit("No .axf files found.")

    for axf_file in files:
        extract_material_assets(axf_file, size_threshold=args.size_threshold)


if __name__ == "__main__":
    main()
