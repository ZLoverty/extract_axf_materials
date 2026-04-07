# extract-axf-materials

Extract material maps and color information from AXF files.

## Install

1. Create a virtual environment and activate it.

```
python -m venv .venv
.venv\Scripts\activate
```

2. Install `extract-axf-materials` 

```powershell
pip install .
```

For editable development install:

```powershell
pip install -e .
```

## CLI

After installation, use:

```powershell
extract-axf-materials "C:\path\to\sample.axf"
```

Or run it on a folder:

```powershell
extract-axf-materials "C:\path\to\folder"
```

Optional threshold:

```powershell
extract-axf-materials "C:\path\to\folder" --size-threshold 128
```

## Outputs

For each AXF file, the tool creates a sibling output folder named after the AXF stem and writes:

- `BaseColor.png` or `BaseColor_preview.png`
- `Normal.png`
- `Height.png`
- `Roughness.png` or `Roughness_scalar.png`
- `color_info.json`
