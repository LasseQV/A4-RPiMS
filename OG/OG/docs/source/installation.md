# Installation

## From PyPI

```bash
pip install vismatch
# or faster with uv
uv pip install vismatch
```

## From Source (development)

```bash
git clone --recursive https://github.com/gmberton/vismatch
cd vismatch
pip install -e .
```

## Optional Dependencies

Some models require extra dependencies not included by default:

```bash
pip install vismatch[all]
```

This includes `torch-geometric` (SphereGlue) and `tensorflow`/`larq` (OmniGlue/ZippyPoint).
