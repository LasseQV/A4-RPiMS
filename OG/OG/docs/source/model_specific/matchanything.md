# MatchAnything (ELoFTR / RoMa)

The `matchanything-eloftr` and `matchanything-roma` wrappers use the upstream MatchAnything repo (HF Space: https://huggingface.co/spaces/LittleFrog/MatchAnything), included here as a git submodule at `vismatch/third_party/MatchAnything`.

## Submodule setup

If you cloned without submodules:

```bash
git submodule update --init --recursive vismatch/third_party/MatchAnything
```

## Use

Run either variant via:
```bash
# ELoFTR backbone (defaults to 832px NPE size)
python vismatch_match.py --matcher matchanything-eloftr --device cuda --img-size 832 --out-dir outputs_matchanything-eloftr

# RoMa backbone (AMP disabled on CPU automatically)
python vismatch_match.py --matcher matchanything-roma --device cuda --img-size 832 --out-dir outputs_matchanything-roma
```
Weights download automatically on first MatchAnything use and are cached in the HF Cache.

## Weights cache location

Checkpoints are cached in the HF_CACHE, usually `~/.cache/huggingface/hub`:

The wrapper will also reuse checkpoints previously downloaded to the legacy location under the MatchAnything submodule.