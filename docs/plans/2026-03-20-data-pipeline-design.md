# Data Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Download 9 robotics datasets, extract action-change keyframes (5/trajectory), generate depth maps via Depth Anything V2 vitl, output unified RGB+depth+meta format. Stream-process to stay under 1TB disk delta.

**Architecture:** A main orchestrator iterates datasets in priority order. Each dataset has a downloader adapter that yields episodes one-at-a-time. A shared keyframe extractor selects 5 frames per trajectory based on action deltas. A depth generator runs Depth Anything V2 vitl on keyframes lacking native depth. Output is saved to `data/processed/{dataset}/episode_{N}/` with RGB PNGs, depth NPYs, and meta JSON. Source files are deleted immediately after processing.

**Tech Stack:** Python 3.13, huggingface_hub, datasets, Depth Anything V2 (torch), av (video decoding), scipy, PIL, numpy.

---

## Data Source Summary (verified)

| # | Dataset | HF Repo / Source | Format | Raw Size | Has Depth |
|---|---------|-----------------|--------|----------|-----------|
| 1 | RLBench | local sample | PNG+pkl | 36MB | native |
| 2 | ALOHA bimanual | lerobot/aloha_static_* (15 tasks) | parquet+mp4 | ~5GB | no |
| 3 | utokyo_xarm_bimanual | jxu124/OpenX-Embodiment | tar | ~0.04GB | no |
| 4 | berkeley_cable_routing | jxu124/OpenX-Embodiment | tar | 0.7GB | no |
| 5 | taco_play | jxu124/OpenX-Embodiment | tar | 7.9GB | no |
| 6 | furniture_bench | jxu124/OpenX-Embodiment | tar | 59.8GB | no |
| 7 | nyu_franka_play | jxu124/OpenX-Embodiment | tar | 1.3GB | no |
| 8 | jaco_play | jxu124/OpenX-Embodiment | tar | 1.1GB | no |
| 9 | Bridge V2 | jxu124/OpenX-Embodiment (bridge) | tar | 36.8GB | no |
| 10 | fractal (RT-1) | jxu124/OpenX-Embodiment | tar | 58.1GB | no |
| 11 | DROID | cadene/droid | parquet+mp4 | ~475GB | stereo |

## Output Format

```
data/processed/{dataset}/episode_{NNNNNN}/
  rgb_0.png .. rgb_4.png     # 256x256 RGB keyframes
  depth_0.npy .. depth_4.npy # 256x256 float32 meters
  meta.json
```

## Disk Budget: ~420 GB peak (within 1TB)

---

## Task 1: Install Dependencies and Depth Anything V2

**Files:**
- Create: `scripts/data_pipeline/install_deps.sh`

**Step 1: Create install script**

```bash
#!/bin/bash
pip install webdataset av timm
# Depth Anything V2 via transformers pipeline
pip install --upgrade transformers
```

**Step 2: Verify Depth Anything V2**

Run:
```bash
python -c "
from transformers import pipeline
import numpy as np
from PIL import Image
pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Large-hf', device='cuda:1')
img = Image.fromarray(np.random.randint(0,255,(256,256,3), dtype=np.uint8))
r = pipe(img)
print(f'Shape: {np.array(r[\"depth\"]).shape}')
print('OK')
"
```
Expected: prints shape and OK.

**Step 3: Commit**

```bash
git add scripts/data_pipeline/install_deps.sh
git commit -m "feat: data pipeline dependency installer"
```

---

## Task 2: Keyframe Extractor and Disk Monitor

**Files:**
- Create: `scripts/data_pipeline/keyframe_extractor.py`
- Create: `scripts/data_pipeline/disk_monitor.py`
- Create: `scripts/data_pipeline/test_keyframe.py`

**Step 1: Write test**

```python
# test_keyframe.py
import numpy as np
from keyframe_extractor import extract_keyframes

def test_basic():
    actions = np.zeros((60, 7))
    actions[10] = 1.0
    actions[30] = -1.0
    actions[50] = 0.5
    indices = extract_keyframes(actions, num_keyframes=5)
    assert len(indices) == 5
    assert indices[0] == 0
    assert indices[-1] == 59
    print("PASS")

if __name__ == "__main__":
    test_basic()
```

**Step 2: Run test — expect FAIL**

```bash
cd scripts/data_pipeline && python test_keyframe.py
```

**Step 3: Implement keyframe_extractor.py**

Algorithm: first frame + last frame + 3 peaks of smoothed action-delta magnitude with minimum gap.

```python
import numpy as np
from scipy.ndimage import gaussian_filter1d

def extract_keyframes(actions: np.ndarray, num_keyframes: int = 5, min_gap_ratio: float = 1/6) -> list[int]:
    T = len(actions)
    if T <= num_keyframes:
        return list(range(T))
    indices = [0, T - 1]
    num_middle = num_keyframes - 2
    if num_middle <= 0:
        return sorted(indices)
    deltas = np.linalg.norm(np.diff(actions, axis=0), axis=-1)
    if len(deltas) > 7:
        deltas = gaussian_filter1d(deltas.astype(np.float64), sigma=3)
    min_gap = max(int(T * min_gap_ratio), 1)
    used = set(indices)
    for _ in range(num_middle):
        best_idx, best_val = -1, -1.0
        for i in range(len(deltas)):
            if any(abs(i - u) < min_gap for u in used):
                continue
            if deltas[i] > best_val:
                best_val = deltas[i]
                best_idx = i
        if best_idx >= 0:
            frame_idx = min(best_idx + 1, T - 1)
            indices.append(frame_idx)
            used.add(frame_idx)
        else:
            step = T // (num_middle + 1)
            indices.append(step * (len(indices) - 1))
    return sorted(set(indices))[:num_keyframes]
```

**Step 4: Implement disk_monitor.py**

```python
import shutil
def check_disk(path="/home/w50037733", warn_gb=100, critical_gb=50):
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < critical_gb: return "critical"
    if free_gb < warn_gb: return "warn"
    return "ok"
def get_free_gb(path="/home/w50037733"):
    return shutil.disk_usage(path).free / 1e9
```

**Step 5: Run test — expect PASS**

**Step 6: Commit**

---

## Task 3: Depth Generator Module

**Files:**
- Create: `scripts/data_pipeline/depth_generator.py`

**Step 1: Implement**

Uses transformers pipeline with Depth-Anything-V2-Large-hf. Normalizes output to [0.01, 5.0] range for compatibility with backproject_depth. Single-image interface + batch wrapper.

**Step 2: Smoke test on random image**

**Step 3: Commit**

---

## Task 4: Episode Saver (Unified Output)

**Files:**
- Create: `scripts/data_pipeline/episode_saver.py`

Saves rgb_N.png (256x256), depth_N.npy (float32), meta.json per episode directory.

**Step 1: Implement**
**Step 2: Commit**

---

## Task 5: OXE Tar Downloader Adapter

**Files:**
- Create: `scripts/data_pipeline/downloaders/__init__.py`
- Create: `scripts/data_pipeline/downloaders/oxe_tar.py`

Handles 8 OXE subsets: bridge, fractal, taco_play, jaco_play, berkeley_cable_routing, furniture_bench, nyu_franka_play, utokyo_xarm_bimanual.

Downloads one tar at a time from jxu124/OpenX-Embodiment via hf_hub_download, parses webdataset format, yields episode dicts, deletes tar after processing.

**Step 1: Implement with dataset metadata dict**
**Step 2: Commit**

---

## Task 6: DROID Downloader Adapter

**Files:**
- Create: `scripts/data_pipeline/downloaders/droid.py`

Streams from cadene/droid: downloads per-episode parquet (actions/language) + mp4 (video), decodes frames via `av` library, yields episode dicts, cleans up per-episode.

93 chunks x 1000 episodes. Primary camera: exterior_image_1_left.

**Step 1: Implement**
**Step 2: Commit**

---

## Task 7: ALOHA Bimanual Downloader Adapter

**Files:**
- Create: `scripts/data_pipeline/downloaders/aloha.py`

Loads 15 lerobot/aloha_static_* datasets via HF datasets library. Groups rows by episode_index, extracts top camera image + actions.

**Step 1: Implement**
**Step 2: Commit**

---

## Task 8: RLBench Adapter (Local Existing Data)

**Files:**
- Create: `scripts/data_pipeline/downloaders/rlbench_local.py`

Reads existing local RLBench sample data. Has native depth (RGB-encoded PNG decoding). Uses RLBench-specific pkl loading for task descriptions.

**Step 1: Implement**
**Step 2: Commit**

---

## Task 9: Main Orchestrator

**Files:**
- Create: `scripts/data_pipeline/run_pipeline.py`
- Create: `scripts/data_pipeline/__init__.py`

Orchestrates all datasets in priority order. Features:
- Resume via progress.json (skip completed datasets)
- Disk monitoring every 100 episodes
- Per-dataset timing and episode counting
- CLI: --datasets (filter), --resume, --depth_device

Processing order: rlbench -> aloha -> small OXE -> bridge -> furniture_bench -> fractal -> droid

**Step 1: Implement**
**Step 2: Commit**

---

## Task 10: End-to-end Smoke Test

**Step 1: Test with RLBench only**

```bash
python scripts/data_pipeline/run_pipeline.py --datasets rlbench --depth_device cuda:1
```

**Step 2: Verify output**

```bash
ls data/processed/rlbench/episode_000000/
cat data/processed/rlbench/episode_000000/meta.json
```

**Step 3: Test with one small OXE dataset**

```bash
python scripts/data_pipeline/run_pipeline.py --datasets berkeley_cable_routing --depth_device cuda:1
```

**Step 4: Test resume**

```bash
python scripts/data_pipeline/run_pipeline.py --resume --depth_device cuda:1
```

**Step 5: Commit and push**

---

## Task 11: Run Full Pipeline

**Step 1: Launch in tmux**

```bash
tmux new -s data_pipeline
cd /home/w50037733/robobrain_3dgs
python scripts/data_pipeline/run_pipeline.py --resume --depth_device cuda:1 2>&1 | tee logs/data_pipeline.log
```

**Step 2: Monitor**

```bash
watch -n 60 'cat data/processed/progress.json 2>/dev/null; echo ---; df -h /home/w50037733/'
```

Expected: ~24-30 hours, ~400 GB output, ~150K-250K episodes.
