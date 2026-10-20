# MORe — Combined3 historical LOSO checkpoint reproduction

Reproduce the reported three-class Combined3 result from **the correct frozen T6 checkpoint set**. No training is necessary.

## Scope and reported result

Dataset: CASME II + SMIC + SAMM, three-class Combined3; 442 samples and 68 LOSO folds. The historical experiment chose each epoch using held-out-subject ACC; therefore the result is **not an unbiased outer-test evaluation**.

| ACC | UAR | UF1 | Correct |
|---|---|---|---|
| 0.9027149321266968 | 0.8987828009284846 | 0.8920268372601029 | 399/442 |

Confusion matrix (rows: true class 0/1/2): `[[228,14,8],[9,96,4],[6,2,75]]`.

## What is and is not included

All MORe model, inference, dataset-loading and historical evaluation **source code** needed for checkpoint replay is contained here. The historical code is retained for numerical equivalence; no external private project checkout is required. Install standard Python dependencies from `requirements.txt`.

**Not included:** raw facial videos/images, original or processed facial images, optical-flow data, facial-region matrices, original dataset annotations, individual checkpoint weights, user directories, usernames, server paths, private logs, or sample-level predictions. Data and checkpoint files must be obtained separately and lawfully by the user. A repository without these inputs cannot produce the numerical results; "standalone" means no missing project code, not that confidential data is embedded.

Just alone, Data and Checkpoints download:  https://pan.baidu.com/s/1NkwzthuJnJNGZyfGQXUpzg?pwd=49qf 

## Inputs expected at runtime

Pass `--data-root` to a local prepared dataset directory containing:

```text
<data-root>/3DB/
  precomputed_tvl1_flows_224/dataset_with_tvl1_flows_224.csv
  mean_landmarks_224x224.npy
  precomputed_matrices/                 # per-sample precomputed facial matrices
  ...                                  # original dataset loader's input paths
```

Provide exactly the 68 frozen T6 B4 checkpoint files in `<checkpoint-dir>`, using the original names (`10.pth`, `s03.pth`, etc.). The **T6 manifest is built on the owner's trusted server**, verified before publication, and contains *only* `filename,bytes,sha256`, not the weight files.

## Installation and replay

```bash
python -m pip install -r requirements.txt
python replay_combined3.py --data-root /path/to/preprocessed/data \
  --checkpoint-dir /path/to/T6/weights \
  --output-dir ./outputs/replay_combined3 \
  --device cuda:0 --verify-sha256
python evaluate.py --predictions ./outputs/replay_combined3/predictions.csv
```

Use a fresh output directory each time. A full run must produce `442` predictions, `68` folds, and the metrics above. Verify that the checkpoint manifest is populated with 68 entries before running.

## Privacy and licensing

Do **not** publish any input data, paths, `.pth` files, `predictions.csv` (contains sample IDs), caches, internal run logs, or private provenance. Read `PRIVACY.md` before `git add`. Original dataset access and pretrained checkpoint sharing require independent authorization. No training or model selection is performed during replay. No original dataset is supplied.

## Source organization

- `replay_combined3.py`: reproducible checkpoint-only Combined3 inference and validation
- `protocol.py`: historical architecture/configuration and epoch-selection description
- `inference.py`, `evaluate.py`: model loading and metric evaluation
- `workdir/`: bundled historical model, dataset, preprocessing, metrics and graph operators
- `checkpoints/t6_checkpoint_manifest.csv`: trusted SHA256 inventory (generated on owner server)
- `train.py`: legacy training entry (not needed for replay; review before using)
- `PRIVACY.md`: repository release controls
