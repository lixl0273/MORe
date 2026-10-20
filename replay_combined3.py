"""Read-only replay of frozen Combined3 LOSO checkpoints.

Run in the original preprocessed data environment; does not train or save weights.
All historical model and dataset code is imported unchanged from workdir/.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description='Frozen Combined3 LOSO: strictly load checkpoint per subject and export sample-level predictions; NO TRAINING')
    p.add_argument('--data-root', required=True, type=Path, help='Original preprocessed microemotion root')
    p.add_argument('--checkpoint-dir', required=True, type=Path, help='Directory of fold_{subject}_best.pth files')
    p.add_argument('--output-dir', required=True, type=Path, help='New directory for CSV, metrics, provenance (never overwrites)')
    p.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--fold', default='', help='Optional single fold for a smoke test, e.g. 10')
    p.add_argument('--verify-sha256', action='store_true', help='Verify checkpoint SHA256 against published manifest')
    a = p.parse_args()
    if a.batch_size < 1 or a.workers < 0:
        p.error('batch-size must be positive and workers nonnegative')
    root = Path(__file__).resolve().parent
    data_root = a.data_root.resolve()
    ckpt_dir = a.checkpoint_dir.resolve()
    index = data_root / '3DB/precomputed_tvl1_flows_224/dataset_with_tvl1_flows_224.csv'
    mean_landmarks = data_root / '3DB/mean_landmarks_224x224.npy'
    cache = data_root / '3DB/precomputed_matrices'
    manifest = root / 'checkpoints/t6_checkpoint_manifest.csv'
    for source in [index, mean_landmarks, manifest]:
        if not source.is_file():
            p.error('Required file missing: ' + str(source))
    if not cache.is_dir():
        p.error('Required matrix cache missing: ' + str(cache))
    if not ckpt_dir.is_dir():
        p.error('Checkpoint directory missing: ' + str(ckpt_dir))

    # Read and validate the complete index BEFORE importing GPU/runtime dependencies.
    import pandas as pd
    df = pd.read_csv(index)
    if df.shape[1] != 10:
        p.error(f'Historical combined index must have 10 columns, found {df.shape[1]}')
    df.columns = ['Dataset','Subject','Filename','Label','OnsetFrame','ApexFrame','OffsetFrame','OnsetPath','ApexPath','FlowPath']
    subjects = [str(s) for s in df['Subject'].drop_duplicates().tolist()]
    if len(df) != 442 or len(subjects) != 68:
        p.error(f'Combined3 index mismatch: samples={len(df)} (expected 442); subjects={len(subjects)} (expected 68)')
    if df[['Subject','Filename','Dataset']].isna().any().any():
        p.error('Missing subject/filename/dataset in index')
    key_cols = ['Dataset','Subject','Filename']
    if df.duplicated(key_cols).any():
        p.error('Duplicate sample identity in index')
    if a.fold and a.fold not in subjects:
        p.error(f'Unknown fold {a.fold!r}; available subjects: {subjects}')
    wanted = [a.fold] if a.fold else subjects
    with open(manifest, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    by_name = {r['filename']: r for r in rows}
    if len(rows) != 68 or len(by_name) != 68:
        p.error('Expected 68 unique entries in manifest')
    expected_filenames = {f'{s}.pth' for s in subjects}
    if set(by_name) != expected_filenames:
        p.error('Index subject IDs and checkpoint manifest filenames differ')
    checkpoints = {s: ckpt_dir / f'{s}.pth' for s in wanted}
    for subject, ckpt in checkpoints.items():
        if not ckpt.is_file():
            p.error(f'Missing checkpoint for subject={subject}: {ckpt}')
        expected = int(by_name[ckpt.name]['bytes'])
        if ckpt.stat().st_size != expected:
            p.error(f'Checkpoint size mismatch for {ckpt.name}: {ckpt.stat().st_size} != {expected}')
    actual_filenames = {x.name for x in ckpt_dir.glob('*.pth')}
    if actual_filenames != expected_filenames:
        p.error(f'Checkpoint inventory mismatch: missing={sorted(expected_filenames-actual_filenames)} extra={sorted(actual_filenames-expected_filenames)}')
    if a.output_dir.exists():
        p.error('Output directory already exists; choose a fresh directory (no overwrite): ' + str(a.output_dir))

    # Match historical seed and model-building argument contract, no training loop.
    import torch
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from protocol import MODEL_ARGS, NUM_CLASSES, EXPECTED_FOLDS, NAME
    sys.path.insert(0, str(root / 'workdir'))
    import main_theory_cser as hist
    from model1.model import MyNet_FRAT
    from model1.dataset import RafDataSet
    from process.facial_matrix import PrecomputedMatrixLoader
    old_argv = sys.argv
    try:
        sys.argv = ['historical_model'] + list(MODEL_ARGS)
        opts = hist.parse_args()
    finally:
        sys.argv = old_argv
    if opts.model_name != 'en_FRAT' or opts.num_classes != 3 or not opts.use_cser or not opts.use_facial_matrices:
        raise RuntimeError('Frozen Combined3 MODEL_ARGS are inconsistent with expected historical architecture')
    random.seed(2)
    import numpy as np
    np.random.seed(2)
    torch.manual_seed(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA selected but unavailable')
    matrix_loader = PrecomputedMatrixLoader(cache_dir=str(cache), default_on_fail=True,
        mean_landmarks_path=str(mean_landmarks), target_size=opts.image_size, patch_size=opts.patch_size)
    apex_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=[0.395,0.386,0.472], std=[0.187,0.182,0.200])])
    # Dataset historically leaves flow tensor unnormalized; use same flow transform passed
    # by the trainer, although RafDataSet does not apply it to the loaded flow tensor.
    flow_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=[0.504,0.471,0.062], std=[0.245,0.265,0.108])])

    a.output_dir.mkdir(parents=True, exist_ok=False)
    out_csv = a.output_dir / 'predictions.csv'
    columns = ['global_index','dataset','subject','sample_id','Label','ReplayPrediction',
               'prob_0','prob_1','prob_2','checkpoint','checkpoint_sha256']
    sample_count = 0
    ckpt_info = {}
    from inference import load_frozen_checkpoint, predict_preprocessed
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for k, subject in enumerate(wanted, 1):
            selected = df.index[df['Subject'].astype(str) == subject].tolist()
            if not selected:
                raise RuntimeError('Empty subject: ' + subject)
            ds = RafDataSet(str(data_root), phase='test', num_loso=subject,
                transform_flow=flow_transform, transform_apex=apex_transform,
                num_classes=3, db='3DB', combined=True, use_facial_matrices=True,
                matrix_cache_dir=str(cache), matrix_loader=matrix_loader)
            if len(ds) != len(selected):
                raise RuntimeError(f'Fold {subject} dataset/index count mismatch: {len(ds)} != {len(selected)}')
            for local_i, global_i in enumerate(selected):
                row = df.iloc[global_i]
                meta = ds.csv_rows[local_i]
                if (str(meta['Dataset']),str(meta['Subject']),str(meta['Filename'])) != (str(row.Dataset),str(row.Subject),str(row.Filename)):
                    raise RuntimeError(f'Sample order mismatch fold={subject}, local_index={local_i}')
            model = MyNet_FRAT(in_channels=3, num_classes=opts.num_classes, depth=opts.trans_depth,
                img_size=opts.image_size, patch_size=opts.patch_size, num_heads=opts.trans_num_heads,
                use_frasa=True, embed_dim=opts.embed_dim, use_mpt=opts.use_mpt,
                mpt_variant=opts.mpt_variant, mpt_residual_scale=opts.mpt_residual_scale)
            if not hasattr(model, 'vit_pos'):
                raise RuntimeError('Historical model vit_pos missing')
            for attr in ['rho_com','rho_reg','motion_tau','prior_type','branch_mode','gate_mode','fixed_gamma_com','fixed_gamma_reg','gate_type']:
                setattr(model.vit_pos, attr, getattr(opts, attr))
            for blk in getattr(model.vit_pos, 'blocks', []):
                if hasattr(blk, 'attn') and hasattr(blk.attn, 'gate_type'):
                    blk.attn.gate_type = opts.gate_type
            model = hist.maybe_wrap_cser(model, opts, opts.image_size, opts.patch_size, opts.embed_dim, opts.num_classes)
            checkpoint = checkpoints[subject]
            digest = sha256(checkpoint) if a.verify_sha256 else 'NOT_VERIFIED'
            if a.verify_sha256 and digest.lower() != by_name[checkpoint.name]['sha256'].lower():
                raise RuntimeError(f'Frozen manifest SHA256 mismatch: {checkpoint.name}')
            model = load_frozen_checkpoint(model, checkpoint, device=device)
            loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, drop_last=False,
                num_workers=a.workers, pin_memory=(device.type == 'cuda'))
            cursor = 0
            for batch in loader:
                flow = batch['flow'].to(device)
                comp = batch['A_Component'].to(device).squeeze(1)
                reg = batch['A_Region'].to(device).squeeze(1)
                result = predict_preprocessed(model, flow, comp, reg)
                pred = result['predictions'].cpu().tolist()
                probs = result['probabilities'].cpu().tolist()
                labels = batch['label'].cpu().tolist()
                for j, (y, yhat, prob) in enumerate(zip(labels, pred, probs)):
                    global_i = selected[cursor + j]
                    row = df.iloc[global_i]
                    expected_label = int(row.Label) if int(row.Label) in (0,1) else 2
                    if int(y) != expected_label:
                        raise RuntimeError(f'Label mapping mismatch global_index={global_i}')
                    writer.writerow(dict(global_index=global_i, dataset=row.Dataset, subject=subject,
                        sample_id=row.Filename, Label=int(y), ReplayPrediction=int(yhat),
                        prob_0=repr(prob[0]), prob_1=repr(prob[1]), prob_2=repr(prob[2]),
                        checkpoint=checkpoint.name, checkpoint_sha256=digest))
                cursor += len(labels)
                sample_count += len(labels)
            if cursor != len(selected):
                raise RuntimeError(f'Fold {subject}: prediction count {cursor} != {len(selected)}')
            ckpt_info[subject] = dict(file=checkpoint.name, bytes=checkpoint.stat().st_size, sha256=digest, samples=cursor)
            print(f'[PASS] fold {k}/{len(wanted)} subject={subject} samples={cursor} strict=True', flush=True)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    expected_samples = len(df) if not a.fold else int((df['Subject'].astype(str) == a.fold).sum())
    if sample_count != expected_samples:
        raise RuntimeError(f'Prediction coverage {sample_count} != {expected_samples}')
    from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
    replay = pd.read_csv(out_csv)
    if len(replay) != sample_count or replay.global_index.nunique() != sample_count:
        raise RuntimeError('Duplicate/missing global_index in prediction CSV')
    y = replay['Label'].astype(int).tolist()
    yp = replay['ReplayPrediction'].astype(int).tolist()
    def metrics(sub):
        yt = sub['Label'].astype(int).tolist()
        yh = sub['ReplayPrediction'].astype(int).tolist()
        return dict(samples=len(sub), ACC=float(accuracy_score(yt,yh)),
            UAR=float(recall_score(yt,yh,labels=[0,1,2],average='macro',zero_division=0)),
            UF1=float(f1_score(yt,yh,labels=[0,1,2],average='macro',zero_division=0)),
            confusion_matrix=confusion_matrix(yt,yh,labels=[0,1,2]).tolist())
    summary = dict(protocol=NAME, seed=2, full_replay=not bool(a.fold),
        index_sha256=sha256(index), mean_landmarks_sha256=sha256(mean_landmarks),
        samples=sample_count, folds=len(wanted), overall=metrics(replay),
        by_dataset={str(k):metrics(v) for k,v in replay.groupby('dataset',sort=True)},
        checkpoint_provenance=ckpt_info, checkpoint_layout='t6', weights_verified=bool(a.verify_sha256),
        caveat='Frozen checkpoint evaluation, NOT independent test-set model selection; held-out ACC selected each epoch.')
    (a.output_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print('[REPLAY_SUCCESS]', out_csv, 'samples=', sample_count, 'folds=',len(wanted),flush=True)
    print(json.dumps(summary['overall'], indent=2))

if __name__ == '__main__':
    main()
