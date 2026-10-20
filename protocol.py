"""Frozen historical MORe Combined3 protocol (not an independent validation protocol).

Configuration belongs here; all executable entrypoints reuse this single contract.
The held-out subject is used to choose the highest-ACC epoch and to report the
fold's performance. Results are optimistic and must not be described as unbiased.
"""
from pathlib import Path
import argparse

NAME = 'combined3_historical_loso_heldout_acc'
NUM_CLASSES = 3
IMAGE_SIZE = 224
PATCH_SIZE = 16
TRANS_DEPTH = 3
TRANS_NUM_HEADS = 2
EMBED_DIM = 512
CHECKPOINT_SUBDIR = 'user-provided T6 checkpoints: <subject>.pth'
EXPECTED_FOLDS = 68
SELECTION = 'best held-out-subject ACC per fold (>=, later epoch wins ties)'

MODEL_ARGS = (
 '--model_name','en_FRAT','--combined','--num_classes','3','--image_size','224',
 '--patch_size','16','--trans_depth','3','--trans_num_heads','2',
 '--embed_dim','512','--use_mpt','--mpt_variant','mul',
 '--mpt_residual_scale','0.5','--prior_type','static',
 '--branch_mode','both','--gate_mode','learned','--gate_type','sigmoid',
 '--use_cser','--cser_mode','residual','--cser_graph_type','soft',
 '--cser_prior_beta','0.7','--cser_alpha','0.05',
 '--use_cser_reliability','--cser_rel_mode','bounded',
 '--cser_rel_min','0.75','--use_facial_matrices','True',
)
# NOTE: These are frozen B4 model configuration parameters observed in prior
# experiments. Training schedule and dataset path are intentionally NOT guessed.
# Specify historical --epochs, --raf_path, --db, etc. via extra CLI options.

def parser():
 p=argparse.ArgumentParser(description='MORe Combined3 historical protocol')
 p.add_argument('--print',action='store_true',dest='show')
 return p

if __name__=='__main__':
 import json
 print(json.dumps({'protocol':NAME,'num_classes':NUM_CLASSES,'checkpoint_folds':EXPECTED_FOLDS,'checkpoint_subdir':CHECKPOINT_SUBDIR,'selection':SELECTION,'model_args':MODEL_ARGS},indent=2))
