"""Thin entrypoint: invokes UNMODIFIED historical Combined3 LOSO training.

No training runs unless --execute is explicitly specified. Never overwrites
historical weights by default. Inspect historical CLI flags with --history-help.
"""
import argparse, subprocess, sys
from pathlib import Path
from protocol import MODEL_ARGS

HERE=Path(__file__).resolve().parent
HIST=HERE/'workdir/main_theory_cser.py'

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--execute',action='store_true',help='Explicitly authorize training')
 p.add_argument('--history-help',action='store_true')
 p.add_argument('--acknowledge-legacy-output-risk',action='store_true',help='Confirm historical script may write to legacy checkpoint destinations')
 a,extra=p.parse_known_args()
 if not HIST.is_file():p.error(f'Missing historical entrypoint: {HIST}')
 if a.history_help:
  raise SystemExit(subprocess.call([sys.executable,str(HIST),'--help'],cwd=HERE/'workdir'))
 if not a.execute:p.error('Training is disabled by default. Supply --execute and verified historical dataset/schedule options.')
 if not extra:p.error('Supply original experimental options after --execute; see --history-help.')
 if not a.acknowledge_legacy_output_risk:
  p.error('Historical script controls its own output paths. Require --acknowledge-legacy-output-risk after verifying them.')
 print('IMPORTANT: historical script may use its own output paths. Confirm configuration before running.',flush=True)
 cmd=[sys.executable,str(HIST),*MODEL_ARGS,*extra]
 print('Executing historical trainer:',cmd,flush=True)
 raise SystemExit(subprocess.call(cmd,cwd=HERE/'workdir'))
if __name__=='__main__':main()
