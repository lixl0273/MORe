# Repository privacy and release checklist

1. Never publish raw or preprocessed image/flow datasets, frame paths, subject metadata, per-sample outputs, private audit files, SSH material, passwords, API tokens or full training logs.
2. Weights are intentionally excluded. A SHA256 manifest is not a redistribution license.
3. Historical model code may contain legacy dataset paths; review scripts before running training. Public replay paths are supplied by CLI.
4. Never upload any README containing a real server filesystem path or personal username. Do not commit outputs or notebook checkpoints.
5. Evaluate original dataset terms and repository code's third-party licensing before making the project public. No license is asserted by this package.
6. A public user must acquire permitted datasets and prepared inputs separately. Do not claim data-free reproduction.
