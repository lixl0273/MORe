"""Tensor-level MORe inference with a pre-constructed, strictly loaded historical model.
This does not implement raw video/optical-flow preprocessing or checkpoint/model construction.
"""
import torch

@torch.no_grad()
def predict_preprocessed(model, flow, a_component, a_region):
    if flow.ndim != 4 or flow.shape[1] != 3:
        raise ValueError('flow must have shape [B,3,H,W]')
    if a_component.ndim != 3 or a_component.shape[1] != 8:
        raise ValueError('a_component must have shape [B,8,N]')
    if a_region.ndim != 3:
        raise ValueError('a_region must have shape [B,R,N]')
    if flow.shape[0] != a_component.shape[0] or flow.shape[0] != a_region.shape[0]:
        raise ValueError('batch sizes differ')
    if a_component.shape[2] != a_region.shape[2]:
        raise ValueError('membership token counts differ')
    model.eval()
    logits = model(flow, a_component, a_region)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise RuntimeError('historical model did not return [B,C] logits')
    probabilities = torch.softmax(logits, dim=1)
    return {'logits':logits, 'probabilities':probabilities,
            'predictions':probabilities.argmax(dim=1)}

def load_frozen_checkpoint(model, checkpoint_path, device='cpu'):
    """Strictly load a trusted, frozen bare PyTorch state_dict into historical model."""
    from pathlib import Path
    p=Path(checkpoint_path)
    if not p.is_file():raise FileNotFoundError(p)
    state=torch.load(p,map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not state:
        raise TypeError('Expected bare non-empty state_dict')
    model.load_state_dict(state,strict=True)
    model.to(device).eval()
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MORe Combined3 tensor inference API. Import load_frozen_checkpoint and predict_preprocessed; raw-video CLI is not implemented.')
    parser.parse_args()
    print('Import load_frozen_checkpoint and predict_preprocessed from inference.py.')
