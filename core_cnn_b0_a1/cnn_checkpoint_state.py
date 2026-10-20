"""Atomic, local-only PyTorch epoch checkpoints for CCAST preemptible CNN jobs.

Checkpoint is a trusted research artifact on private CCAST storage. Never commit or
publish its model/optimizer tensors, RNG state, samples or paths to GitHub.
"""
from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

VERSION = 1


def rng_snapshot() -> dict:
    n = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': {'bit_generator': n[0], 'keys': n[1].tolist(), 'pos': int(n[2]),
                  'has_gauss': int(n[3]), 'cached_gaussian': float(n[4])},
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['bit_generator'], np.asarray(n['keys'], dtype=np.uint32),
                         n['pos'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if torch.cuda.is_available() and state['torch_cuda']:
        if len(state['torch_cuda']) != torch.cuda.device_count():
            raise RuntimeError('Checkpoint CUDA device count differs from current environment')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def atomic_checkpoint(path: Path, payload: dict) -> None:
    """No partially-written file can replace the last successful epoch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.checkpoint.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as target:
            torch.save(payload, target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(name, path)
        # Best effort durability on POSIX cluster filesystem.
        try:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_checkpoint(path: Path, expected_identity: dict) -> dict | None:
    if not path.exists():
        return None
    # Do not deserialize arbitrary custom pickled classes from a checkpoint.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('format_version') != VERSION:
        raise RuntimeError('Unsupported or malformed local checkpoint format')
    if checkpoint.get('run_contract') != expected_identity:
        raise RuntimeError('Checkpoint input/code/config fingerprint mismatch; refusing stale resume')
    if not isinstance(checkpoint.get('completed_epoch'), int) or checkpoint['completed_epoch'] < 1:
        raise RuntimeError('Invalid completed epoch in checkpoint')
    return checkpoint


def snapshot_state(epoch: int, identity: dict, model, optimizer, scheduler, scaler,
                   history: list, best_state: dict, best_loss: float, best_epoch: int,
                   patience_remaining: int, unfrozen: bool) -> dict:
    # Write GPU state as CPU tensors so checkpoints can resume on another GPU.
    return {'format_version': VERSION, 'completed_epoch': epoch, 'run_contract': identity,
            'model': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'history': history,
            'best_model': best_state, 'best_loss': float(best_loss),
            'best_epoch': int(best_epoch), 'patience_remaining': int(patience_remaining),
            'unfrozen': bool(unfrozen), 'rng': rng_snapshot()}
