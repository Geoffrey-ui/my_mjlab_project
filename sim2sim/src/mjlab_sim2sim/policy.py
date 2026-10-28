"""ONNX policy wrapper that maps observation groups onto ONNX inputs."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch

from mjlab_sim2sim.bundle import Bundle


class OnnxPolicy:
  def __init__(self, bundle: Bundle):
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    self.session = ort.InferenceSession(str(bundle.policy_path), providers=providers)
    self.inputs = bundle.spec["policy"]["inputs"]  # onnx input -> [obs groups]
    self.output = bundle.spec["policy"]["output"]
    self.batch = {i.name: i.shape[0] for i in self.session.get_inputs()}

  def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    feeds = {
      name: torch.cat([obs[g] for g in groups], dim=-1 if obs[groups[0]].ndim == 2 else 1)
      .detach().cpu().numpy().astype(np.float32)
      for name, groups in self.inputs.items()
    }
    n = next(iter(feeds.values())).shape[0]
    fixed = all(isinstance(b, int) and b == 1 for b in self.batch.values())
    if fixed and n > 1:  # exported with a static batch of 1
      out = np.concatenate(
        [self.session.run([self.output], {k: v[i : i + 1] for k, v in feeds.items()})[0] for i in range(n)]
      )
    else:
      out = self.session.run([self.output], feeds)[0]
    device = next(iter(obs.values())).device
    return torch.from_numpy(out).to(device)
