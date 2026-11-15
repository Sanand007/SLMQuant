# Adapted from MLX-LM mlx_lm/quant/gptq.py, Copyright (c) 2025 Apple Inc.
# Distributed under MLX-LM's MIT license. Adaptations: per-module bits,
# float32 Hessian accumulation and dead-column numerical handling.
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.quant.gptq import quantize

class Catcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.H = mx.array(0.0)

    def __call__(self, x, *args, **kwargs):
        xf = x.flatten(0, -2).astype(mx.float32)
        self.H = self.H + xf.T @ xf
        return self.module(x, *args, **kwargs)

def quantize_mixed(model, data, allocation=None, group_size=64, progress=None, guard=None, limit=None):
    allocation = allocation or {}
    layers = [(k, Catcher(v)) for k, v in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
              if type(v) is nn.Linear and v.weight.shape[-1] % group_size == 0]
    if limit is not None:
        layers = layers[:limit]
    model.update_modules(tree_unflatten(layers))
    for batch in data:
        if guard:
            guard()
        output = model(mx.array(batch[None, :-1]))
        mx.eval(output, [m.H for _, m in layers])
        del output
    config = {"bits": 4, "group_size": group_size, "mode": "affine"}
    def error_fn(bits):
        @mx.compile
        def error(w, d, scales, biases):
            q = mx.clip(mx.round((w - biases) / scales), 0.0, 2 ** bits - 1)
            return (w - (scales * q + biases)) / d
        return error
    errors = {b: error_fn(b) for b in (2, 4, 8)}
    for index, (name, wrapper) in enumerate(layers):
        if guard:
            guard()
        bits = allocation.get(name, 4)
        if bits not in errors:
            raise ValueError(f"Unsupported GPTQ precision: {bits}")
        H = wrapper.H
        del wrapper.H
        W = wrapper.module.weight.astype(mx.float32)
        original_dtype = wrapper.module.weight.dtype
        diagonal = mx.arange(H.shape[0])
        dead = mx.diag(H) == 0
        W = mx.where(dead[None, :], 0, W)
        H[diagonal, diagonal] = mx.where(dead, 1, mx.diag(H))
        with mx.stream(mx.cpu):
            damping = mx.maximum(1e-6, 1e-2 * mx.mean(mx.diag(H)))
            H[diagonal, diagonal] += damping
            inv = mx.linalg.cholesky_inv(mx.linalg.cholesky(H))
            Hinv = mx.linalg.cholesky(inv, upper=True)
            mx.eval(Hinv)
        del H, inv
        all_scales, all_biases = [], []
        for i in range(0, W.shape[-1], group_size):
            j = i + group_size
            _, scales, biases = mx.quantize(W[..., i:j], bits=bits, group_size=group_size)
            all_scales.append(scales)
            all_biases.append(biases)
            err = mx.zeros_like(W[..., i:j])
            for k in range(i, j):
                e = errors[bits](W[..., k:k + 1], Hinv[k, k], scales, biases)
                W[..., k:j] -= e @ Hinv[k:k + 1, k:j]
                err[..., k - i:k - i + 1] = e
                mx.eval(W, err)
            W[..., j:] -= err @ Hinv[i:j, j:]
        scales, biases = mx.concatenate(all_scales, -1), mx.concatenate(all_biases, -1)
        q = wrapper.module.to_quantized(bits=bits, group_size=group_size)
        q.weight = quantize(W, bits, scales, biases)
        q.scales, q.biases = scales, biases
        q.set_dtype(original_dtype)
        mx.eval(q.parameters())
        model.update_modules(tree_unflatten([(name, q)]))
        config[name] = {"bits": bits, "group_size": group_size, "mode": "affine"}
        layers[index] = (name, None)
        del W, Hinv, wrapper
        if progress:
            progress(index + 1, len(layers))
    remaining = []
    for name, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if hasattr(module, "to_quantized") and module.weight.shape[-1] % group_size == 0:
            remaining.append((name, module.to_quantized(bits=4, group_size=group_size)))
    model.update_modules(tree_unflatten(remaining))
    mx.eval(model.parameters())
    return model, config
