"""Full-weight CLOAK training with float32 Adam state and updates on the CPU.

The GPU computes the released model's loss and gradients. Microbatch gradients
are averaged on the CPU before one global clip and AdamW update. This preserves
full-model training on GPUs too small to hold parameters, gradients and moments.
Checkpoints use the upstream format, including optimizer state and exact step.
"""
from __future__ import annotations

import dataclasses
import json
import time

import numpy as np

from .wandb_run import scalars


def gradient_function(cfg, model_def):
    import jax.numpy as jnp
    from flax import nnx

    def gradients(params, rng, batch):
        model = nnx.merge(model_def, params)
        model.train()

        def loss(model):
            return jnp.mean(model.compute_loss(rng, *batch, train=True))

        return nnx.value_and_grad(loss, argnums=nnx.DiffState(0, cfg.trainable_filter))(model)

    return gradients


def optimizer_function(tx):
    import jax.numpy as jnp
    import optax

    def update(params, grads, state):
        norm = optax.global_norm(grads)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, {'grad_norm': norm, 'finite': jnp.isfinite(norm)}

    return update


def main(trainer, cfg, loader_factory, *, microbatch=1, stop_at=None, tracker=None):
    """Train to ``stop_at`` (default: the end). ``tracker``: the run's W&B run (wandb_run.session), which gets
    every step's metrics next to metrics.jsonl; upstream's wandb logging does not cover this loop."""
    import jax
    import jax.numpy as jnp
    from openpi.shared import array_typing as at
    from openpi.training import checkpoints, sharding

    if jax.process_count() != 1 or len(jax.devices()) != 1:
        raise ValueError('CPU optimizer training requires one host and one accelerator')
    if cfg.batch_size % microbatch or microbatch < 1 or cfg.ema_decay is not None:
        raise ValueError('Batch must divide into positive microbatches; EMA is unsupported')
    if stop_at is not None and stop_at < 1:
        raise ValueError('stop_at must be a positive optimizer step')
    end = min(cfg.num_train_steps, stop_at or cfg.num_train_steps)
    accelerator, cpu = jax.devices()[0], jax.devices('cpu')[0]
    jax.config.update('jax_compilation_cache_dir', str(cfg.checkpoint_dir.parent / '.jax-cache'))
    host = jax.sharding.SingleDeviceSharding(cpu)
    device = jax.sharding.SingleDeviceSharding(accelerator)
    mesh = sharding.make_mesh(1)
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        cfg.checkpoint_dir, keep_period=cfg.keep_period, overwrite=False, resume=cfg.resume)
    trainer.save_config(cfg)
    rng = jax.random.key(cfg.seed)
    train_rng, init_rng = jax.random.split(rng)
    shape, _ = trainer.init_train_state(cfg, init_rng, mesh, resume=True)
    with at.disable_typechecking():
        host_sharding = jax.tree.map(lambda _: host, shape)
    if resuming:
        state = checkpoints.restore_state(manager, shape, host_sharding)
    else:
        values = trainer._load_weights_and_validate(cfg.weight_loader, shape.params.to_pure_dict())
        if any(isinstance(x, jax.ShapeDtypeStruct) for x in jax.tree.leaves(values)):
            raise ValueError('Host optimizer requires a complete pretrained checkpoint')
        params = shape.params
        params.replace_by_pure_dict(values)
        params = jax.device_put(params, host)
        del values
        init_optimizer = jax.jit(shape.tx.init, out_shardings=host)
        with jax.default_device(cpu):
            optimizer = init_optimizer(params)
        state = dataclasses.replace(shape, step=jax.device_put(np.int32(0), host),
                                    params=params, opt_state=optimizer)
    start = int(state.step)
    loader = loader_factory(cfg, sharding=host, shuffle=True, skip_batches=start)
    batches = iter(loader)
    grad = jax.jit(gradient_function(cfg, state.model_def),
                   in_shardings=(device, device, device), out_shardings=(device, device),
                   donate_argnums=(0,))
    update = jax.jit(optimizer_function(state.tx),
                     in_shardings=(host, host, host), out_shardings=(host, host, host),
                     donate_argnums=(0, 1, 2))
    add = jax.jit(lambda a, b: jax.tree.map(jnp.add, a, b),
                  in_shardings=(host, host), out_shardings=host, donate_argnums=(0, 1))
    scale = jax.jit(lambda a: jax.tree.map(lambda x: x / (cfg.batch_size // microbatch), a),
                    in_shardings=host, out_shardings=host, donate_argnums=(0,))
    metadata = checkpoints.restore_metadata(manager) if resuming else {}
    seen = checkpoints.restore_seen_episodes(manager) if resuming else set()
    started = time.monotonic()
    previous_elapsed = metadata.get('elapsed_seconds', 0.)
    metrics_path = cfg.checkpoint_dir / 'metrics.jsonl'
    try:
        for step in range(start, end):
            before = time.monotonic()
            raw = next(batches)
            batch, extra = raw[:2], raw[2]
            seen.update(str(x) for x in extra['file_path'])
            loaded = time.monotonic()
            accumulated, loss_sum = None, 0.
            for i, offset in enumerate(range(0, cfg.batch_size, microbatch)):
                micro = jax.tree.map(lambda x: x[offset:offset + microbatch], batch)
                key = jax.random.fold_in(jax.random.fold_in(train_rng, step), i)
                device_params = jax.device_put(state.params, device, may_alias=False)
                loss, grads = grad(device_params, jax.device_put(key, device), jax.device_put(micro, device))
                host_grads = jax.device_put(grads, host)
                jax.block_until_ready(host_grads)
                loss_sum += float(loss)
                del device_params, grads, loss
                accumulated = host_grads if accumulated is None else add(accumulated, host_grads)
            if cfg.batch_size != microbatch:
                accumulated = scale(accumulated)
            gradients_ready = time.monotonic()
            # Never write a checkpoint after a nonfinite loss/gradient update.
            if not np.isfinite(loss_sum):
                raise FloatingPointError(f'Nonfinite loss at optimizer step {step}')
            params, optimizer, info = update(state.params, accumulated, state.opt_state)
            jax.block_until_ready(params)
            if not bool(info['finite']):
                raise FloatingPointError(f'Nonfinite gradient at optimizer step {step}')
            state = dataclasses.replace(state, step=jax.device_put(np.int32(step + 1), host),
                                        params=params, opt_state=optimizer)
            metrics = {'step': step + 1, 'loss': loss_sum / (cfg.batch_size // microbatch),
                       'grad_norm': float(info['grad_norm']), 'seconds': time.monotonic() - before,
                       'data_seconds': loaded - before,
                       'gradient_and_transfer_seconds': gradients_ready - loaded,
                       'optimizer_seconds': time.monotonic() - gradients_ready,
                       'unique_episodes': len(seen), 'optimizer_device': 'cpu', 'all_weights_trainable': True}
            with metrics_path.open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
            if tracker is not None:
                tracker.log(scalars(metrics), step=step + 1)
            if step == start or (step + 1) % cfg.log_interval == 0:
                print(json.dumps(metrics), flush=True)
            if (step + 1) % cfg.save_interval == 0 or step + 1 == end:
                metadata.update(elapsed_seconds=previous_elapsed + time.monotonic() - started,
                                engine='full_weight_cpu_adam', microbatch=microbatch)
                checkpoints.save_state(manager, state, loader, train_metadata=metadata, seen_episodes=seen)
                # Donation on the next step must not invalidate async checkpoint buffers.
                manager.wait_until_finished()
    finally:
        manager.wait_until_finished()
        manager.close()
    return int(state.step)
