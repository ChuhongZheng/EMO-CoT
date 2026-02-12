import json
import os

from munch import Munch
from tqdm import tqdm
import time
import torch
import yaml

import models
from samplers import get_data_sampler
from tasks import get_task_sampler


def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = models.build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        # hacky way of ensuring that if I'm trying to load keys that
        # aren't in the checkpoint, just use random
        for key in model.state_dict().keys():
            if key not in state["model_state_dict"].keys():
                state["model_state_dict"][key] = model.state_dict()[key]
        model.load_state_dict(state["model_state_dict"])
    elif step == 0:
        # return random init model
        return model, conf
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        # hacky way of ensuring that if I'm trying to load keys that
        # aren't in the checkpoint, just use random
        for key in model.state_dict().keys():
            if key not in state_dict.keys():
                state_dict[key] = model.state_dict()[key]
        model.load_state_dict(state_dict)

    return model, conf


def eval_batch(model, task_sampler, xs, task_name=None):
    task = task_sampler()
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2"]:
        device = "cuda"
    else:
        device = "cpu"
    if task_name in ['relu_nn_regression']:
        ys, layer_activations = task.evaluate(xs)
        layer_activations = [act.to(device) for act in layer_activations]
    else:
        raise NotImplementedError

    if model.name.split("_")[0] in ["gpt2"]:
        # Show inner per-point progress only when requested (to avoid spamming output)
        show_inner_progress = getattr(model, "_eval_show_inner_progress", False)
        pred = model.predict(
            xs.to(device),
            ys.to(device),
            layer_activations=layer_activations,
            show_progress=show_inner_progress,
        ).detach()
    else:
        raise NotImplementedError
    metrics = task.get_metric()(pred.cpu(), ys)

    # Stepwise losses per point (s1, s2, ..., y) for alignment with training; same step indices as wandb.
    stepwise_losses = None
    if task_name == "relu_nn_regression" and getattr(model, "n_in_intermediate", 0) > 0:
        with torch.no_grad():
            losses, _ = model(
                xs.to(device), ys.to(device),
                task.get_training_metric(),
                layer_activations=layer_activations,
                return_per_point=True,
            )
        # losses: list of (n_points,) tensors, one per step
        stepwise_losses = [t.cpu() for t in losses]

    return metrics, stepwise_losses


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def eval_model(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    num_eval_examples=1280,
    batch_size=64,
    task_sampler_kwargs={},
):
    
    assert num_eval_examples % batch_size == 0
    data_sampler = get_data_sampler(data_name, n_dims)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )

    n_batches = num_eval_examples // batch_size
    print(f"[eval] task={task_name} n_points={n_points} batch_size={batch_size} n_batches={n_batches}")
    t0 = time.time()
    all_metrics = []
    all_stepwise = []  # list of list of (n_points,) tensors per batch
    pbar = tqdm(range(n_batches), desc=f"Evaluating ({task_name})", leave=True)
    for i in pbar:
        # Only show per-point predict progress for the first batch (visibility without huge output)
        model._eval_show_inner_progress = (i == 0)
        batch_t0 = time.time()
        xs = data_sampler.sample_xs(n_points, batch_size)
        metrics, stepwise_losses = eval_batch(model, task_sampler, xs, task_name)
        all_metrics.append(metrics)
        if stepwise_losses is not None:
            all_stepwise.append(stepwise_losses)
        batch_dt = time.time() - batch_t0
        elapsed = time.time() - t0
        avg = elapsed / (i + 1)
        eta = avg * (n_batches - (i + 1))
        pbar.set_postfix({"batch_s": f"{batch_dt:.2f}", "eta_s": f"{eta:.0f}"})
    if hasattr(model, "_eval_show_inner_progress"):
        delattr(model, "_eval_show_inner_progress")

    metrics = torch.cat(all_metrics, dim=0)
    results = aggregate_metrics(metrics)
    # Aggregate stepwise loss per point: average over batches, result stepwise_loss.{k} = list of length n_points
    if all_stepwise:
        n_steps = len(all_stepwise[0])
        for k in range(n_steps):
            # stack batches: each batch has tensor (n_points,) -> stack -> (n_batches, n_points) -> mean(0) -> (n_points,)
            stacked = torch.stack([batch[k] for batch in all_stepwise], dim=0)
            results[f"stepwise_loss.{k}"] = stacked.mean(dim=0).tolist()
    return results


def build_evals(conf):
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size

    task_name = conf.training.task
    data_name = conf.training.data

    evaluation_kwarg = {
        "task_name": task_name,
        "task_sampler_kwargs": conf.training.task_kwargs,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
    }

    return evaluation_kwarg


def compute_evals(model, evaluation_kwargs, save_path=None):
    metrics = eval_model(model, **evaluation_kwargs)

    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(metrics, fp, indent=2)

    return metrics


def get_run_metrics(run_path, step=-1, cache=True):
    model, conf = get_model_from_run(run_path, step)
    model = model.cuda().eval()
    evaluation_kwargs = build_evals(conf)

    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    try:
        with open(save_path) as fp:
            metrics = json.load(fp)
            return metrics
    except Exception:
        metrics = {}

    metrics = compute_evals(model, evaluation_kwargs, save_path)
    return metrics



