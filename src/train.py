import os
import shutil
from random import randint
import uuid
import datetime

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model



import wandb

torch.backends.cudnn.benchmark = True



def _train_step_cot_io_sum(model, xs, ys, optimizer, loss_func, layer_activations=None):
    """
    原始 CoT-I/O 行为：直接把所有 level 的 loss 求和作为优化目标。
    """
    optimizer.zero_grad()
    losses, total_loss = model(xs, ys, loss_func, layer_activations=layer_activations)
    total_loss.backward()
    optimizer.step()
    losses = [ls.detach().item() for ls in losses]
    return losses, total_loss.detach().item()


def _train_step_penalty(model, xs, ys, optimizer, loss_func, layer_activations, rho: float):
    """
    Penalty Method：L = sum_k rho^{K-1-k} * f_k(theta)
    """
    optimizer.zero_grad()
    losses, _ = model(xs, ys, loss_func, layer_activations=layer_activations)
    K = len(losses)
    weights = [rho ** (K - 1 - k) for k in range(K)]
    total_loss = sum(w * l for w, l in zip(weights, losses))
    total_loss.backward()
    optimizer.step()
    losses = [ls.detach().item() for ls in losses]
    return losses, total_loss.detach().item()


def _train_step_emo_g(model, xs, ys, optimizer, loss_func, layer_activations=None):
    """
    EMO-G：计算每一层 loss 的梯度范数，选择范数最大的那一层做一次真正的更新。
    """
    # 先算出所有 level 的 loss
    optimizer.zero_grad()
    losses, _ = model(xs, ys, loss_func, layer_activations=layer_activations)

    # 为每个 level 单独计算梯度范数（不更新参数）
    grad_norms = []
    for l in losses:
        optimizer.zero_grad()
        l.backward(retain_graph=True)
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm_sq += param_norm * param_norm
        grad_norms.append(total_norm_sq ** 0.5)

    # 选择梯度范数最大的 level
    max_idx = max(range(len(grad_norms)), key=lambda i: grad_norms[i])

    # 用该 level 的 loss 做一次真正的更新
    optimizer.zero_grad()
    final_loss = losses[max_idx]
    final_loss.backward()
    optimizer.step()

    losses = [ls.detach().item() for ls in losses]
    return losses, final_loss.detach().item()


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    data_sampler_args = {}
    task_sampler_args = {}

    task = task_sampler(**task_sampler_args)

    algo = args.training.optimization.algo
    penalty_rho = getattr(args.training.optimization, "penalty_rho", 1.0)

    for i in pbar:
        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )

        loss_func = task.get_training_metric()
        if args.training.task in ['relu_nn_regression']:
            task = task_sampler(**task_sampler_args)
            ys, layer_activations = task.evaluate(xs)
            layer_activations = [act.cuda() for act in layer_activations]
            if algo == "cot_io_sum":
                losses, loss = _train_step_cot_io_sum(
                    model, xs.cuda(), ys.cuda(), optimizer,
                    loss_func, layer_activations=layer_activations,
                )
            elif algo == "penalty":
                losses, loss = _train_step_penalty(
                    model, xs.cuda(), ys.cuda(), optimizer,
                    loss_func, layer_activations=layer_activations,
                    rho=penalty_rho,
                )
            elif algo == "emo_g":
                losses, loss = _train_step_emo_g(
                    model, xs.cuda(), ys.cuda(), optimizer,
                    loss_func, layer_activations=layer_activations,
                )
            else:
                raise NotImplementedError(f"Unknown optimization.algo: {algo}")
        else:
            raise NotImplementedError

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "stepwise/loss": dict(
                        zip(list(range(len(losses))), losses)
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ) or (i == args.training.train_steps - 1):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            mode="disabled" if args.debug_mode else "online",
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

    if args.debug_mode:
        # delete wandb directory when done
        print("Deleting out_dir {} because of debug mode".format(args.out_dir))
        shutil.rmtree("{}".format(args.out_dir), ignore_errors=True)


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2_nn"]
    print(f"Running with: {args}")

    if args.debug_mode:
        args.out_dir = "../models/debug"

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir
        # add a timestamp here
        args.wandb['timestamp'] = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
