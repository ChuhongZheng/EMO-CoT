from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge


model_schema = {
    "family": merge(tstring, allowed(["gpt2_nn"])),
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
    "hidden_layer_size": merge(tinteger, default(4)),
    "n_in_intermediate": merge(tinteger, default(0)),  # if using chain of thought, number of output intermediate steps
    "n_out_intermediate": merge(tinteger, default(0)),  # if using chain of thought, number of input intermediate steps
    "hidden_sep_embed": merge(tboolean, default(False)),
}

curriculum_base_schema = {
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
}

optimization_schema = {
    # 优化 / 损失组合算法：
    # - "cot_io_sum": 原始的所有 level 直接求和（当前默认行为）
    # - "penalty":    惩罚函数法，按幂次加权求和
    # - "emo_g":      基于梯度范数选择单一 level 更新
    # - "emo":        Epigraph Multi-level Optimization（简化实现）
    # - "emo_m":      Moreau-like 平滑版本（运行时近似）
    "algo": merge(
        tstring,
        allowed(["cot_io_sum", "penalty", "emo_g", "emo", "emo_m"]),
        default("cot_io_sum"),
    ),
    # penalty 方法的超参数：rho > 1 时，低层级的损失权重更大
    "penalty_rho": merge(tfloat, default(1.0)),
    # EMO：z 的搜索区间
    "emo_z_low": merge(tfloat, default(0.0)),
    "emo_z_high": merge(tfloat, default(1.0)),
    # EMO-M：平滑强度（越大越快接近当前 loss）
    "emo_m_alpha": merge(tfloat, default(0.1)),
}

TASK_LIST = [
    "relu_nn_regression",
]

training_schema = {
    "task": merge(tstring, allowed(TASK_LIST)),
    "task_kwargs": merge(tdict, required),
    "data": merge(tstring, allowed(["gaussian"])),
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "optimization": stdict(optimization_schema),
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
}

wandb_schema = {
    "project": merge(tstring, default("dissecting-cot")),
    "entity": merge(tstring, default("dissecting-cot")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
    "debug_mode": merge(tboolean, default(False)),
}
