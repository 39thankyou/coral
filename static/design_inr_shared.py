"""Train shared input/output INRs with the existing second-order adaptation loop."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")

from coral.metalearning import outer_step
from run_codelib.checkpoints import validate_checkpoint, validate_config
from coral.utils.data.load_data import set_seed, get_operator_data
from coral.utils.data.operator_dataset import OperatorDataset
from coral.utils.models.load_inr import create_inr_instance
from coral.utils.plot import show

@hydra.main(version_base=None, config_path="config/static/", config_name="design_shared.yaml")
def main(cfg: DictConfig) -> None:

    validate_config(cfg)
    torch.set_default_dtype(torch.float32)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    data_dir = cfg.data.dir              # 数据存放目录（可为相对路径）
    if data_dir is not None and not os.path.isabs(data_dir):
        data_dir = os.path.join(get_original_cwd(), data_dir)
    dataset_name = cfg.data.dataset_name # 数据集名称
    ntrain = cfg.data.ntrain             # 训练样本数（为空时由数据接口决定）
    ntest = cfg.data.ntest               # 测试样本数
    sub_tr = cfg.data.sub_tr             # 训练集下采样比例
    seed = cfg.data.seed                 # 随机种子

    batch_size = cfg.optim.batch_size                   # 训练批次大小
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )                                                   # 验证批次大小（未指定时沿用训练批次大小）
    lr_inr = cfg.optim.lr_inr                           # INR 网络参数的学习率
    gamma_step = cfg.optim.gamma_step                   # 学习率衰减步长（当前流程未使用，保留）
    lr_code = cfg.optim.lr_code                         # 隐编码（code）的基础学习率
    meta_lr_code = cfg.optim.meta_lr_code               # 可学习步长 alpha 的元学习率
    weight_decay_code = cfg.optim.weight_decay_code     # 隐编码的权重衰减（当前流程未使用，保留）
    inner_steps = cfg.optim.inner_steps                 # 训练时的内层优化步数
    test_inner_steps = cfg.optim.test_inner_steps       # 测试时的内层优化步数
    epochs = cfg.optim.epochs                           # 训练总轮数
    lr_mlp = cfg.optim.lr_mlp                           # MLP（回归网络）学习率（当前流程未使用，保留）
    weight_decay_mlp = cfg.optim.weight_decay_mlp       # MLP 权重衰减（当前流程未使用，保留）

    latent_dim_in = cfg.inr_in.latent_dim
    latent_dim_out = cfg.inr_out.latent_dim

    entity = cfg.wandb.entity        # WandB 团队/账号名
    project = cfg.wandb.project      # WandB 项目名
    run_id = cfg.wandb.id            # 本次运行的 id
    run_name = cfg.wandb.name        # 本次运行的名称
    wandb_root = os.getenv("WANDB_DIR") or os.getcwd()
    run_dir = (
        os.path.join(wandb_root, f"wandb/{cfg.wandb.dir}")
        if cfg.wandb.dir is not None
        else None
    )                                # 运行目录（可选），若配置了 dir 则拼接出完整路径
    sweep_id = cfg.wandb.sweep_id    # WandB sweep（超参搜索）的 id

    print("run dir given", run_dir)

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        id=run_id,
        dir=None,
    )
    if run_dir is not None:
        os.symlink(run.dir.split("/files")[0], run_dir)

    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    run_name = cfg.wandb.name or run.name or "airfoil-shared-inr"

    print("id", run.id)
    print("dir", run.dir)

    RESULTS_DIR = Path(wandb_root) / dataset_name / "inr"
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    wandb.log({"results_dir": str(RESULTS_DIR)}, step=0, commit=False)

    set_seed(seed)

    run.tags = (
        ("different-inr-regression",)
        + (dataset_name,)
        + (f"sub={sub_tr}",)
    )   # 为本次运行打上标签

    set_seed(seed)

    x_train, y_train, x_test, y_test, grid_tr, grid_te = get_operator_data(
        data_dir, dataset_name, ntrain, ntest,
        sub_tr=cfg.data.sub_tr, sub_te=cfg.data.sub_te, same_grid=cfg.data.same_grid,
    )

    print('x_train', x_train.shape)
    print('y_train', y_train.shape)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)
    print('grid_tr', grid_tr.shape)
    print('grid_te', grid_te.shape)

    trainset = OperatorDataset(x_train,
        y_train,
        grid_tr,
        latent_dim_a=cfg.inr_in.latent_dim,   # 输入场 a 对应的隐向量维度
        latent_dim_u=cfg.inr_out.latent_dim,  # 输出场 u 对应的隐向量维度
        dataset_name=None,
        data_to_encode=None,
    )

    testset = OperatorDataset(x_test,
        y_test,
        grid_te,
        latent_dim_a=cfg.inr_in.latent_dim,
        latent_dim_u=cfg.inr_out.latent_dim,
        dataset_name=None,
        data_to_encode=None,
    )
    ntrain = len(trainset)   # 用实际数据量覆盖配置中的 ntrain
    ntest = len(testset)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size_val, shuffle=False)

    print("train", len(trainset))
    print('test', len(testset))

    input_dim = grid_tr.shape[-1]
    output_dim_in = x_train.shape[-1]
    output_dim_out = y_train.shape[-1]

    cfg.inr = cfg.inr_in
    inr_in = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim_in, device=device
    )
    cfg.inr = cfg.inr_out
    inr_out = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim_out, device=device
    )
    from anchormix.factory import initialize_models, parameter_groups, project_positions
    from anchormix.checkpoint import training_metadata
    adapt = outer_step
    if cfg.get("network_package") == "anchormix":
        from anchormix.metalearning import outer_step as adapt
    resume_path = cfg.get("resume")
    resume = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path else None
    if resume is None:
        initialize_models((inr_in, inr_out), grid_tr)
    else:
        validate_checkpoint(resume)
        inr_in.load_state_dict(resume["inr_in"])
        inr_out.load_state_dict(resume["inr_out"])

    alpha_in = nn.Parameter(torch.Tensor([lr_code]).to(device))
    alpha_out = nn.Parameter(torch.Tensor([lr_code]).to(device))

    optimizer_in = torch.optim.AdamW(
        [
            *parameter_groups(inr_in, lr_inr),
            {"params": alpha_in, "lr": meta_lr_code, "weight_decay": 0},
        ],
        lr=lr_inr,
        weight_decay=0,
    )

    optimizer_out = torch.optim.AdamW(
        [
            *parameter_groups(inr_out, lr_inr),
            {"params": alpha_out, "lr": meta_lr_code, "weight_decay": 0},
        ],
        lr=lr_inr,
        weight_decay=0,
    )

    best_loss = np.inf
    start_epoch = 0
    if resume is not None:
        from anchormix.checkpoint import restore_training_state
        start_epoch, best_loss = restore_training_state(
            resume, (alpha_in, alpha_out), (optimizer_in, optimizer_out))
        if start_epoch >= epochs:
            raise ValueError(f"Resume epoch {start_epoch} already reaches requested total epochs {epochs}")
        destination = RESULTS_DIR / f"{run_name}.pt"
        if not destination.exists():
            best_snapshot = resume
            source = Path(resume_path)
            if source.name.endswith(".last.pt"):
                best_source = source.with_name(source.name[:-len(".last.pt")] + ".pt")
                if not best_source.exists():
                    raise ValueError("Relocating a .last checkpoint also requires its companion best .pt checkpoint")
                best_snapshot = torch.load(best_source, map_location="cpu", weights_only=False)
                if (best_snapshot["epoch"] > resume["epoch"] or
                        best_snapshot.get("best_train_loss") != resume.get("best_train_loss")):
                    raise ValueError("Resume .last checkpoint and companion best checkpoint are inconsistent")
            temporary = destination.with_suffix(".tmp")
            torch.save(best_snapshot, temporary)
            temporary.replace(destination)

    for step in range(start_epoch, epochs):
        fit_train_mse_in = 0
        fit_test_mse_in = 0
        rel_train_mse_out = 0
        rel_test_mse_out = 0
        fit_train_mse_out = 0
        fit_test_mse_out = 0
        use_pred_loss = step % 20 == 0    # 每 20 个 epoch 计算一次相对损失
        visualize_every = int(cfg.optim.get("visualize_every", 200))
        step_show = visualize_every > 0 and step % visualize_every == 0

        for substep, (a_s, u_s, za_s, zu_s, coords, idx) in enumerate(
            train_loader
        ):
            inr_in.train()
            inr_out.train()

            a_s = a_s.to(device)
            u_s = u_s.to(device)
            za_s = za_s.to(device)
            zu_s = zu_s.to(device)
            coords = coords.to(device)
            n_samples = a_s.shape[0]   # 当前 batch 的样本数

            outputs = adapt(
                inr_in,
                coords,
                a_s,
                inner_steps,
                alpha_in,
                is_train=True,
                modulations=torch.zeros_like(za_s),   # latent 从零开始
                use_rel_loss=False,
            )

            optimizer_in.zero_grad()
            outputs["loss"].backward()
            nn.utils.clip_grad_value_(inr_in.parameters(), clip_value=1.0)  # 梯度裁剪
            optimizer_in.step()
            project_positions(inr_in)
            loss = outputs["loss"].cpu().detach()
            fit_train_mse_in += loss.item() * n_samples   # 加权累计训练损失

            z0 = outputs["modulations"].detach()

            if step_show and substep == 0:
                u_pred = inr_in.modulated_forward(
                    coords, z0
                )
                with torch.no_grad():
                    show(a_s, u_pred, coords, "train_input", num_examples=4)

            outputs = adapt(
                inr_out,
                coords,
                u_s,
                inner_steps,
                alpha_out,
                is_train=True,
                modulations=torch.zeros_like(zu_s),   # latent 从零开始
                use_rel_loss=use_pred_loss,
            )

            optimizer_out.zero_grad()
            outputs["loss"].backward()
            nn.utils.clip_grad_value_(inr_out.parameters(), clip_value=1.0)
            optimizer_out.step()
            project_positions(inr_out)
            loss = outputs["loss"].cpu().detach()
            fit_train_mse_out += loss.item() * n_samples   # 加权累计训练损失

            z1 = outputs["modulations"].detach()

            if use_pred_loss:
                rel_train_mse_out += outputs["rel_loss"].item() * n_samples

        train_loss_in = fit_train_mse_in / (ntrain)
        train_loss_out = fit_train_mse_out / (ntrain)

        if use_pred_loss:
            rel_train_loss_out = rel_train_mse_out / (ntrain)

        for substep, (a_s, u_s, za_s, zu_s, coords, idx) in enumerate(
            test_loader
        ):
            inr_in.eval()
            inr_out.eval()
            a_s = a_s.to(device)
            u_s = u_s.to(device)
            za_s = za_s.to(device)
            zu_s = zu_s.to(device)
            coords = coords.to(device)
            n_samples = a_s.shape[0]

            outputs = adapt(
                inr_in,
                coords,
                a_s,
                test_inner_steps,
                alpha_in,
                is_train=False,
                modulations=torch.zeros_like(za_s),
                use_rel_loss=False,
            )

            loss = outputs["loss"].cpu().detach()
            fit_test_mse_in += loss.item() * n_samples   # 加权累计测试损失
            z0 = outputs["modulations"].detach()

            if step_show and substep == 0:
                u_pred = inr_in.modulated_forward(
                    coords, z0
                )
                with torch.no_grad():
                    show(a_s, u_pred, coords, "test_input", num_examples=4)

            outputs = adapt(
                inr_out,
                coords,
                u_s,
                test_inner_steps,
                alpha_out,
                is_train=False,
                modulations=torch.zeros_like(zu_s),
                use_rel_loss=use_pred_loss,
            )

            loss = outputs["loss"].cpu().detach()
            fit_test_mse_out += loss.item() * n_samples
            z1 = outputs["modulations"].detach()

            if use_pred_loss:
                rel_test_mse_out += outputs["rel_loss"].item() * n_samples

        test_loss_in = fit_test_mse_in / (ntest)
        test_loss_out = fit_test_mse_out / (ntest)

        print(f"inr epoch={step + 1}/{epochs} train_in={train_loss_in:.8g} "
              f"train_out={train_loss_out:.8g} test_out={test_loss_out:.8g}", flush=True)

        if use_pred_loss:
            rel_test_loss_out = rel_test_mse_out / (ntest)

        COMMIT = not use_pred_loss   # 非相对损失 epoch 才提交本次日志
        wandb.log(
            {
                "train_loss_in": train_loss_in,
                "test_loss_in": test_loss_in,
                "train_loss_out": train_loss_out,
                "test_loss_out": test_loss_out,
            },
            step=step,
            commit=COMMIT
        )

        if use_pred_loss:
            COMMIT = True
            wandb.log(
                {
                    "train_rel_loss_out": rel_train_loss_out,
                    "test_rel_loss_out": rel_test_loss_out,
                })

        improved = train_loss_out < best_loss
        if improved:
            best_loss = train_loss_out
        independent = cfg.get("network_package") == "anchormix"
        if improved or independent:
            state = {
                "cfg": cfg,
                "epoch": step,
                "inr_in": inr_in.state_dict(),
                "inr_out": inr_out.state_dict(),
                "optimizer_inr_in": optimizer_in.state_dict(),
                "optimizer_inr_out": optimizer_out.state_dict(),
                "loss": test_loss_out,
                "alpha_in": alpha_in,
                "alpha_out": alpha_out,
                "best_train_loss": best_loss,
                **(training_metadata({"in": inr_in, "out": inr_out}, cfg) if independent else {}),
            }
            destinations = [RESULTS_DIR / f"{run_name}.pt"] if improved else []
            if independent:
                destinations.append(RESULTS_DIR / f"{run_name}.last.pt")
            for destination in destinations:
                temporary = destination.with_suffix(".tmp")
                torch.save(state, temporary)
                temporary.replace(destination)

    return test_loss_out

if __name__ == "__main__":
    main()
