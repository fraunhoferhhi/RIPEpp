import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

SEED = 32000

import collections
import os
import random

import hydra
from hydra.utils import instantiate
from lightning.fabric import Fabric
from utils.utils import check_all_received_grads

os.environ["PYTHONHASHSEED"] = str(SEED)

import numpy as np
import psutil
import torch
import tqdm
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

import wandb
from ripepp import utils
from ripepp.losses.rl_loss import rl_loss

# from x_dd.utils.hpatches_utils import eval_hpatches
from ripepp.utils.checkpoint import load_checkpoint, save_checkpoint
from ripepp.utils.utils import (
    dry_run_print,
    gridify,
    save_hydra_config_file,
    save_source_code,
)
from ripepp.utils.wandb_utils import get_flattened_wandb_cfg

log = utils.get_pylogger(__name__)
from pathlib import Path

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_directory_if_missing(f):
    if not os.path.exists(f):
        os.makedirs(f)
    else:
        log.info(f"Directory {f} already exists")


def unpack_batch(batch):
    src_image = batch["src_image"]
    trg_image = batch["trg_image"]

    src_mask = batch.get("src_mask", None)
    trg_mask = batch.get("trg_mask", None)

    label = batch.get("label", None)

    return src_image, trg_image, src_mask, trg_mask, label


def adapt_bs_and_gradient_acc_depending_on_GPU(cfg):
    total = torch.cuda.get_device_properties(0).total_memory

    total_in_gb = total / (1024**3)

    if total_in_gb < 45.0:
        log.info(
            f"GPU memory is {total_in_gb:.2f} GB, which is less than 45 GB. Adjusting batch size and gradient accumulation."
        )
        cfg.batch_size = 4
        cfg.num_grad_accs = 2
    else:
        log.info(
            f"GPU memory is {total_in_gb:.2f} GB, which is sufficient for the configured batch size and gradient accumulation."
        )

    return cfg


@hydra.main(config_path="../conf/", config_name="config", version_base=None)
def train(cfg):
    """This function implements custom training loop and different training strategies for the DEAL
    detector & descriptor, alongside the custom losses for joint detection and description of
    deformation-aware keypoints.

    For detailed discussion please refer to the paper. All hyperparams defined here were used for
    the experiments in the paper
    """
    #  Prepare model, data and hyperparms

    strategy = "ddp" if cfg.num_gpus > 1 else "auto"

    if cfg.dry_run:
        log.info("Running in dry run mode")
        log.info("No saving of models")
        log.info(f"Number of steps: {cfg.num_steps}")

    # if not cfg.dry_run:  # this is no sophisticated function, simply switches for the A100 40 GB and A100 80GB
    #     cfg = adapt_bs_and_gradient_acc_depending_on_GPU(cfg)

    # sanity checks for training mode and dataset config
    assert cfg.train_mode in ["positive_only", "positive_and_negative"], (
        "train_mode must be either 'positive_only' or 'positive_and_negative'"
    )

    log.info(f"Training mode: {cfg.train_mode}")

    # Ensure gradient accumulation and DDP are mutually exclusive
    if cfg.num_gpus > 1 and cfg.num_grad_accs > 1:
        raise ValueError(
            f"Gradient accumulation (num_grad_accs={cfg.num_grad_accs}) and multi-GPU DDP (num_gpus={cfg.num_gpus}) "
            "are mutually exclusive. Use one or the other, not both:\n"
            "  - For single GPU: set num_gpus=1 and use num_grad_accs > 1\n"
            "  - For multi-GPU: set num_gpus > 1 and num_grad_accs=1"
        )

    output_dir = Path(cfg.output_dir)
    experiment_name = os.environ.get("SLURM_JOB_ID", "NO_ID") + cfg.name

    # setup logger
    wandb_logger = wandb.init(
        project=cfg.project_name,
        name=experiment_name,
        config=get_flattened_wandb_cfg(cfg),
        dir=cfg.wandb_output_dir,
        mode=cfg.wandb_mode,
    )

    # start lightning fabric
    fabric = Fabric(
        accelerator=cfg.accelerator,
        devices=cfg.num_gpus,
        precision=cfg.precision,
        strategy=strategy,
    )
    fabric.launch()

    # check if output directory exists
    if fabric.is_global_zero:
        create_directory_if_missing(output_dir)
        # save final config file
        save_hydra_config_file(cfg, output_dir / "final_config.yaml")
        # save the code as zip
        save_source_code(root, output_dir)

    min_nums_matches = {"homography": 4, "fundamental": 8, "fundamental_7pt": 7}
    min_num_matches = min_nums_matches[cfg.transformation_model]
    log.info(f"Minimum number of matches for {cfg.transformation_model} is {min_num_matches}")

    batch_size = cfg.batch_size
    steps = cfg.num_steps  # originally 80_000

    num_grad_accs = (
        cfg.num_grad_accs
    )  # this performs grad accumulation to simulate larger batch size, set to 1 to disable;

    if cfg.dry_run:
        batch_size = 2

    # instantiate dataset
    ds = instantiate(cfg.data)

    if cfg.train_mode == "positive_only":
        assert ds.positive_only, "data.positive_only must be True when train_mode is 'positive_only'"
    elif cfg.train_mode == "positive_and_negative":
        assert not ds.positive_only, "data.positive_only must be False when train_mode is 'positive_and_negative'"

    g = torch.Generator()
    g.manual_seed(SEED)

    # prepare dataloader
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True if not cfg.dry_run else False,
        drop_last=True,
        persistent_workers=False,
        num_workers=0 if cfg.dry_run else cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    dl = fabric.setup_dataloaders(dl)
    i_dl = iter(dl)

    # matching pipeline
    matcher = instantiate(cfg.matcher)
    robust_estimator = instantiate(cfg.estimator)
    geometric_matching = instantiate(cfg.geometric_matching)(matcher=matcher, robust_estimator=robust_estimator)

    # create network
    net = instantiate(cfg.network)(
        net=instantiate(cfg.backbones),
        descriptor_dim=cfg.descriptor_dim,
    ).train()

    # check if training runs with DDP and multiple GPUs
    if cfg.num_gpus > 1:
        raise NotImplementedError("DDP with multiple GPUs is not supported in this version.")
        # log.info(f"Training with DDP on {cfg.num_gpus} GPUs -> replacing BatchNorm with SyncBatchNorm!")
        # # replace BatchNorm layers with SyncBatchNorm
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    list_desc_losses = []
    if len(cfg.descriptor_loss) > 0:
        for loss_cfg in cfg.descriptor_loss:
            list_desc_losses.append(
                {
                    "name": loss_cfg.name,
                    "desc_key": loss_cfg.desc_key,
                    "weight": loss_cfg.weight,
                    "module": instantiate(loss_cfg.loss),
                }
            )

    # get num parameters
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.info(f"Number of parameters: {num_params}")

    num_steps_already_taken = 0
    if cfg.load_model:
        # check if the path exists
        if not os.path.exists(cfg.load_model):
            raise FileNotFoundError(f"Model path {cfg.load_model} does not exist.")

        # check if the path is a folder
        if os.path.isdir(cfg.load_model):
            # find the latest checkpoint in the folder
            checkpoints = list(Path(cfg.load_model).glob("*.pth"))
            if len(checkpoints) == 0:
                raise FileNotFoundError(f"No checkpoint found in folder {cfg.load_model}")

            # check if checkpoint containing "best" in its name exists, if yes, remove it
            checkpoints = [ckpt for ckpt in checkpoints if "best" not in ckpt.stem]

            # sort list of checkpoints by there name, extracting the step number
            checkpoints.sort(key=lambda x: int(x.stem.split("_")[-2 if "final" in x.stem else -1]))

            path_model = checkpoints[-1]
        else:
            path_model = cfg.load_model

        log.info(f"Loading model from {path_model}")
        checkpoint = load_checkpoint(path_model)
        net.load_state_dict(checkpoint["state_dict"])

        if not cfg.dry_run:
            path_model = Path(path_model)
            filename = path_model.stem
            num_steps_already_taken = checkpoint.get("step") or int(
                filename.split("_")[-2 if "final" in filename else -1]
            )
            log.info(f"Model load at step {num_steps_already_taken}")

    # log gradients
    if cfg.log_gradients:
        wandb_logger.watch(net)

    fp_penalty = cfg.fp_penalty  # small penalty for not finding a match
    kp_penalty = cfg.kp_penalty  # small penalty for low logprob keypoints
    kp_penalty_inliers_only = cfg.kp_penalty_inliers_only

    layers_to_train = filter(lambda x: x.requires_grad, net.parameters())

    opt_pi = AdamW(
        [
            {"params": layers_to_train, "lr": cfg.lr},
        ],
        weight_decay=1e-5,
    )
    net, opt_pi = fabric.setup(net, opt_pi)

    # mark get_descriptors as forward method
    net.mark_forward_method("get_descriptors")
    net.mark_forward_method("detectAndCompute")

    if cfg.lr_scheduler:
        scheduler = instantiate(cfg.lr_scheduler)(
            initial_lr=cfg.lr,
            final_lr=cfg.final_lr,
            optimizer=opt_pi,
            steps_init=num_steps_already_taken,
        )
    else:
        scheduler = None

    val_benchmark = instantiate(cfg.val)(
        conf_inference=cfg.conf_inference,
        output_dir=output_dir,
        matcher=matcher,
        dry_run=cfg.dry_run,
    )
    best_auc = 0.0

    # Initialize curriculum learning
    curriculum_enabled = cfg.curriculum_learning.enabled
    if curriculum_enabled:
        topK = cfg.curriculum_learning.topk_init
        topK_max = cfg.curriculum_learning.topk_max
        topK_increment = cfg.curriculum_learning.topk_increment
        increment_every_n_steps = cfg.curriculum_learning.increment_every_n_steps
        global_selection = cfg.curriculum_learning.global_selection
        reward_history_size = cfg.curriculum_learning.reward_history_size

        # Initialize reward history for stable threshold computation with small batches
        reward_history = collections.deque(maxlen=reward_history_size)

        if global_selection and cfg.num_gpus > 1:
            log.info(
                f"Curriculum learning enabled (GLOBAL mode across {cfg.num_gpus} GPUs): starting at {topK}%, max {topK_max}%, increment {topK_increment}% every {increment_every_n_steps} steps"
            )
        else:
            log.info(
                f"Curriculum learning enabled (LOCAL mode with history_size={reward_history_size}): starting at {topK}%, max {topK_max}%, increment {topK_increment}% every {increment_every_n_steps} steps"
            )
    else:
        topK = 100  # Use all samples if curriculum learning is disabled
        global_selection = False
        reward_history = None

    src = trg = src_mask = trg_mask = label = None

    if cfg.dry_run:
        src, trg, src_mask, trg_mask, label = unpack_batch(next(i_dl))
        src, trg, src_mask, trg_mask, label = unpack_batch(next(i_dl))

        # IMPORTANT: under DDP/SyncBatchNorm, running the model forward can involve collectives.
        # If only rank 0 calls into the model, rank 0 can block forever waiting for other ranks.
        # So: all ranks run the forward; only global-zero writes images to disk.
        for b in range(batch_size):
            dry_run_print(
                src[b],
                trg[b],
                net,
                conf_inference=cfg.conf_inference,
                i=b,
                mode="start_pos",
                transformation_model=cfg.transformation_model,
                output_path=output_dir,
                save_outputs=fabric.is_global_zero,
            )

    log.info("Starting training loop")

    ma_skipped_batches = collections.deque(maxlen=100)

    opt_pi.zero_grad()

    # initialize scheduler
    alpha_scheduler = instantiate(cfg.alpha_scheduler)
    beta_scheduler = instantiate(cfg.beta_scheduler)
    inl_th_scheduler = instantiate(cfg.inl_th)
    inlier_reward_scheduler = instantiate(cfg.inlier_reward_scheduler)
    outlier_penalty_scheduler = instantiate(cfg.outlier_penalty_scheduler)
    descriptor_blend_scheduler = instantiate(cfg.descriptor_blend_scheduler)
    heatmap_entropy_scheduler = instantiate(cfg.heatmap_entropy_scheduler)

    heatmap_regularization_loss_fn = (
        instantiate(cfg.heatmap_regularization).to(fabric.device) if "heatmap_regularization" in cfg else None
    )

    if cfg.use_distance_based_rewards:
        log.info("Using distance-based rewards")
        kernel_fn = instantiate(cfg.reward_kernel)
    else:
        kernel_fn = None

    # ======  Training Loop  ======
    # check if the model is in training mode
    net.train()

    last_step = num_steps_already_taken

    with tqdm.tqdm(total=steps) as pbar:
        for i_step in range(num_steps_already_taken, steps):
            last_step = i_step
            alpha = alpha_scheduler(i_step)
            beta = beta_scheduler(i_step)
            inl_th = inl_th_scheduler(i_step)
            inlier_reward = inlier_reward_scheduler(i_step)
            outlier_penalty = outlier_penalty_scheduler(i_step)
            heatmap_entropy_weight = heatmap_entropy_scheduler(i_step)

            if scheduler:
                scheduler.step()

            # Initialize vars for current step
            # We need to handle batching because the description can have arbitrary number of keypoints
            sum_reward_batch = 0
            loss = None
            loss_policy_stack = None
            loss_kp_stack = None

            # Track per-sample rewards for curriculum learning
            sample_rewards = []
            num_selected_samples = 0  # Track number of samples selected by curriculum learning

            if not cfg.dry_run:  # reuse same batch during dry run
                try:
                    batch = next(i_dl)
                except StopIteration:
                    if hasattr(ds, "resample"):
                        ds.resample()
                    i_dl = iter(dl)
                    batch = next(i_dl)

                src, trg, src_mask, trg_mask, label = unpack_batch(batch)

            assert (
                src is not None
                and trg is not None
                and src_mask is not None
                and trg_mask is not None
                and label is not None
            )

            # pos_kpts, pos_logprobs, pos_selected_mask, pos_mask_padding_grid, pos_logits_selected,
            src_out = net(src, src_mask)
            # neg_kpts, neg_logprobs, neg_selected_mask, neg_mask_padding_grid, neg_logits_selected,
            trg_out = net(trg, trg_mask)

            H, W = src.shape[2:4]

            desc_blend_weight = descriptor_blend_scheduler(i_step)

            src_desc = net.get_descriptors(src_out, desc_blend_weight)
            trg_desc = net.get_descriptors(trg_out, desc_blend_weight)

            src_out = {**src_out, **src_desc}
            trg_out = {**trg_out, **trg_desc}

            # remove keypoints that are in padding
            src_out["mask_matching"] = src_out["mask_selected"] & src_out["mask_padding"]
            trg_out["mask_matching"] = trg_out["mask_selected"] & trg_out["mask_padding"]

            # ======  Heatmap entropy regularization  ======
            loss_heatmap_entropy = None
            if heatmap_entropy_weight > 0:
                window_size = net.module.detector.window_size if hasattr(net, "module") else net.detector.window_size

                for out in [src_out, trg_out]:
                    heatmap = out["heatmap"]  # [B, 1, H, W]
                    B_h = heatmap.shape[0]
                    h_grid = heatmap.shape[2] // window_size
                    w_grid = heatmap.shape[3] // window_size

                    # [B, 1, h_grid, w_grid, window_size**2]
                    patch_logits = gridify(heatmap, window_size)
                    patch_probs = torch.softmax(patch_logits, dim=-1)

                    # Entropy per patch: -sum(p * log(p)), shape [B, 1, h_grid, w_grid]
                    entropy = -(patch_probs * torch.log(patch_probs + 1e-8)).sum(dim=-1)

                    # mask_matching is [B, h_grid*w_grid], reshape to match entropy
                    mask = out["mask_matching"].view(B_h, 1, h_grid, w_grid)

                    if mask.any():
                        entropy_selected = entropy[mask].mean()
                    else:
                        entropy_selected = torch.tensor(0.0, device=heatmap.device)

                    if loss_heatmap_entropy is None:
                        loss_heatmap_entropy = entropy_selected
                    else:
                        loss_heatmap_entropy = loss_heatmap_entropy + entropy_selected

                loss_heatmap_entropy = loss_heatmap_entropy / 2.0  # average over src and trg

            # ======  Heatmap regularization  ======
            loss_heatmap_regularization = None
            if heatmap_regularization_loss_fn is not None:
                for img, out in [(src, src_out), (trg, trg_out)]:
                    gp_loss = heatmap_regularization_loss_fn(out["heatmap"], img)
                    loss_heatmap_regularization = (
                        gp_loss if loss_heatmap_regularization is None else loss_heatmap_regularization + gp_loss
                    )
                loss_heatmap_regularization = loss_heatmap_regularization / 2.0  # average over src and trg

            # ======  Matching  ======
            # get putative matches between anchor and positive and between anchor and negative
            src_trg_matching = geometric_matching(src_out, trg_out, inl_th, None)

            desc_loss_accumulator = {spec["name"]: [] for spec in list_desc_losses}

            for b in range(batch_size):
                # ======  Policy loss ======
                src_trg_current_loss_policy, src_trg_dense_rewards_sum = rl_loss(
                    src_trg_matching,
                    src_out,
                    trg_out,
                    b,
                    label,
                    cfg,
                    fp_penalty * alpha,
                    inlier_reward,
                    outlier_penalty,
                    kernel_fn,
                )

                if src_trg_current_loss_policy is None:
                    ma_skipped_batches.append(1)
                    sample_rewards.append(0.0)  # Track zero reward for skipped samples
                    continue
                else:
                    ma_skipped_batches.append(0)

                src_trg_current_loss_policy = src_trg_current_loss_policy.mean()

                sum_reward_batch += src_trg_dense_rewards_sum
                sample_rewards.append(src_trg_dense_rewards_sum)  # Track reward for this sample

                loss_policy_stack = (
                    src_trg_current_loss_policy
                    if loss_policy_stack is None
                    else torch.hstack((loss_policy_stack, src_trg_current_loss_policy))
                )

                # ======  Descriptor loss (optional)  ======
                for spec in list_desc_losses:
                    desc_key = spec["desc_key"]
                    if desc_key not in src_out or desc_key not in trg_out:
                        raise RuntimeError(f"Descriptor key {desc_key} not found in network output.")
                    if src_out[desc_key] is None or trg_out[desc_key] is None:
                        raise RuntimeError(f"Descriptor key {desc_key} is None in network output.")

                    src_for_loss = {**src_out, "desc": src_out[desc_key]}
                    trg_for_loss = {**trg_out, "desc": trg_out[desc_key]}

                    loss_desc = spec["module"](
                        inp_image1=src_for_loss,
                        inp_image2=trg_for_loss,
                        inp_match_1_2=src_trg_matching,
                        b=b,
                        label=label,
                    )

                    desc_loss_accumulator[spec["name"]].append(loss_desc)

                # ======  Descriptor loss end  ======
                if kp_penalty != 0.0:
                    if kp_penalty_inliers_only:
                        # Only penalize keypoints that are RANSAC inliers
                        idx_matches = src_trg_matching["idx_matches"][b]
                        ransac_inliers = src_trg_matching["ransac_inliers"][b]

                        if idx_matches is not None and ransac_inliers is not None and ransac_inliers.sum() > 0:
                            inlier_matches = idx_matches[ransac_inliers]
                            src_inlier_idx = inlier_matches[:, 0]
                            trg_inlier_idx = inlier_matches[:, 1]

                            src_logprobs_inlier = src_out["logprobs"][b][src_inlier_idx]
                            trg_logprobs_inlier = trg_out["logprobs"][b][trg_inlier_idx]

                            loss_kp = (src_logprobs_inlier * kp_penalty * beta).mean()
                            loss_kp = loss_kp + (trg_logprobs_inlier * kp_penalty * beta).mean()

                            loss_kp_stack = loss_kp if loss_kp_stack is None else torch.hstack((loss_kp_stack, loss_kp))
                    else:
                        # Original behavior: penalize all selected keypoints in valid regions
                        loss_kp = (
                            src_out["logprobs"][b][src_out["mask_matching"][b]]
                            * torch.full_like(
                                src_out["logprobs"][b][src_out["mask_matching"][b]],
                                kp_penalty * beta,
                            )
                        ).mean()
                        loss_kp = (
                            loss_kp
                            + (
                                trg_out["logprobs"][b][trg_out["mask_matching"][b]]
                                * torch.full_like(
                                    trg_out["logprobs"][b][trg_out["mask_matching"][b]],
                                    kp_penalty * beta,
                                )
                            ).mean()
                        )

                        loss_kp_stack = loss_kp if loss_kp_stack is None else torch.hstack((loss_kp_stack, loss_kp))

            # loss = -(loss_vals.mean() + loss_kp.mean())
            if loss_policy_stack is None:
                raise RuntimeError("All batches were skipped due to no matches found.")

            # Apply curriculum learning: select top-K samples based on rewards
            # NOTE: Works with gradient accumulation (num_grad_accs > 1) and DDP (num_gpus > 1)
            # - Gradient accumulation: Each mini-batch independently filters before backward()
            # - DDP with global_selection=True: Synchronizes rewards across GPUs for global top-K
            # - DDP with global_selection=False: Uses reward history for stable threshold
            if curriculum_enabled and len(sample_rewards) > 1:
                # Convert rewards to tensor
                rewards_tensor = torch.tensor(sample_rewards, device=fabric.device)

                # For multi-GPU with global selection, gather rewards from all GPUs
                if global_selection and cfg.num_gpus > 1:
                    # Gather rewards from all GPUs to compute global threshold
                    # all_gather returns list of tensors, one per GPU
                    all_rewards = fabric.all_gather(rewards_tensor)  # Shape: [num_gpus, batch_size_per_gpu]

                    # Flatten to get all rewards globally
                    global_rewards = torch.cat([r for r in all_rewards])  # Shape: [num_gpus * batch_size_per_gpu]

                    # Calculate global top-K threshold
                    total_samples = len(global_rewards)
                    select_topB_global = max(int(total_samples * topK / 100), 1)
                    sorted_global_rewards, _ = torch.sort(global_rewards, descending=True)
                    threshold_reward = sorted_global_rewards[min(select_topB_global - 1, total_samples - 1)]
                else:
                    # Local selection: use reward history for stable threshold computation
                    # Add current batch rewards to history
                    reward_history.extend(sample_rewards)

                    # Compute threshold from reward history
                    if len(reward_history) >= topK:  # Ensure we have enough samples
                        historical_rewards = torch.tensor(list(reward_history), device=fabric.device)
                        select_topB_historical = max(int(len(historical_rewards) * topK / 100), 1)
                        sorted_historical, _ = torch.sort(historical_rewards, descending=True)
                        threshold_reward = sorted_historical[
                            min(select_topB_historical - 1, len(historical_rewards) - 1)
                        ]
                    else:
                        # Not enough history yet, use current batch only
                        num_samples = len(sample_rewards)
                        select_topB = max(int(num_samples * topK / 100), 1)
                        sorted_rewards, _ = torch.sort(rewards_tensor, descending=True)
                        threshold_reward = sorted_rewards[min(select_topB - 1, num_samples - 1)]

                # Create mask: 1 for samples to include, 0 for samples to exclude
                # Apply the (potentially global) threshold to local samples
                mask_topk = (rewards_tensor >= threshold_reward).float()
            else:
                # No curriculum learning: use all samples
                num_selected_samples = len(sample_rewards)
                mask_topk = torch.ones_like(loss_policy_stack, device=fabric.device)

            # Apply mask to policy loss
            loss_policy_masked = loss_policy_stack * mask_topk

            # Compute mean only over selected samples
            num_selected = mask_topk.sum()
            num_selected_samples = int(num_selected.item())  # Track for logging
            loss = loss_policy_masked.sum() / num_selected if num_selected > 0 else loss_policy_stack.mean()

            loss = -loss  # need the negative sign because we want to maximize rewards, but optimizers minimize loss

            # Apply mask to keypoint penalty loss if present
            if loss_kp_stack is not None:
                loss_kp_masked = loss_kp_stack * mask_topk
                loss = loss + (loss_kp_masked.sum() / num_selected if num_selected > 0 else loss_kp_stack.mean())

            desc_loss_logs = {}

            total_desc_loss = torch.tensor(0.0, device=fabric.device)
            for spec in list_desc_losses:
                spec_losses = desc_loss_accumulator.get(spec["name"], [])
                spec_losses = torch.stack(spec_losses)

                spec_losses_masked = spec_losses * mask_topk
                spec_loss = spec_losses_masked.sum() / num_selected if num_selected > 0 else spec_losses.mean()

                total_desc_loss = total_desc_loss + spec["weight"] * spec_loss
                desc_loss_logs[f"loss_desc_{spec['name']}"] = spec_loss.detach()

            loss = loss + total_desc_loss

            if loss_heatmap_entropy is not None:
                loss = loss + heatmap_entropy_weight * loss_heatmap_entropy

            if loss_heatmap_regularization is not None:
                loss = loss + loss_heatmap_regularization

            if i_step % cfg.update_interval == 0 and i_step != 0:  # update the progress bar every 100 steps
                desc_loss_val = total_desc_loss.item() if total_desc_loss is not None else 0.0
                pbar.set_description(
                    f"LP: {loss.item():.4f} - Desc: {desc_loss_val:.3f}, #mRwd: {sum_reward_batch / batch_size:.1f}"
                )
                pbar.update(cfg.update_interval)

            # backward pass
            loss /= num_grad_accs
            fabric.backward(loss)

            if i_step % num_grad_accs == 0:
                if cfg.dry_run and not check_all_received_grads(
                    net,
                    [
                        "dedode_desc_network",
                        "external_descriptor",
                    ],
                ):
                    raise ValueError("Some parameters did not receive gradients")

                opt_pi.step()
                opt_pi.zero_grad()

            # Update curriculum learning topK
            if curriculum_enabled and i_step > 0 and i_step % increment_every_n_steps == 0:
                if topK < topK_max:
                    topK = min(topK_max, topK + topK_increment)
                    log.info(f"Step {i_step}: Curriculum learning topK increased to {topK}%")

            if i_step % cfg.log_interval == 0 and fabric.is_global_zero:
                proc = psutil.Process(os.getpid())
                rss_main = proc.memory_info().rss
                rss_workers = sum(c.memory_info().rss for c in proc.children(recursive=True))
                log_payload = {
                    "rss_gb_main": rss_main / 1e9,
                    "rss_gb_workers": rss_workers / 1e9,
                    "rss_gb_total": (rss_main + rss_workers) / 1e9,
                    "loss": loss.item(),
                    "loss_policy": -loss_policy_stack.mean().item(),
                    "loss_kp": loss_kp_stack.mean().item() if loss_kp_stack is not None else 0.0,
                    "loss_desc": total_desc_loss.item() if total_desc_loss is not None else 0.0,
                    "mean_reward": sum_reward_batch / batch_size,
                    "ma_skipped_batches": sum(ma_skipped_batches) / len(ma_skipped_batches),
                    "inl_th": inl_th,
                    "kp_penalty": kp_penalty * beta,
                    "inlier_reward": inlier_reward,
                    "outlier_penalty": outlier_penalty,
                    "desc_blend_weight": desc_blend_weight,
                    "loss_heatmap_entropy": loss_heatmap_entropy.item() if loss_heatmap_entropy is not None else 0.0,
                    "heatmap_entropy_weight": heatmap_entropy_weight,
                    "loss_heatmap_regularization": loss_heatmap_regularization.item()
                    if loss_heatmap_regularization is not None
                    else 0.0,
                    **(scheduler.get_lr_dict() if scheduler else {"lr": cfg.lr}),
                }

                # Add curriculum learning metrics
                if curriculum_enabled:
                    log_payload["curriculum_topK"] = topK
                    log_payload["curriculum_selected_samples"] = num_selected_samples

                for loss_name, loss_value in desc_loss_logs.items():
                    log_payload[loss_name] = loss_value.item()

                wandb_logger.log(
                    log_payload,
                    step=i_step,
                )

            if i_step % cfg.save_model_every_n_steps == 0 and not cfg.dry_run and fabric.is_global_zero:
                save_checkpoint(
                    state_dict=net.state_dict(),
                    config=cfg,
                    path=output_dir / ("model_" + cfg.name + "_%06d" % i_step + ".pth"),
                    step=i_step,
                )

            # Validation has to run for all ranks to avoid deadlocks caused by collective operations
            # but only global zero logs and saves the best model
            if i_step % cfg.val_interval == 0:
                val_benchmark.conf_inference["desc_blend_ratio"] = desc_blend_weight

                val_benchmark.evaluate(net, fabric.device, progress_bar=False)
                if fabric.is_global_zero:
                    val_benchmark.log_results(logger=wandb_logger, step=i_step)
                    if cfg.val_interval_plot > 0:
                        if i_step % cfg.val_interval_plot == 0:
                            val_benchmark.plot_results(logger=wandb_logger, step=i_step)

                    result = val_benchmark.get_auc(threshold=5)
                    if result > best_auc:
                        best_auc = result
                        save_checkpoint(
                            state_dict=net.state_dict(),
                            config=cfg,
                            path=output_dir / f"model_{cfg.name}_best.pth",
                            step=i_step,
                            metadata={"best_auc": float(best_auc)},
                        )

                        best_auc = result

                        log.info(f"New best AUC: {best_auc:.4f}")

                # check if the model is in training mode
                net.train()

            # break the loop if in dry run mode
            if i_step == cfg.early_stopping_at_step:
                break

    # save the model
    save_checkpoint(
        state_dict=net.state_dict(),
        config=cfg,
        path=output_dir / ("model" + "_" + cfg.name + "_" + str(last_step + 1) + "_final" + ".pth"),
        step=last_step + 1,
    )

    if cfg.dry_run and src is not None and trg is not None:
        if fabric.is_global_zero:
            cfg.conf_inference["desc_blend_ratio"] = desc_blend_weight

            for b in range(batch_size):
                dry_run_print(
                    src[b],
                    trg[b],
                    net,
                    conf_inference=cfg.conf_inference,
                    i=b,
                    mode="end_pos",
                    transformation_model=cfg.transformation_model,
                    output_path=output_dir,
                )


if __name__ == "__main__":
    train()
