from distutils import log
import einops
import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

import wandb
from utils.video import VideoRecorder
import pickle
from datasets.core import TrajectoryEmbeddingDataset, split_traj_datasets
from datasets.vqbet_repro import TrajectorySlicerDataset
from envs.venv import SubprocVectorEnv
from utils.normalizer import LinearNormalizer
from datetime import timedelta, datetime
from accelerate import Accelerator
import logging
from accelerate import InitProcessGroupKwargs, DistributedDataParallelKwargs
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
import torch.distributed as dist

os.environ["WANDB_START_METHOD"] = "thread"
logger = logging.getLogger(__name__)

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


@hydra.main(config_path="eval_configs", version_base="1.2")
def main(cfg):
    if HydraConfig.get().mode == RunMode.MULTIRUN:
        logger.info(" Multirun setup begin...")
        logger.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
        logger.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
        # ==== init ddp process group ====
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(hours=2),  # Extended timeout for eval variance
            )
            log.info("Multirun setup completed.")
        except Exception as e:
            log.error(f"DDP setup failed: {e}")
            raise
        torch.distributed.barrier()
    process_group_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(hours=2),  # Extended timeout for eval variance
        )
    dist_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="wandb", kwargs_handlers=[process_group_kwargs, dist_kwargs]
    )
    logger.info(f"Mixed precision: {accelerator.mixed_precision}")
    logger.info(
        f"rank: {accelerator.local_process_index}"
    )
    device = accelerator.device
    logger.info(f"device: {device}")
    assert cfg.batch_size % accelerator.num_processes == 0, (
        "Batch size must be divisible by the number of processes. "
        f"Batch_size: {cfg.batch_size} num_processes: {accelerator.num_processes}."
    )

    OmegaConf.set_struct(cfg, False)
    cfg.effective_batch_size = cfg.batch_size
    cfg.gpu_batch_size = cfg.batch_size // accelerator.num_processes
    OmegaConf.set_struct(cfg, True)

    accelerator.wait_for_everyone()
    
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    use_diffusion = "diffusion" in cfg.model['_target_']

    save_path = Path(".")
    model_name = os.getcwd().split("outputs/")[-1]
    
    # Hydra changes working directory to the run directory, so just use current directory
    if accelerator.is_main_process:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=model_name,  # Use current directory name
        )

    print("Saving to {}".format(os.getcwd()))
    video = VideoRecorder(dir_name=save_path)

    # init datasets
    dataset = hydra.utils.instantiate(cfg.dataset)
    encoder = hydra.utils.instantiate(cfg.encoder)
    # encoder = encoder.to(cfg.device).eval()
    encoder = accelerator.prepare(encoder)

    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=cfg.optim.betas,
    ) # init optimizer attribute

    if use_diffusion:
        actions = dataset.get_all_actions()
        action_normalizer = LinearNormalizer()
        action_normalizer.fit(actions)
        cbet_model.set_normalizer(action_normalizer)

    if cfg.load_path: # TODO: not checked for diffusion 
        cbet_model.load_model(Path(cfg.load_path))
        optimizer = cbet_model.optim
    else:
        optimizer = cbet_model.configure_optimizers(
            weight_decay=cfg.optim.weight_decay,
            learning_rate=cfg.optim.lr,
            betas=cfg.optim.betas,
        )
    
    cbet_model, optimizer = accelerator.prepare(cbet_model, optimizer)

    # No need to create directory or wait - Hydra already set up the working directory
    accelerator.wait_for_everyone()
    
    print("Saving to {}".format(os.getcwd()))
    video = VideoRecorder(dir_name=save_path)

    # init datasets
    dataset = hydra.utils.instantiate(cfg.dataset)
    train_data, test_data = split_traj_datasets(
        dataset,
        train_fraction=cfg.train_fraction,
        random_seed=cfg.seed,
    )
    use_libero_goal = cfg.data.get("use_libero_goal", False)

    precompute_embeddings = cfg.get("precompute_embeddings", True)
    if precompute_embeddings:
        train_data = TrajectoryEmbeddingDataset(
            encoder, train_data, device='cpu', embed_goal=use_libero_goal
        )
        test_data = TrajectoryEmbeddingDataset(
            encoder, test_data, device='cpu', embed_goal=use_libero_goal
        )
    traj_slicer_kwargs = {
        "window": cfg.data.window_size,
        "action_window": cfg.data.action_window_size,
        "vqbet_get_future_action_chunk": cfg.data.vqbet_get_future_action_chunk,
        "future_conditional": (cfg.data.goal_conditional == "future"),
        "min_future_sep": cfg.data.action_window_size,
        "future_seq_len": cfg.data.future_seq_len,
        "use_libero_goal": use_libero_goal,
    }

    train_data = TrajectorySlicerDataset(train_data, **traj_slicer_kwargs)
    test_data = TrajectorySlicerDataset(test_data, **traj_slicer_kwargs)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.gpu_batch_size, shuffle=True, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.gpu_batch_size, shuffle=False, pin_memory=False
    )
    log.info(f"dataloader batch size: {cfg.gpu_batch_size}")

    train_loader = accelerator.prepare(train_loader)
    test_loader = accelerator.prepare(test_loader)

    env_fn = lambda: hydra.utils.instantiate(cfg.env.gym)
    env = SubprocVectorEnv([env_fn for _ in range(cfg.num_envs)])
    if "use_libero_goal" in cfg.data:
        with torch.no_grad():
            # calculate goal embeddings for each task
            goals_cache = []
            for i in range(10):
                idx = i * 50
                last_obs, _, _ = dataset.get_frames(idx, [-1])  # 1 V C H W
                last_obs = last_obs.to(cfg.device)
                embd = encoder(last_obs)[0]  # V P E
                assert embd.ndim == 3, "expect V P E here"
                embd = einops.rearrange(embd, "V P E -> (V P) E") # don't flatten patch dim
                goals_cache.append(embd)

        def goal_fn(goal_idx):
            return goals_cache[goal_idx]
    else:
        empty_tensor = torch.zeros(1)

        def goal_fn(goal_idx):
            return empty_tensor


    @torch.no_grad()
    def eval_on_env(
        cfg,
        num_evals=cfg.num_env_evals,
        videorecorder=None,
        epoch=None,
    ):
        def embed(enc, obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device)  # N V C H W
            return einops.rearrange(enc(obs), "N V P E -> N (V P) E")
        assert num_evals % cfg.num_envs == 0, "num_evals must be multiple of num_envs"

        avg_reward = 0
        action_list = []
        completion_id_list = []
        avg_max_coverage = []
        avg_final_coverage = []
        env.seed([cfg.seed + i for i in range(cfg.num_envs)])
        num_batches = num_evals // cfg.num_envs
        for goal_idx in range(num_batches):
            if videorecorder is not None:
                videorecorder.init(enabled=True)
            obs_stack = deque(maxlen=cfg.eval_window_size)
            this_obs = env.reset(goal_idx=goal_idx)  # N V C H W
            print(f"Eval on goal {goal_idx}/{num_batches}, {cfg.num_envs} episodes")
            assert (
                this_obs.min() >= 0 and this_obs.max() <= 1
            ), "expect 0-1 range observation"
            this_obs_enc = embed(encoder, this_obs)  # N (V P) E. from now on V is folded into P
            obs_stack.append(this_obs_enc)
            done, total_reward = np.array([False]), 0
            goal = goal_fn(goal_idx)  # V C H W
            # TODO: replace done with a horizon. This assumes all envs are done at the same time
            while not done.all():
                obs = torch.stack(tuple(obs_stack)).float().to(cfg.device)
                obs = einops.rearrange(obs, "T N P E -> N T P E")
                goal = torch.as_tensor(goal, dtype=torch.float32, device=cfg.device)
                # goal = embed(encoder, goal)
                goal = einops.repeat(goal, '... -> N T ...', N=cfg.num_envs, T=cfg.eval_window_size)
                action, _, _ = cbet_model(obs, goal, None)
                # action was T, Chunk, Action_dim for single env
                # now it's N, T, C, A for vector env
                if use_diffusion:
                    for t in range(action.shape[1]):
                        exec_action = action[:, t].cpu().detach().numpy()
                        this_obs, reward, done, info = env.step(exec_action)
                        obs_stack.append(embed(encoder, this_obs))
                        if videorecorder.enabled:
                            videorecorder.record(info[0]["image"])
                        total_reward += reward.sum()
                        goal = goal_fn(goal_idx)
                        if done.all():
                            break
                else:
                    if cfg.action_window_size > 1:
                        action_list.append(action[:, -1].cpu().detach().numpy())
                        if len(action_list) > cfg.action_window_size:
                            action_list = action_list[1:]
                        curr_action = np.array(action_list)  # W, N, C, A
                        curr_action = curr_action.mean(axis=0)[:, 0]  # N, A
                        new_action_list = []
                        for a_chunk in action_list:
                            new_action_list.append(
                                np.concatenate(
                                    (a_chunk[:, 1:], np.zeros((a_chunk.shape[0], 1, a_chunk.shape[-1]))), axis=1
                                )
                            )
                            # ! better but not tested yet:
                            # new_action_list.append(np.roll(a_chunk, -1, axis=1))
                        action_list = new_action_list
                    else:
                        curr_action = action[:, -1, 0, :].cpu().detach().numpy()

                        this_obs, reward, done, info = env.step(curr_action)
                        this_obs_enc = embed(encoder, this_obs)
                        obs_stack.append(this_obs_enc)

                if videorecorder.enabled:
                    videorecorder.record(info[0]["image"])
                total_reward += reward.sum()
                goal = goal_fn(goal_idx)
            avg_reward += total_reward
            if cfg.env.gym.id == "pusht":
                base_seed = cfg.seed + goal_idx * cfg.num_envs
                env.seed([base_seed + j for j in range(cfg.num_envs)])
                avg_max_coverage += [info[i]["max_coverage"] for i in range(len(info))]
                avg_final_coverage += [info[i]["final_coverage"] for i in range(len(info))]
            elif cfg.env.gym.id in ["blockpush", "cube"]:
                avg_max_coverage += [info[i]["moved"] for i in range(len(info))]
                avg_final_coverage += [info[i]["entered"] for i in range(len(info))]
            completion_id_list += [info[i]["all_completions_ids"] for i in range(len(info))]
            videorecorder.save("eval_{}_{}.mp4".format(epoch, goal_idx))
        return (
            avg_reward / num_evals,
            completion_id_list,
            avg_max_coverage,
            avg_final_coverage,
        )

    metrics_history = []
    reward_history = []

    if accelerator.is_main_process:
        print("cbet_model type", type(cbet_model))
        if hasattr(cbet_model, "module"):
            cbet_model_raw = accelerator.unwrap_model(cbet_model)
            torch.save(cbet_model_raw, "{}/model_{}.pt".format(save_path, "init"))
        else:
            torch.save(cbet_model, "{}/model_{}.pt".format(save_path, "init"))

    for epoch in tqdm.trange(cfg.epochs):
        accelerator.wait_for_everyone()
        cbet_model.eval()
        if epoch % cfg.eval_on_env_freq == 0: 
            avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
                cfg,
                videorecorder=video,
                epoch=epoch,
            )
            reward_history.append(avg_reward)
            with open("{}/completion_idx_{}.json".format(save_path, epoch), "wb") as fp:
                pickle.dump(completion_id_list, fp)
            if accelerator.is_main_process:
                wandb.log({"eval_on_env": avg_reward})
            if cfg.env.gym.id in ["pusht", "blockpush", "cube"]:
                metric_final = (
                    "final coverage" if cfg.env.gym.id == "pusht" else "entered"
                )
                metric_max = "max coverage" if cfg.env.gym.id == "pusht" else "moved"
                metrics = {
                    f"{metric_final} mean": sum(final_coverage) / len(final_coverage),
                    f"{metric_final} max": max(final_coverage),
                    f"{metric_final} min": min(final_coverage),
                    f"{metric_max} mean": sum(max_coverage) / len(max_coverage),
                    f"{metric_max} max": max(max_coverage),
                    f"{metric_max} min": min(max_coverage),
                }
                if accelerator.is_main_process:
                    wandb.log(metrics)
                metrics_history.append(metrics)
            
            # Synchronize all processes after eval_on_env with extended timeout
            logger.info(f"Process {accelerator.local_process_index} finished eval_on_env, waiting for others...")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            else:
                accelerator.wait_for_everyone()
            logger.info(f"Process {accelerator.local_process_index} synchronized after eval_on_env")

        if epoch % cfg.eval_freq == 0:
            total_loss = 0
            action_diff = 0
            action_diff_tot = 0
            action_diff_mean_res1 = 0
            action_diff_mean_res2 = 0
            action_diff_max = 0
            with torch.no_grad():
                for data in test_loader:
                    obs, act, goal = (x.to(cfg.device) for x in data)
                    if not precompute_embeddings:
                        obs = encoder(obs)  # N T V P E
                        if use_libero_goal:
                            goal = encoder(goal)  # N T V P E
                    assert obs.ndim == 5, "expect N T V P E for obs"
                    assert goal.ndim == 5, "expect N T V P E for goals"
                    obs = einops.rearrange(obs, "N T V P E -> N T (V P) E") # keep the patch dim
                    goal = einops.rearrange(goal, "N T V P E -> N T (V P) E")
                    predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
                    # TODO: gather_for_metrics
                    total_loss += loss.item()
                    if accelerator.is_main_process:
                        wandb.log({"eval/{}".format(x): y for (x, y) in loss_dict.items()})
                    if not use_diffusion:
                        action_diff += loss_dict["action_diff"]
                        action_diff_tot += loss_dict["action_diff_tot"]
                        action_diff_mean_res1 += loss_dict["action_diff_mean_res1"]
                        action_diff_mean_res2 += loss_dict["action_diff_mean_res2"]
                        action_diff_max += loss_dict["action_diff_max"]
            print(f"Test loss: {total_loss / len(test_loader)}") 
            if accelerator.is_main_process and not use_diffusion:
                wandb.log({"eval/epoch_wise_action_diff": action_diff})
                wandb.log({"eval/epoch_wise_action_diff_tot": action_diff_tot})
                wandb.log({"eval/epoch_wise_action_diff_mean_res1": action_diff_mean_res1})
                wandb.log({"eval/epoch_wise_action_diff_mean_res2": action_diff_mean_res2})
                wandb.log({"eval/epoch_wise_action_diff_max": action_diff_max})

        accelerator.wait_for_everyone()
        cbet_model.train()
        train_loss = 0
        for data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            obs, act, goal = (x.to(cfg.device) for x in data)
            if not precompute_embeddings:
                obs = encoder(obs)  # N T V P E
                if use_libero_goal:
                    goal = encoder(goal)  # N T V P E
            obs = einops.rearrange(obs, "N T V P E -> N T (V P) E")
            goal = einops.rearrange(goal, "N T V P E -> N T (V P) E")
            predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
            train_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()

            if use_diffusion:
                if hasattr(cbet_model, "module"):
                    cbet_model_raw = accelerator.unwrap_model(cbet_model)
                    cbet_model_raw.ema_step()
                else:
                    cbet_model.ema_step()

            if accelerator.is_main_process:
                wandb.log({"train/{}".format(x): y for (x, y) in loss_dict.items()})

        print(f"Train loss: {train_loss / len(train_loader)}")

        # save model
        if epoch % cfg.save_every == 0 and accelerator.is_main_process:
            print("cbet_model type", type(cbet_model))
            if hasattr(cbet_model, "module"):
                cbet_model_raw = accelerator.unwrap_model(cbet_model)
                torch.save(cbet_model_raw, "{}/model_{}.pt".format(save_path, epoch))
            else:
                torch.save(cbet_model, "{}/model_{}.pt".format(save_path, epoch))

    avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
        cfg,
        num_evals=cfg.num_final_evals,
        videorecorder=video,
        epoch=cfg.epochs,
    )
    
    # Synchronize all processes after final eval_on_env
    logger.info(f"Process {accelerator.local_process_index} finished final eval_on_env, waiting for others...")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    else:
        accelerator.wait_for_everyone()
    logger.info(f"Process {accelerator.local_process_index} synchronized after final eval_on_env")
    
    reward_history.append(avg_reward)
    if cfg.env.gym.id in ["pusht", "blockpush", "cube"]:
        metric_final = "final coverage" if cfg.env.gym.id == "pusht" else "entered"
        metric_max = "max coverage" if cfg.env.gym.id == "pusht" else "moved"
        metrics = {
            f"{metric_final} mean": sum(final_coverage) / len(final_coverage),
            f"{metric_final} max": max(final_coverage),
            f"{metric_final} min": min(final_coverage),
            f"{metric_max} mean": sum(max_coverage) / len(max_coverage),
            f"{metric_max} max": max(max_coverage),
            f"{metric_max} min": min(max_coverage),
        }
        if accelerator.is_main_process:
            wandb.log(metrics)
        metrics_history.append(metrics)

    with open("{}/completion_idx_final.json".format(save_path), "wb") as fp:
        pickle.dump(completion_id_list, fp)
    if cfg.env.gym.id == "pusht":
        final_eval_on_env = max([x["final coverage mean"] for x in metrics_history])
    elif cfg.env.gym.id == "blockpush" or cfg.env.gym.id == "cube":
        final_eval_on_env = max([x["entered mean"] for x in metrics_history])
    elif cfg.env.gym.id == "libero_goal":
        final_eval_on_env = max(reward_history)
    elif cfg.env.gym.id == "kitchen-v0":
        final_eval_on_env = avg_reward
    if accelerator.is_main_process:
        wandb.log({"final_eval_on_env": final_eval_on_env})
    return final_eval_on_env


if __name__ == "__main__":
    main()
