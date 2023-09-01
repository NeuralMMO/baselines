import os
import logging
import torch

from pufferlib.vectorization import Serial, Multiprocessing
from pufferlib.policy_store import DirectoryPolicyStore
from pufferlib.frameworks import cleanrl
import clean_pufferl

import environment

from reinforcement_learning import policy
from reinforcement_learning import config

# NOTE: this file changes when running curriculum generation track
# Run test_task_encoder.py to regenerate this file (or get it from the repo)
BASELINE_CURRICULUM_FILE = "reinforcement_learning/curriculum_with_embedding.pkl"
CUSTOM_CURRICULUM_FILE = "curriculum_generation/custom_curriculum_with_embedding.pkl"

def setup_env(args):
    run_dir = os.path.join(args.runs_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logging.info("Training run: %s (%s)", args.run_name, run_dir)
    logging.info("Training args: %s", args)

    policy_store = None
    if args.policy_store_dir is None:
        args.policy_store_dir = os.path.join(run_dir, "policy_store")
        logging.info("Using policy store from %s", args.policy_store_dir)
        policy_store = DirectoryPolicyStore(args.policy_store_dir)

    def make_policy(envs):
        learner_policy = policy.Baseline(
            envs,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            task_size=args.task_size
        )
        return cleanrl.Policy(learner_policy)

    trainer = clean_pufferl.CleanPuffeRL(
        device=torch.device(args.device),
        seed=args.seed,
        env_creator=environment.make_env_creator(args),
        env_creator_kwargs={},
        agent_creator=make_policy,
        data_dir=run_dir,
        exp_name=args.run_name,
        policy_store=policy_store,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_extra_data=args,
        checkpoint_interval=args.checkpoint_interval,
        vectorization=Serial if args.use_serial_vecenv else Multiprocessing,
        total_timesteps=args.train_num_steps,
        num_envs=args.num_envs,
        num_cores=args.num_cores or args.num_envs,
        num_buffers=args.num_buffers,
        batch_size=args.rollout_batch_size,
        learning_rate=args.ppo_learning_rate,
        selfplay_learner_weight=args.learner_weight,
        selfplay_num_policies=args.max_opponent_policies + 1,
        #record_loss = args.record_loss,
    )
    return trainer

def reinforcement_learning_track(trainer, args):
    while not trainer.done_training():
        trainer.evaluate()
        trainer.train(
            update_epochs=args.ppo_update_epochs,
            bptt_horizon=args.bptt_horizon,
            batch_rows=args.ppo_training_batch_size // args.bptt_horizon,
        )

def curriculum_generation_track(trainer, args, use_elm=True):
    from curriculum_generation.task_encoder import TaskEncoder
    LLM_CHECKPOINT = "Salesforce/codegen25-7b-instruct"

    if use_elm:
        from curriculum_generation import manual_curriculum
        from curriculum_generation.elm import OpenELMTaskGenerator
        AGENT_MODEL_PATH = ""
        NUM_SEED_TASKS = 20
        NUM_NEW_TASKS = 5
        ELM_DEBUG = True

        task_encoder = TaskEncoder(LLM_CHECKPOINT, manual_curriculum, batch_size=2)
        task_generator = OpenELMTaskGenerator(manual_curriculum.curriculum, LLM_CHECKPOINT)

        # @daveey: We need a baseline checkpoint for this
        #load_agent_model(AGENT_MODEL_PATH)

        # Generating new tasks and evaluating all candidate training tasks
        for _ in range(3):
            # NOTE: adjust NUM_SEED_TASKS to fit your gpu
            seed_task_list = task_generator.sample_tasks(NUM_SEED_TASKS, random_ratio=1)
            new_task_list = task_generator.evolve_tasks(seed_task_list, NUM_NEW_TASKS, debug=ELM_DEBUG)
            task_generator.add_tasks(new_task_list)
            task_encoder.get_task_embedding(seed_task_list + new_task_list, save_to_file=CUSTOM_CURRICULUM_FILE)
            # CHECK ME: the trainer will automatically use the new task embedding file
            _, _, infos = trainer.evaluate()
            task_generator.update(infos) # update the task stats

        # NOTE: sample_tasks() uses task stats to sample learnable tasks
        curriculum = task_generator.sample_tasks(NUM_SEED_TASKS*3, random_ratio=0.3) # NOTE: arbitrary numbers

    else:
        from curriculum_generation import curriculum_tutorial  # custom tutorial
        task_encoder = TaskEncoder(LLM_CHECKPOINT, curriculum_tutorial, batch_size=2)
        curriculum = curriculum_tutorial.curriculum

    # Use the train_task_spec to train agents
    task_encoder.get_task_embedding(curriculum, save_to_file=CUSTOM_CURRICULUM_FILE)
    task_encoder.close()
    reinforcement_learning_track(trainer, args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # You can either edit the defaults in config.py or set args
    # from the commandline.
    args = config.create_config(config.Config)

    # Avoid OOMing your machine for local testing
    if args.local_mode:
        args.num_envs = 1
        args.num_buffers = 1
        args.use_serial_vecenv = True
        args.rollout_batch_size = 2**10

    if args.track == "rl":
      args.tasks_path = BASELINE_CURRICULUM_FILE
      trainer = setup_env(args)
      reinforcement_learning_track(trainer, args)
    elif args.track == "curriculum":
      args.tasks_path = CUSTOM_CURRICULUM_FILE
      trainer = setup_env(args)
      curriculum_generation_track(trainer, args, use_elm=True)
    else:
      raise ValueError(f"Unknown track {args.track}, must be 'rl' or 'curriculum'")

    trainer.close()
