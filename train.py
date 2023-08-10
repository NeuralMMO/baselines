import os
import logging

from pufferlib.vectorization.multiprocessing import VecEnv as MPVecEnv
from pufferlib.vectorization.serial import VecEnv as SerialVecEnv
from pufferlib.policy_store import DirectoryPolicyStore
import clean_pufferl

import environment

from reinforcement_learning import policy
from reinforcement_learning import config

from curriculum_generation import manual_curriculum
from curriculum_generation.task_encoder import TaskEncoder

LLM_CHECKPOINT = "Salesforce/codegen25-7b-instruct"
AGENT_MODEL_PATH = ""
NUM_SEED_TASKS = 20
NUM_NEW_TASKS = 5
DEBUG = True
CURRICULUM = manual_curriculum

# NOTE: this file changes when running curriculum generation track
# Run test_task_encoder.py to regenerate this file (or get it from the repo)
CURRICULUM_FILE_PATH = "curriculum_generation/curriculum_with_embedding.pkl"

def setup_env(args):
    run_dir = os.path.join(args.runs_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logging.info("Training run: %s (%s)", args.run_name, run_dir)
    logging.info("Training args: %s", args)
    binding = environment.create_binding(args)
    policy_store = None
    if args.policy_store_dir is not None:
        logging.info("Using policy store from %s", args.policy_store_dir)
        policy_store = DirectoryPolicyStore(args.policy_store_dir)
    learner_policy = policy.Baseline.create_policy(binding, args.__dict__)
    trainer = clean_pufferl.CleanPuffeRL(
        binding=binding,
        agent=learner_policy,
        data_dir=run_dir,
        exp_name=args.run_name,
        policy_store=policy_store,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_extra_data=args,
        checkpoint_interval=args.checkpoint_interval,
        vec_backend=SerialVecEnv if args.use_serial_vecenv else MPVecEnv,
        total_timesteps=args.train_num_steps,
        num_envs=args.num_envs,
        num_cores=args.num_cores or args.num_envs,
        num_buffers=args.num_buffers,
        batch_size=args.rollout_batch_size,
        learning_rate=args.ppo_learning_rate,
        selfplay_learner_weight=args.learner_weight,
        selfplay_num_policies=args.max_opponent_policies + 1,
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
    if use_elm:
        from curriculum_generation.elm import OpenELMTaskGenerator
        task_encoder = TaskEncoder(LLM_CHECKPOINT, CURRICULUM, batch_size=2)
        task_generator = OpenELMTaskGenerator(CURRICULUM.task_spec, LLM_CHECKPOINT)

        # @daveey: We need a baseline checkpoint for this
        #load_agent_model(AGENT_MODEL_PATH)

        # Generating new tasks and evaluating all candidate training tasks
        for _ in range(5):
            # NOTE: adjust NUM_SEED_TASKS to fit your gpu
            seed_task_spec = task_generator.sample_tasks(NUM_SEED_TASKS, random_ratio=1)
            new_task_spec = task_generator.evolve_tasks(seed_task_spec, NUM_NEW_TASKS, debug=DEBUG)
            task_generator.add_tasks(new_task_spec)
            task_encoder.get_task_embedding(seed_task_spec + new_task_spec, save_to_file=CURRICULUM_FILE_PATH)
            # CHECK ME: the trainer will automatically use the new task embedding file
            _, _, infos = trainer.evaluate()
            task_generator.update(infos) # update the task stats

        # NOTE: sample_tasks() uses task stats to sample learnable tasks
        train_task_spec = task_generator.sample_tasks(NUM_SEED_TASKS*3, random_ratio=0.3) # NOTE: arbitrary numbers

    else:
        # using the manually curated curriculum
        from curriculum_generation.task_sampler import LearnableTaskSampler
        from curriculum_generation import custom_curriculum
        task_encoder = TaskEncoder(LLM_CHECKPOINT, custom_curriculum, batch_size=2)
        train_task_spec = custom_curriculum.task_spec

    # Use the train_task_spec to train agents
    task_encoder.get_task_embedding(train_task_spec, save_to_file=CURRICULUM_FILE_PATH)
    task_encoder.close()
    reinforcement_learning_track(trainer, args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create a local config for testing that won't OOM your machine
    # You can either edit the defaults in config.py or set args
    # from the commandline.
    args = config.create_config(config.LocalConfig)
    args.tasks_path = CURRICULUM_FILE_PATH # NOTE: this file must exist
    trainer = setup_env(args)

    # Uncomment the following line to run reinforcement learning track
    #reinforcement_learning_track(trainer, args)

    # Uncomment the following line to run curriculum generation track
    curriculum_generation_track(trainer, args, use_elm=True)

    trainer.close()
