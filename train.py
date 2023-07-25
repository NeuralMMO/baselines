import os
import logging
import environment
import policy
import config
from curriculum.task_encoder import TaskEncoder
#from elm_for_nmmo.elm_curriculum_gen import OpenELMTaskGenerator, SimpleTaskGenerator
from pufferlib.vectorization.multiprocessing import VecEnv as MPVecEnv
from pufferlib.vectorization.serial import VecEnv as SerialVecEnv
from pufferlib.policy_store import DirectoryPolicyStore
import clean_pufferl

import manual_curriculum

LLM_CHECKPOINT = "Salesforce/codegen-350M-mono"
AGENT_MODEL_PATH = ""
NUM_TRAIN_TASKS = 30
NUM_TEST_TASKS = 5
NUM_NEW_TASKS = 5
EPOCHS_PER_ELM_UPDATE = 10
DEBUG = True
CURRICULUM_FILE_PATH = "submission/custom_curriculum_with_embedding.pkl"

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
    learner_policy = policy.NmmoPolicy.create_policy(binding, args.__dict__)
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
    task_generator = OpenELMTaskGenerator(manual_curriculum.task_spec, LLM_CHECKPOINT) if use_elm else SimpleTaskGenerator(manual_curriculum.task_spec)
    train_task_spec = task_generator.generate_tasks(NUM_TRAIN_TASKS)
    task_encoder = TaskEncoder(LLM_CHECKPOINT, manual_curriculum, batch_size=2)
    task_encoder.get_task_embedding(train_task_spec, save_to_file=CURRICULUM_FILE_PATH)
    # @daveey: We need a baseline checkpoint for this
    #load_agent_model(AGENT_MODEL_PATH)
    for epoch in range(30):
        if use_elm and epoch % EPOCHS_PER_ELM_UPDATE == 9:
            print("eval fn evol!, epoch:", epoch)
            new_task_spec = task_generator.evolve_tasks(train_task_spec, NUM_NEW_TASKS, debug=DEBUG)
            train_task_spec += new_task_spec
            task_encoder.get_task_embedding(train_task_spec, save_to_file=CURRICULUM_FILE_PATH)
        trainer.train()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create a local config for testing that won't OOM your machine
    # You can either edit the defaults in config.py or set args
    # from the commandline.
    args = config.create_config(config.LocalConfig)
    trainer = setup_env(args)

    # Uncomment the following line to run reinforcement learning track
    reinforcement_learning_track(trainer, args)

    # Uncomment the following line to run curriculum generation track
    #curriculum_generation_track(trainer, args, use_elm=True)

    trainer.close()
