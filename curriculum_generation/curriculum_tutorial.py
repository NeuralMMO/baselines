"""This file explains how you can manually create your curriculum.
1. Use pre-built evaluation functions and TaskSpec to define training tasks.
2. Define your own evaluation functions.
3. Check if your training task tasks are valid and pickable. Must satisfy both.
4. Generate the task embedding file using the task encoder.
5. Train agents using the task embedding file.
6. Extract the training task stats.
"""

# allow custom functions to use pre-built eval functions without prefix
from nmmo.task.base_predicates import CountEvent, InventorySpaceGE, TickGE, norm
from nmmo.task.task_spec import TaskSpec, check_task_spec


##############################################################################
# Use pre-built eval functions and TaskSpec class to define each training task
# See manual_curriculum.py for detailed examples based on pre-built eval fns

curriculum = []

# Make training tasks for each of the following events
# Agents have completed the task if they have done the event N times
essential_events = [
    "GO_FARTHEST",
    "EAT_FOOD",
    "DRINK_WATER",
    "SCORE_HIT",
    "HARVEST_ITEM",
    "LEVEL_UP",
]

for event_code in essential_events:
    curriculum.append(
        TaskSpec(
            eval_fn=CountEvent,  # is a pre-built eval function
            eval_fn_kwargs={"event": event_code, "N": 10},  # kwargs for CountEvent
        )
    )


##############################################################################
# Create training tasks using custom evaluation functions

def PracticeEating(gs, subject):
    """The progress, the max of which is 1, should
    * increase small for each eating
    * increase big for the 1st and 3rd eating
    * reach 1 with 10 eatings
    """
    num_eat = len(subject.event.EAT_FOOD)
    progress = num_eat * 0.06
    if num_eat >= 1:
        progress += 0.1
    if num_eat >= 3:
        progress += 0.3
    return norm(progress)  # norm is a helper function to normalize the value to [0, 1]

curriculum.append(TaskSpec(eval_fn=PracticeEating, eval_fn_kwargs={}))

# You can also use pre-built eval functions to define your own eval functions
def PracticeInventoryManagement(gs, subject, space, num_tick):
    return norm(InventorySpaceGE(gs, subject, space) * TickGE(gs, subject, num_tick))

for space in [2, 4, 8]:
    curriculum.append(
        TaskSpec(
            eval_fn=PracticeInventoryManagement,
            eval_fn_kwargs={"space": space, "num_tick": 500},
        )
    )


if __name__ == "__main__":
    # Import the custom curriculum
    print("------------------------------------------------------------")
    import curriculum_tutorial  # which is this file
    CURRICULUM = curriculum_tutorial.curriculum
    print("The number of training tasks in the curriculum:", len(CURRICULUM))

    # Check if these task specs are valid in the nmmo environment
    # Invalid tasks will crash your agent training
    print("------------------------------------------------------------")
    print("Checking whether the task specs are valid ...")
    results = check_task_spec(CURRICULUM)
    num_error = 0
    for result in results:
        if result["runnable"] is False:
            print("ERROR: ", result["spec_name"])
            num_error += 1
    assert num_error == 0, "Invalid task specs will crash training. Please fix them."
    print("All training tasks are valid.")

    # The task_spec must be picklable to be used for agent training
    print("------------------------------------------------------------")
    print("Checking if the training tasks are picklable ...")
    CURRICULUM_FILE_PATH = "custom_curriculum_with_embedding.pkl"
    with open(CURRICULUM_FILE_PATH, "wb") as f:
        import dill
        dill.dump(CURRICULUM, f)
    print("All training tasks are picklable.")

    # To use the curriculum for agent training, the curriculum, task_spec, should be
    # saved to a file with the embeddings using the task encoder. The task encoder uses
    # a coding LLM to encode the task_spec into a vector.
    print("------------------------------------------------------------")
    print("Generating the task spec with embedding file ...")
    from task_encoder import TaskEncoder
    LLM_CHECKPOINT = "Salesforce/codegen25-7b-instruct"

    # Get the task embeddings for the training tasks and save to file
    # You need to provide the curriculum file as a module to the task encoder
    with TaskEncoder(LLM_CHECKPOINT, curriculum_tutorial) as task_encoder:
        task_encoder.get_task_embedding(CURRICULUM, save_to_file=CURRICULUM_FILE_PATH)
    print("Done.")

    # Initialize the trainer with the custom curriculum
    # These lines are the same as the RL track. If these don't run, please see train.py
    from reinforcement_learning import config
    from train import setup_env
    args = config.create_config(config.Config)
    args.tasks_path = CURRICULUM_FILE_PATH  # This is the curriculum file saved by the task encoder

    # Remove below lines if you want to use the default training config
    local_mode = True
    if local_mode:
        args.num_envs = 1
        args.num_buffers = 1
        args.use_serial_vecenv = True
        args.rollout_batch_size = 2**12

    print("------------------------------------------------------------")
    print("Setting up the agent training env ...")
    trainer = setup_env(args)

    # Train agents using the curriculum file
    # NOTE: this is basically the same as the reinforcement_learning_track function in the train.py
    while not trainer.done_training():
        print("------------------------------------------------------------")
        print("Evaluating the agents ...")
        _, _, infos = trainer.evaluate()
        # The training task stats are available in infos, which then can be use for training task selection
        if len(infos) > 0:
            print("------------------------------------------------------------")
            print("Training task stats:")
            curri_keys = [key for key in infos.keys() if key.startswith("curriculum/")]
            for key in curri_keys:
                completed = []
                max_progress = []
                reward_signal_count = []
                for sub_list in infos[key]:
                    for prog, rcnt in sub_list:
                        completed.append(int(prog>=1)) # progress >= 1 is considered task complete
                        max_progress.append(prog)
                        reward_signal_count.append(rcnt)
                print(f"{key} -- task tried: {len(completed)}, completed: {sum(completed)}, " +
                      f"avg max progress: {sum(max_progress)/len(max_progress):.3f}, " +
                      f"avg reward signal count: {sum(reward_signal_count)/len(reward_signal_count):.3f}")

            print("------------------------------------------------------------")
            print("The tutorial is done.")
            break

        print("------------------------------------------------------------")
        print("Training the agents ...")
        trainer.train(
            update_epochs=args.ppo_update_epochs,
            bptt_horizon=args.bptt_horizon,
            batch_rows=args.ppo_training_batch_size // args.bptt_horizon,
        )
