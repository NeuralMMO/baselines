python -O -m main \
--rollout.num_envs=4 \
--rollout.num_buffers=4  \
--rollout.num_steps=32 \
--wandb.entity=daveey \
--wandb.project=nmmo \
--checkpoint_dir=/fsx/home-daveey/checkpoints/baseline \
--resume_from=latest \
--train.num_steps=100000000
