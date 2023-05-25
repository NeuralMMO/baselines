import argparse
import wandb
from lib.policy_pool.json_policy_pool import JsonPolicyPool

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--path", dest="policy_pool", type=str, required=True,
    help="path to policy pool to use for evaluation (default: None)")
  parser.add_argument(
    "--wandb.project", dest="wandb_project", type=str, default=None,
      help="wandb project name (default: None)")
  parser.add_argument(
    "--wandb.entity", dest="wandb_entity", type=str, default=None,
      help="wandb entity name (default: None)")

  args = parser.parse_args()
  pool = JsonPolicyPool(args.policy_pool)
  pool._load()

  print(pool.to_table().to_string(max_rows=None, index=False))

  if args.wandb_project:
    run_name = f"policy_pool.{args.policy_pool.split('/')[-1].split('.')[0]}"
    table = pool.to_table()

    wandb.init(
      project=args.wandb_project,
      entity=args.wandb_entity,
      name=f"policy_pool.{args.policy_pool.split('/')[-1].split('.')[0]}",
    )
    wandb.log({"policy_pool": wandb.Table(dataframe=table)})

    for index, row in table.iterrows():
      # Log metrics over time
      wandb.log({row["metric_name"]: row["metric_value"]})

