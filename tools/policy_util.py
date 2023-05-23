import argparse
from lib.policy_pool.json_policy_pool import JsonPolicyPool

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--path", dest="policy_pool", type=str, required=True,
    help="path to policy pool to use for evaluation (default: None)")
  args = parser.parse_args()
  pool = JsonPolicyPool(args.policy_pool)
  pool._load()

  print(pool.to_table().to_string(max_rows=None))
