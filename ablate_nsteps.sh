#!/usr/bin/env bash
# Pi0.5 去噪步数消融，固定闭环重规划频率并输出每组成功率。

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
POLICY_PATH="${POLICY_PATH:?请设置 POLICY_PATH 为 Pi0.5 checkpoint 路径或模型 ID}"
EVAL_BIN="${EVAL_BIN:-${ROOT_DIR}/pi0_venv/bin/lerobot-eval}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/pi0_venv/bin/python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/eval}"
N_EPISODES="${N_EPISODES:-5}"
BATCH_SIZE="${BATCH_SIZE:-5}"
N_ACTION_STEPS="${N_ACTION_STEPS:-5}"
STEPS="${STEPS:-1 2 5 10 50}"

mkdir -p "${OUTPUT_ROOT}"

for num_steps in ${STEPS}; do
  run_dir="${OUTPUT_ROOT}/pi05_nstep${num_steps}_spatial"
  log_file="${run_dir}.log"
  echo "========== num_inference_steps=${num_steps} =========="
  start_time=$(date +%s)
  MUJOCO_GL="${MUJOCO_GL:-egl}" "${EVAL_BIN}" \
    --policy.path="${POLICY_PATH}" \
    --policy.num_inference_steps="${num_steps}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.n_episodes="${N_EPISODES}" \
    --eval.batch_size="${BATCH_SIZE}" \
    --output_dir="${run_dir}" \
    >"${log_file}" 2>&1
  end_time=$(date +%s)
  success_rate=$("${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["overall"]["pc_success"])' \
    "${run_dir}/eval_info.json")
  echo "num_inference_steps=${num_steps} -> 成功率 ${success_rate}%  耗时 $((end_time - start_time))s"
done
