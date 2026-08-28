#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

datasets="all"
methods="ours,ig,dyna,winit"
stage="all"
winit_epochs=1000
results_dir="$repo_root/results/table1"

usage() {
    code="${1:-0}"
    cat <<'EOF'
Usage: ./run_table1.sh [options]

Reproduce the functional rows of Table 1 over all five published folds.

Options:
  --datasets LIST       all, freqshape, or seqcomb_uv (comma-separated)
  --methods LIST        ours,ig,dyna,winit (comma-separated)
  --stage STAGE         all, train, or evaluate
  --winit-epochs N      WinIT generator epochs (default: 1000)
  --results-dir PATH    Log directory (default: results/table1)
  -h, --help            Show this help

Examples:
  ./run_table1.sh --datasets freqshape --methods ours,ig
  ./run_table1.sh --stage evaluate --methods ours,ig,dyna

Set TIMEX_DATA_ROOT to override the default dataset/ directory.
CoRTX and SGT+Grad are not accepted because their committed scripts are
incomplete one-fold research snapshots and cannot reproduce Table 1 as-is.
EOF
    exit "$code"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) datasets="$2"; shift 2 ;;
        --methods) methods="$2"; shift 2 ;;
        --stage) stage="$2"; shift 2 ;;
        --winit-epochs) winit_epochs="$2"; shift 2 ;;
        --results-dir) results_dir="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 2 ;;
    esac
done

case "$stage" in
    all|train|evaluate) ;;
    *) echo "Invalid stage: $stage" >&2; usage 2 ;;
esac

contains() {
    [[ ",$1," == *",$2,"* ]]
}

if [[ "$datasets" == "all" ]]; then
    datasets="freqshape,seqcomb_uv"
fi

for method in cortx sgt sgt+grad; do
    if contains "$methods" "$method"; then
        echo "Cannot run $method reproducibly: its upstream Table 1 pipeline is incomplete." >&2
        exit 2
    fi
done

for method in ${methods//,/ }; do
    case "$method" in
        ours|ig|dyna|winit) ;;
        *) echo "Unknown method: $method" >&2; usage 2 ;;
    esac
done

mkdir -p "$results_dir"

run_logged() {
    log_file="$1"
    shift
    echo
    echo ">>> $*"
    "$@" 2>&1 | tee "$log_file"
}

train_dataset() {
    dataset="$1"
    case "$dataset" in
        freqshape)
            experiment_dir="experiments/freqshape"
            ;;
        seqcomb_uv)
            experiment_dir="experiments/scs_better"
            ;;
        *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
    esac

    run_logged "$results_dir/${dataset}_predictor_train.log" \
        uv run python "$experiment_dir/train_transformer.py"

    if contains "$methods" ours; then
        run_logged "$results_dir/${dataset}_timex_train.log" \
            uv run python "$experiment_dir/bc_model_ptype.py"
    fi

    if contains "$methods" winit; then
        if [[ ! -f txai/baselines/WinIT/winit/explainer/winitexplainers.py ]]; then
            echo "WinIT submodule is missing. Run: git submodule update --init --recursive" >&2
            exit 1
        fi
        eval_name="$dataset"
        [[ "$dataset" == "seqcomb_uv" ]] && eval_name="scs_better"
        run_logged "$results_dir/${dataset}_winit_train.log" \
            uv run python experiments/evaluation/winit_wrapper.py \
                --dataset "$eval_name" \
                --models_path "$experiment_dir/models" \
                --epochs "$winit_epochs"
    fi
}

evaluate_dataset() {
    dataset="$1"
    case "$dataset" in
        freqshape)
            eval_name="freqshape"
            model_dir="experiments/freqshape/models"
            ;;
        seqcomb_uv)
            eval_name="scs_better"
            model_dir="experiments/scs_better/models"
            ;;
        *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
    esac

    for method in ${methods//,/ }; do
        if [[ "$method" == "ours" ]]; then
            model_path="$model_dir/bc_full_split=1.pt"
        else
            model_path="$model_dir/Scomb_transformer_split=1.pt"
        fi

        if [[ ! -f "$model_path" ]]; then
            echo "Missing checkpoint: $model_path" >&2
            echo "Run this script with --stage train first." >&2
            exit 1
        fi

        run_logged "$results_dir/${dataset}_${method}_evaluation.log" \
            uv run python experiments/evaluation/saliency_exp_synth.py \
                --dataset "$eval_name" \
                --exp_method "$method" \
                --split_no -1 \
                --model_path "$model_path"
    done
}

for dataset in ${datasets//,/ }; do
    case "$dataset" in
        freqshape|seqcomb_uv) ;;
        *) echo "Unknown dataset: $dataset" >&2; usage 2 ;;
    esac

    if [[ "$stage" == "all" || "$stage" == "train" ]]; then
        train_dataset "$dataset"
    fi
    if [[ "$stage" == "all" || "$stage" == "evaluate" ]]; then
        evaluate_dataset "$dataset"
    fi
done

echo
echo "Table 1 runs completed. Logs: $results_dir"
