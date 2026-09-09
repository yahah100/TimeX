#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

datasets="all"
methods="ours,ig,dyna,winit,cortx,sgt+grad"
stage="all"
winit_epochs=1000
seed=0
results_dir="$repo_root/results/table1"

usage() {
    code="${1:-0}"
    cat <<'EOF'
Usage: ./run_table1.sh [options]

Reproduce the functional rows of Table 1 over all five published folds.

Options:
  --datasets LIST       all, freqshape, or seqcomb_uv (comma-separated)
  --methods LIST        ours,ig,dyna,winit,cortx,sgt+grad (comma-separated)
  --stage STAGE         all, train, or evaluate
  --winit-epochs N      WinIT generator epochs (default: 1000)
  --seed N              Base random seed (default: 0)
  --results-dir PATH    Logs, JSON, and summaries (default: results/table1)
  -h, --help            Show this help

Examples:
  ./run_table1.sh --datasets freqshape --methods ours,ig
  ./run_table1.sh --stage evaluate --methods ours,ig,dyna

Set TIMEX_DATA_ROOT to override the default dataset/ directory.
CoRTX and SGT+Grad run through experiments/other_baselines/train_synth_baselines.py
rather than their committed one-fold research snapshots.
EOF
    exit "$code"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) datasets="$2"; shift 2 ;;
        --methods) methods="$2"; shift 2 ;;
        --stage) stage="$2"; shift 2 ;;
        --winit-epochs) winit_epochs="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
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

for method in ${methods//,/ }; do
    case "$method" in
        ours|ig|dyna|winit|cortx|sgt+grad) ;;
        *) echo "Unknown method: $method" >&2; usage 2 ;;
    esac
done

mkdir -p "$results_dir"

all_exist() {
    template="$1"
    for split in 1 2 3 4 5; do [[ -e "${template//__SPLIT__/$split}" ]] || return 1; done
}

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
    model_dir="$experiment_dir/models"
    mkdir -p "$model_dir"

    if all_exist "$model_dir/Scomb_transformer_split=__SPLIT__.pt"; then
        echo ">>> $dataset predictor (checkpoints already complete)"
    else
        run_logged "$results_dir/${dataset}_predictor_train.log" \
            uv run python "$experiment_dir/train_transformer.py" --seed "$seed"
    fi

    if contains "$methods" ours && ! all_exist "$model_dir/bc_full_split=__SPLIT__.pt"; then
        run_logged "$results_dir/${dataset}_timex_train.log" \
            uv run python "$experiment_dir/bc_model_ptype.py" --seed "$seed"
    fi

    if contains "$methods" cortx && ! all_exist "$model_dir/cortx_split=__SPLIT__.pt"; then
        run_logged "$results_dir/${dataset}_cortx_train.log" \
            uv run python experiments/other_baselines/train_synth_baselines.py \
                --dataset "$dataset" --method cortx --seed "$seed"
    fi

    if contains "$methods" sgt+grad && ! all_exist "$model_dir/sgt_split=__SPLIT__.pt"; then
        run_logged "$results_dir/${dataset}_sgt_train.log" \
            uv run python experiments/other_baselines/train_synth_baselines.py \
                --dataset "$dataset" --method sgt --seed "$seed"
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
                --epochs "$winit_epochs" \
                --seed "$seed"
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
        case "$method" in
            ours) model_path="$model_dir/bc_full_split=1.pt" ;;
            cortx) model_path="$model_dir/cortx_split=1.pt" ;;
            sgt+grad) model_path="$model_dir/sgt_split=1.pt" ;;
            *) model_path="$model_dir/Scomb_transformer_split=1.pt" ;;
        esac

        if ! all_exist "${model_path/split=1/split=__SPLIT__}"; then
            echo "Missing checkpoints for $dataset $method: $model_path" >&2
            echo "Run this script with --stage train first." >&2
            exit 1
        fi

        eval_extra=()
        if [[ -n "${TIMEX_MAX_SAMPLES:-}" ]]; then
            eval_extra+=(--max-samples "$TIMEX_MAX_SAMPLES")
        fi

        run_logged "$results_dir/${dataset}_${method//+/_}_evaluation.log" \
            uv run python experiments/evaluation/saliency_exp_synth.py \
                --dataset "$eval_name" \
                --exp_method "$method" \
                --split_no -1 \
                --model_path "$model_path" \
                --seed "$seed" \
                --results-json "$results_dir/${dataset}_${method//+/_}_results.json" \
                --no-progress \
                "${eval_extra[@]}"
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

if [[ "$stage" == "all" || "$stage" == "evaluate" ]]; then
    uv run python experiments/evaluation/summarize_synth.py --table 1 "$results_dir"
fi

echo
echo "Table 1 runs completed. Outputs: $results_dir"
