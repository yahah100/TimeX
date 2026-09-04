#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

datasets="all"
methods="ours,ig,dyna,winit,cortx,sgt+grad"
stage="all"
winit_epochs=1000
seed=42
results_dir="$repo_root/results/table2"

usage() {
    code="${1:-0}"
    cat <<'EOF'
Usage: ./run_table2.sh [options]

Reproduce Table 2 over all five published folds.

Options:
  --datasets LIST       all, seqcomb_mv, or lowvar (comma-separated)
  --methods LIST        ours,ig,dyna,winit,cortx,sgt+grad
  --stage STAGE         all, train, or evaluate
  --winit-epochs N      WinIT generator epochs (default: 1000)
  --seed N              Base random seed (default: 42)
  --results-dir PATH    Logs, JSON, and summaries (default: results/table2)
  -h, --help            Show this help

Set TIMEX_DATA_ROOT to override the default dataset/ directory.
EOF
    exit "$code"
}

die() { echo "Error: $*" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets|--methods|--stage|--winit-epochs|--seed|--results-dir)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            case "$1" in
                --datasets) datasets="$2" ;; --methods) methods="$2" ;;
                --stage) stage="$2" ;; --winit-epochs) winit_epochs="$2" ;;
                --seed) seed="$2" ;; --results-dir) results_dir="$2" ;;
            esac
            shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 2 ;;
    esac
done
[[ "$stage" =~ ^(all|train|evaluate)$ ]] || die "invalid stage: $stage"
[[ "$seed" =~ ^[0-9]+$ ]] || die "seed must be a non-negative integer"
[[ "$winit_epochs" =~ ^[1-9][0-9]*$ ]] || die "WinIT epochs must be positive"
[[ "$datasets" == all ]] && datasets="seqcomb_mv,lowvar"

contains() { [[ ",$1," == *",$2,"* ]]; }
for dataset in ${datasets//,/ }; do
    [[ "$dataset" =~ ^(seqcomb_mv|lowvar)$ ]] || die "unknown dataset: $dataset"
done
for method in ${methods//,/ }; do
    [[ "$method" =~ ^(ours|ig|dyna|winit|cortx|sgt\+grad)$ ]] || die "unknown method: $method"
done
mkdir -p "$results_dir"

run_logged() {
    log_file="$1"; description="$2"; shift 2
    echo ">>> $description"
    if ! "$@" >"$log_file" 2>&1; then
        echo "FAILED: $description (log: $log_file)" >&2
        tail -n 40 "$log_file" >&2 || true
        return 1
    fi
}

all_exist() {
    template="$1"
    for split in 1 2 3 4 5; do [[ -e "${template//__SPLIT__/$split}" ]] || return 1; done
}

for dataset in ${datasets//,/ }; do
    if [[ "$dataset" == seqcomb_mv ]]; then
        experiment="experiments/seqcomb_mv"; eval_dataset="seqcomb_mv"
    else
        experiment="experiments/lowvardetect"; eval_dataset="lowvardetect"
    fi
    models="$experiment/models"
    mkdir -p "$models"

    if [[ "$stage" == all || "$stage" == train ]]; then
        if ! all_exist "$models/transformer_split=__SPLIT__.pt"; then
            run_logged "$results_dir/${dataset}_predictor_train.log" "$dataset predictor" \
                uv run python "$experiment/train_transformer.py" --seed "$seed"
        else
            echo ">>> $dataset predictor (checkpoints already complete)"
        fi
        if contains "$methods" ours && ! all_exist "$models/bc_full_split=__SPLIT__.pt"; then
            run_logged "$results_dir/${dataset}_timex_train.log" "$dataset TimeX" \
                uv run python "$experiment/bc_model_ptype.py" --seed "$seed"
        fi
        if contains "$methods" winit; then
            [[ -f txai/baselines/WinIT/winit/explainer/winitexplainers.py ]] || die "WinIT submodule is missing"
            run_logged "$results_dir/${dataset}_winit_train.log" "$dataset WinIT" \
                uv run python experiments/evaluation/winit_wrapper.py --dataset "$eval_dataset" \
                --models_path "$models" --epochs "$winit_epochs" --seed "$seed" --skip-existing
        fi
        if contains "$methods" cortx && ! all_exist "$models/cortx_split=__SPLIT__.pt"; then
            run_logged "$results_dir/${dataset}_cortx_train.log" "$dataset CoRTX" \
                uv run python experiments/other_baselines/train_table2.py --dataset "$dataset" --method cortx --seed "$seed"
        fi
        if contains "$methods" sgt+grad && ! all_exist "$models/sgt_split=__SPLIT__.pt"; then
            run_logged "$results_dir/${dataset}_sgt_train.log" "$dataset SGT + Grad" \
                uv run python experiments/other_baselines/train_table2.py --dataset "$dataset" --method sgt --seed "$seed"
        fi
    fi

    if [[ "$stage" == all || "$stage" == evaluate ]]; then
        for method in ${methods//,/ }; do
            case "$method" in
                ours) checkpoint="$models/bc_full_split=1.pt" ;;
                cortx) checkpoint="$models/cortx_split=1.pt" ;;
                sgt+grad) checkpoint="$models/sgt_split=1.pt" ;;
                *) checkpoint="$models/transformer_split=1.pt" ;;
            esac
            all_exist "${checkpoint/split=1/split=__SPLIT__}" || die "missing checkpoints for $dataset $method; run --stage train first"
            if [[ "$method" == winit ]]; then
                for split in 1 2 3 4 5; do [[ -d "$models/winit_split=$split" ]] || die "missing WinIT generator directory: $models/winit_split=$split"; done
            fi
            eval_extra=()
            if [[ -n "${TIMEX_MAX_SAMPLES:-}" ]]; then
                eval_extra+=(--max-samples "$TIMEX_MAX_SAMPLES")
            fi
            run_logged "$results_dir/${dataset}_${method//+/_}_evaluation.log" "$dataset $method evaluation" \
                uv run python experiments/evaluation/saliency_exp_synth.py \
                --dataset "$eval_dataset" --exp_method "$method" --split_no -1 \
                --model_path "$checkpoint" --seed "$seed" \
                --results-json "$results_dir/${dataset}_${method//+/_}_results.json" --no-progress \
                "${eval_extra[@]}"
        done
    fi
done

if [[ "$stage" == all || "$stage" == evaluate ]]; then
    uv run python experiments/evaluation/summarize_table2.py "$results_dir"
fi
echo "Table 2 workflow completed. Outputs: $results_dir"
