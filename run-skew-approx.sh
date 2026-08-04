#!/bin/bash

set -euo pipefail

MAX_CONTAINERS="${MAX_CONTAINERS:-15}"
CORES_PER_CONTAINER="${CORES_PER_CONTAINER:-1}"
N_SKEW_SAMPLES="${N_SKEW_SAMPLES:-1000}"

INPUT_DIR=""
MODELS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input_dir)
            if [ "$#" -lt 2 ]; then
                echo "Error: --input_dir requires a value"
                exit 1
            fi
            INPUT_DIR="$2"
            shift 2
            ;;
        --*)
            echo "Error: unknown option '$1'"
            exit 1
            ;;
        *)
            MODELS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$INPUT_DIR" ]; then
    echo "Error: --input_dir is required"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: input directory not found: $INPUT_DIR"
    exit 1
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "Error: at least one model is required"
    exit 1
fi

job_idx=0
for model in "${MODELS[@]}"; do
    found_any=0

    while IFS= read -r dir; do
        found_any=1
        job_idx=$((job_idx + 1))

        container_name="skew_${model//[^a-zA-Z0-9_.-]/_}_${job_idx}"
        exp_dir="${dir#./}"

        echo "Launching ${container_name}: ${exp_dir}"
        docker run -d \
            --name "${container_name}" \
            --cpus="${CORES_PER_CONTAINER}" \
            -v "$(pwd)":/home \
            --user "$(id -u):$(id -g)" \
            bayes-el \
            python skew-approx.py "/home/${exp_dir}" "${N_SKEW_SAMPLES}"

        while [ "$(docker ps -q | wc -l)" -ge "$MAX_CONTAINERS" ]; do
            sleep 1
        done
    done < <(find "$INPUT_DIR" -maxdepth 1 -type d -name "model=${model} algo=ep *" | sort)

    if [ "$found_any" -eq 0 ]; then
        echo "No EP directories found for model=${model} under ${INPUT_DIR}"
    fi
done

echo "Submitted ${job_idx} skew-approx job(s)."
