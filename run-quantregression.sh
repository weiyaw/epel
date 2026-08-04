#!/bin/bash

# Initialize variables
id="${1}"
mode="${2:-}"
model="quantregression"

if [ "$mode" = "--gold" ]; then
    mode="gold"
fi

if [ "$mode" = "gold" ]; then
    MAX_CONTAINERS=10               # Legacy gold-standard limit
    CORES_PER_CONTAINER=6           # Legacy gold-standard CPU setting
    seeds=$(seq 304900 304909)      # Legacy gold-standard seeds (10 reps)
    algorithms="rw"
else
    MAX_CONTAINERS=15               # Limit to 15 containers running simultaneously
    CORES_PER_CONTAINER=4           # Number of CPUs per container
    seeds=$(seq 991100 991149)      # List of seeds (50 reps)
    algorithms="ep hmc rw vi"
fi

# Loop through algorithms
for algorithm in $algorithms; do
    # Set algorithm-specific parameters
    if [ "$mode" = "gold" ]; then
        algo_params="algorithm=rw algorithm.rw.sd_shrink_factor=0.7 algorithm.rw.n_samples=2000000 algorithm.rw.n_warmup=5000 algorithm.rw.thinning=10"
    else
        case "$algorithm" in
            "ep")
                algo_params="algorithm=ep algorithm.ep.n_points=20"
                ;;
            "hmc")
                algo_params="algorithm=hmc"
                ;;
            "rw")
                algo_params="algorithm=rw algorithm.rw.sd_shrink_factor=0.7 algorithm.rw.n_samples=3000000 algorithm.rw.thinning=30"
                ;;
            "vi")
                algo_params="algorithm=vi"
                ;;
        esac
    fi

    # Loop through seed values
    for seed in $seeds; do
        # Generate a unique container name
        container_name="${model}_${algorithm}_${seed}"

        # Run the 'bayes-el' container with the specified CPU limit
        docker run -d \
            --name "${container_name}" \
            --cpus="${CORES_PER_CONTAINER}" \
            -v "$(pwd):/home" \
            --user $(id -u):$(id -g) \
            bayes-el \
            python main.py model=${model} seed=${seed} id=${id} \
            ${algo_params}

        # Wait if the number of running containers reaches the limit
        while [ $(docker ps -q | wc -l) -ge $MAX_CONTAINERS ]; do
            sleep 1
        done
    done
done
