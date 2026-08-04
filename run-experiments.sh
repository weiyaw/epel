#!/bin/bash

# Function to remove all stopped containers that ran successfully
cleanup() {
  docker ps -a -q -f exited=0 | xargs -r docker rm
}

# Function to wait until all Docker containers have stopped
wait_for_docker() {
  animation=("." ".." "...")
  while [ -n "$(docker ps -q)" ]; do
    for frame in "${animation[@]}"; do
      echo -ne "\r\033[KWaiting for running Docker containers to finish$frame "
      sleep 1
    done
  done
  echo -e "\r\033[KAll Docker containers have stopped." # Clear the line
  cleanup
}


OUTPUT_DIR="main-exp"
GOLD_DIR="gold"

wait_for_docker
./run-regression.sh ${GOLD_DIR} --gold
./run-regression10.sh ${GOLD_DIR} --gold
./run-gee.sh ${GOLD_DIR} --gold
./run-kyphosis.sh ${GOLD_DIR} --gold
./run-mroz.sh ${GOLD_DIR} --gold
./run-quantregression.sh ${GOLD_DIR} --gold
./run-orings.sh ${GOLD_DIR} --gold
./run-breastfeed.sh ${GOLD_DIR} --gold

wait_for_docker
./run-regression.sh ${OUTPUT_DIR}

wait_for_docker
./run-regression10.sh ${OUTPUT_DIR}

wait_for_docker
./run-gee.sh ${OUTPUT_DIR}

wait_for_docker
./run-kyphosis.sh ${OUTPUT_DIR}

wait_for_docker
./run-mroz.sh ${OUTPUT_DIR}

wait_for_docker
./run-quantregression.sh ${OUTPUT_DIR}

wait_for_docker
./run-orings.sh ${OUTPUT_DIR}

wait_for_docker
./run-breastfeed.sh ${OUTPUT_DIR}

# post-processing
wait_for_docker
./run-skew-approx.sh --input_dir outputs/${OUTPUT_DIR} orings breastfeed

# compute nbp
wait_for_docker
for model in gee kyphosis mroz regression regression10 quantregression; do
    Rscript compute-stats.R --model=${model} --input_dir outputs/${OUTPUT_DIR} --save_dir outputs/nbp/${OUTPUT_DIR} --gold_dir outputs/${GOLD_DIR} --cores 50
done

for model in orings breastfeed; do
    Rscript compute-stats.R --model=${model} --skew --input_dir outputs/${OUTPUT_DIR} --save_dir outputs/nbp/${OUTPUT_DIR} --gold_dir outputs/${GOLD_DIR} --cores 50
done
