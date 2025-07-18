#!/usr/bin/env bash

set -u 

# Define the range of values for each parameter
BK_VALUES=(8 16 32 64)
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
WM_VALUES=(32 64 128 256)
WN_VALUES=(32 64 128 256)
WNITER_VALUES=(1 2 4 8)
TM_VALUES=(8)
TN_VALUES=(1)
NUM_THREADS_VALUES=(128 256)

RUNNER="./profiler.cu"
OUTPUT="./results_oned_tm8.txt"

# Clear the output file
echo "" > $OUTPUT

# Set GPU to use
export DEVICE="0"
WARPSIZE=32
MAX_SHARED_MEM=101376


TOTAL_CONFIGS="$(( ${#BK_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} * ${#WM_VALUES[@]} * ${#WN_VALUES[@]} * ${#WNITER_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#NUM_THREADS_VALUES[@]} ))"
CONFIG_NUM=0

# Loop through all combinations of parameters
for BK in "${BK_VALUES[@]}"; do
for BM in "${BM_VALUES[@]}"; do
for BN in "${BN_VALUES[@]}"; do
for WM in "${WM_VALUES[@]}"; do
for WN in "${WN_VALUES[@]}"; do
for WN_ITER in "${WNITER_VALUES[@]}"; do
for TM in "${TM_VALUES[@]}"; do
for TN in "${TN_VALUES[@]}"; do
for NUM_THREADS in "${NUM_THREADS_VALUES[@]}"; do
SHARED_MEM_SIZE=$(( (BM * BK + BK * BN) * 4 ))
CONFIG_NUM=$(( CONFIG_NUM + 1 ))
# skip configurations that don't fullfil preconditions
NUM_WARPS=$(( NUM_THREADS / 32 ))
if ! (( BN % WN == 0 && BM % WM == 0 )); then
  continue
fi
if (( SHARED_MEM_SIZE > MAX_SHARED_MEM )); then
  continue
fi
if ! (( (BN / WN) * (BM / WM) == NUM_WARPS )); then
  continue
fi
if ! (( (WM * WN) % (WARPSIZE * TM * TN * WN_ITER) == 0 )); then
  continue
fi
WM_ITER=$(( (WM * WN) / (WARPSIZE * TM * TN * WN_ITER) ))
if ! (( WM % WM_ITER == 0 && WN % WN_ITER == 0 )); then
  continue
fi
if ! (( (NUM_THREADS * 4) % BK == 0 )); then
  continue
fi
if ! (( (NUM_THREADS * 4) % BN == 0 )); then
  continue
fi
if ! (( BN % (16 * TN) == 0 )); then
  continue
fi
if ! (( BM % (16 * TM) == 0 )); then
  continue
fi
if ! (( (BM * BK) % (4 * NUM_THREADS) == 0 )); then
  continue
fi
if ! (( (BN * BK) % (4 * NUM_THREADS) == 0 )); then
  continue
fi

# Update the parameters in the source code
sed -i "s/const uint K10_NUM_THREADS = .*/const uint K10_NUM_THREADS = $NUM_THREADS;/" $RUNNER
sed -i "s/const uint K10_BN = .*/const uint K10_BN = $BN;/" $RUNNER
sed -i "s/const uint K10_BM = .*/const uint K10_BM = $BM;/" $RUNNER
sed -i "s/const uint K10_BK = .*/const uint K10_BK = $BK;/" $RUNNER
sed -i "s/const uint K10_WM = .*/const uint K10_WM = $WM;/" $RUNNER
sed -i "s/const uint K10_WN = .*/const uint K10_WN = $WN;/" $RUNNER
sed -i "s/const uint K10_WNITER = .*/const uint K10_WNITER = $WN_ITER;/" $RUNNER
sed -i "s/const uint K10_TM = .*/const uint K10_TM = $TM;/" $RUNNER
sed -i "s/const uint K10_TN = .*/const uint K10_TN = $TN;/" $RUNNER

# Rebuild the program
nvcc profiler.cu -o sgemm || { echo "Compilation failed"; break; }
echo "($CONFIG_NUM/$TOTAL_CONFIGS): BK=$BK BM=$BM BN=$BN WM=$WM WN=$WN WN_ITER=$WN_ITER TM=$TM TN=$TN NUM_THREADS=$NUM_THREADS" |& tee -a $OUTPUT
# Run the benchmark and get the result
# Kill the program after 4 seconds if it doesn't finish
timeout -k 2 8 ./sgemm 10 | tee -a $OUTPUT
done
done
done
done
done
done
done
done
done
