#!/bin/bash

# ==============================================================================
# Master Execution Script for Federated Learning IDS Benchmarks
# ==============================================================================
# Note: This script uses '--benchmark True' to suppress per-round prints 
# and speed up execution by keeping the terminal clean.

# Base variables (from your successful stabilization testing)
PYTHON="python3 src/main.py"

# Arrays for algorithms
ALL_FL_ALGOS=("fedavg" "fedprox" "scaffold" "ditto")

echo "======================================================="
echo " STARTING EXPERIMENT SUITE"
echo "======================================================="

# ------------------------------------------------------------------------------
# EXPERIMENT 1: Baselines & Core Benchmark
# Evaluates Centralized + all FL algorithms with standard parameters.
# ------------------------------------------------------------------------------

#echo -e "\n>>> Running Exp 1: Centralized Baseline"
#$PYTHON --algorithm centralized --benchmark True --exp_name "Exp1_Core" --local_epochs 1

NUM_CLIENTS=(24 12 4)

for num in "${NUM_CLIENTS[@]}"; do
    for algo in "${ALL_FL_ALGOS[@]}"; do
        echo -e "\n>>> Running Exp 1: Core Benchmark -> Algorithm: $algo"
        $PYTHON --algorithm $algo --benchmark True --exp_name "Exp6_Num_Clients${num}" --num_clients $num
    done
done

<<"END"
# ------------------------------------------------------------------------------
# EXPERIMENT 2: System Scalability (Participation Rate C)
# C in {0.1, 0.3, 0.5} (C=1.0 is already covered in Exp1_Core)
# ------------------------------------------------------------------------------
FRACTIONS=(0.1 0.3 0.5)

for c in "${FRACTIONS[@]}"; do
    for algo in "${ALL_FL_ALGOS[@]}"; do
        echo -e "\n>>> Running Exp 2: Scalability (C=$c) -> Algorithm: $algo"
        $PYTHON --algorithm $algo --client_fraction $c \
                --benchmark True --exp_name "Exp2_Scalability_C${c}"
    done
done


# ------------------------------------------------------------------------------
# EXPERIMENT 4: Sensitivity Analysis - FedProx Proximal Term (mu)
# mu in {0.001, 0.01, 1.0} (0.1 is already covered in Exp1_Core)
# ------------------------------------------------------------------------------
MUS=(0.001 0.01 1.0)

for mu in "${MUS[@]}"; do
    echo -e "\n>>> Running Exp 4: FedProx Mu (mu=$mu)"
    $PYTHON --algorithm fedprox --mu $mu \
            --benchmark True --exp_name "Exp4_FedProx_Mu${mu}"
done


# ------------------------------------------------------------------------------
# EXPERIMENT 5: Sensitivity Analysis - Ditto Regularization (lambda)
# lambda in {0.1, 1.0} (0.5 is already covered in Exp1_Core)
# ------------------------------------------------------------------------------
LAMS=(0.1 1.0)

for lam in "${LAMS[@]}"; do
    echo -e "\n>>> Running Exp 5: Ditto Lambda (lam=$lam)"
    $PYTHON --algorithm ditto --lam $lam  \
            --benchmark True --exp_name "Exp5_Ditto_Lam${lam}"
done

# ------------------------------------------------------------------------------
# EXPERIMENT 3: Sensitivity Analysis - Local Epochs (E)
# E in {1, 5, 10}
# ------------------------------------------------------------------------------
EPOCHS=(1 5 10)

for e in "${EPOCHS[@]}"; do
    for algo in "${ALL_FL_ALGOS[@]}"; do
        echo -e "\n>>> Running Exp 3: Local Epochs (E=$e) -> Algorithm: $algo"
        $PYTHON --algorithm $algo --local_epochs $e \
                --benchmark True \
                --exp_name "Exp3_Epochs_E${e}"
    done
done

END

echo "======================================================="
echo " ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "======================================================="