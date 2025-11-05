#!/usr/bin/env bash
# Run all case studies and generate artifacts

set -e  # Exit on error

echo "========================================"
echo "Running all pyBasin case studies"
echo "========================================"
echo ""

# Activate virtual environment if not already active
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    fi
fi

# Create artifacts directory if it doesn't exist
mkdir -p artifacts/{figures,results,reports}

echo "1. Running Duffing Oscillator case studies..."
echo "----------------------------------------"
# python case_studies/duffing_oscillator/main_supervised.py
# python case_studies/duffing_oscillator/main_unsupervised.py
echo "Skipped - to be run after refactoring"
echo ""

echo "2. Running Lorenz system case studies..."
echo "----------------------------------------"
# python case_studies/lorenz/main_lorenz.py
# python case_studies/lorenz/main_lorenz_sigma.py
# python case_studies/lorenz/main_lorenz_hyperpN.py
echo "Skipped - to be run after refactoring"
echo ""

echo "3. Running Pendulum case studies..."
echo "----------------------------------------"
# python case_studies/pendulum/main_case1.py
# python case_studies/pendulum/main_case2.py
echo "Skipped - to be run after refactoring"
echo ""

echo "4. Running Friction system case studies..."
echo "----------------------------------------"
# python case_studies/friction/main_friction.py
# python case_studies/friction/main_friction_v_study.py
echo "Skipped - to be run after refactoring"
echo ""

echo "========================================"
echo "All case studies completed!"
echo "Artifacts saved to: artifacts/"
echo "========================================"
