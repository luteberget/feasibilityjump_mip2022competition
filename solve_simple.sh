#!/bin/bash

# Simple solve script for the feasibility jump heuristic without Gurobi

if [ $# -eq 0 ]; then
    echo "Usage: $0 <input.mps> [output.sol]"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${2:-${INPUT_FILE%.*}.sol}"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

echo "Solving $INPUT_FILE without presolving..."
echo "Output will be saved to $OUTPUT_FILE"

# Build the project
cargo build --release

# Run the solver directly on the original MPS file
./target/release/lsmip "$INPUT_FILE" "$OUTPUT_FILE"

echo "Done!"
