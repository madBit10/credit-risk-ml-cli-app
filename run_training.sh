#!/bin/bash

echo "Starting Credit Risk ML Training Pipeline..."
echo "--------------------------------------------"

# Run tests first
echo "Running tests..."

python3 -m pytest tests/ -v

if [ $? -eq 0 ]; then 
    echo "Tests passed. Starting training..."
    python3 -m src.main --train
    echo "Pipeline complete!"
else
    echo "Tests failed. Traning aborted."
fi


