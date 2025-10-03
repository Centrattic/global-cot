#!/bin/bash

# Simple script to run 7 parallel activation extraction processes
# Each process handles 50 completions

echo "Starting 4 parallel activation extraction processes..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Start all 7 processes in background
python -m src.activation_extractor --start 0 --end 5 > logs/process_1.log 2>&1 &
python -m src.activation_extractor --start 5 --end 10 > logs/process_2.log 2>&1 &
python -m src.activation_extractor --start 10 --end 15 > logs/process_3.log 2>&1 &

echo "All 4 processes started!"
echo "Log files:"
echo "  logs/process_1.log (completions 0-49)"
echo "  logs/process_2.log (completions 50-99)"
echo "  logs/process_3.log (completions 100-149)"
echo ""
echo "To monitor progress:"
echo "  tail -f logs/process_1.log"
echo "  tail -f logs/process_2.log"
echo "  etc."
echo ""
echo "Script finished. Processes are running in background."