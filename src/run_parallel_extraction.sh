#!/bin/bash

# Script to run 7 parallel activation extraction processes
# Each process handles 50 completions

echo "Starting 7 parallel activation extraction processes..."
echo "Each process will handle 50 completions"
echo ""

# Function to run extraction for a given range
run_extraction() {
    local start=$1
    local end=$2
    local process_id=$3
    
    echo "Starting process $process_id: completions $start to $((end-1))"
    
    # Run the extraction in background
    python src/activation_extractor.py --start $start --end $end > "logs/process_${process_id}_${start}_${end}.log" 2>&1 &
    
    # Store the PID
    local pid=$!
    echo "Process $process_id (PID: $pid) started for range $start-$((end-1))"
    
    # Return the PID
    echo $pid
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Array to store PIDs
pids=()

# Start 7 processes
echo "Launching processes..."

# Process 1: 0-49
pid1=$(run_extraction 0 50 1)
pids+=($pid1)

# Process 2: 50-99
pid2=$(run_extraction 50 100 2)
pids+=($pid2)

# Process 3: 100-149
pid3=$(run_extraction 100 150 3)
pids+=($pid3)

# Process 4: 150-199
pid4=$(run_extraction 150 200 4)
pids+=($pid4)

# Process 5: 200-249
pid5=$(run_extraction 200 250 5)
pids+=($pid5)

# Process 6: 250-299
pid6=$(run_extraction 250 300 6)
pids+=($pid6)

# Process 7: 300-349
pid7=$(run_extraction 300 350 7)
pids+=($pid7)

echo ""
echo "All 7 processes started!"
echo "PIDs: ${pids[@]}"
echo ""
echo "Log files are being written to:"
echo "  logs/process_1_0_50.log"
echo "  logs/process_2_50_100.log"
echo "  logs/process_3_100_150.log"
echo "  logs/process_4_150_200.log"
echo "  logs/process_5_200_250.log"
echo "  logs/process_6_250_300.log"
echo "  logs/process_7_300_350.log"
echo ""

# Function to check if all processes are still running
check_processes() {
    local all_running=true
    for pid in "${pids[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            all_running=false
            break
        fi
    done
    echo $all_running
}

# Monitor processes
echo "Monitoring processes... (Press Ctrl+C to stop monitoring)"
echo ""

while true; do
    if check_processes; then
        echo -n "."
        sleep 10
    else
        echo ""
        echo "One or more processes have completed!"
        break
    fi
done

echo ""
echo "Checking final status of all processes..."

# Check final status
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    process_num=$((i+1))
    if kill -0 $pid 2>/dev/null; then
        echo "Process $process_num (PID: $pid) is still running"
    else
        echo "Process $process_num (PID: $pid) has completed"
    fi
done

echo ""
echo "Script finished. Check the log files for detailed output."
echo "To view a log file: tail -f logs/process_X_Y_Z.log"
