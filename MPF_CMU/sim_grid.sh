#!/bin/bash
dir="../simulation/data/not_connected_nn21_nsim500_l1_mpf"
generate_random_string() {
    cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1
}

# Generate grid values from -1.0 to 1.0 with increments of 0.05
grid_values=()
for i in $(seq -w -1.0 0.1 1.0); do
    grid_values+=($i)
done

for file in "$dir"/*
do
    for grid_value in "${grid_values[@]}"
    do
        for i in {1..32}
        do
            random_id=$(generate_random_string)
            output_file="${file}_${grid_value}_${random_id}_log.txt"
            ./mpf -l "$file" "$grid_value" 1.0 > "$output_file" &
            # If the number of background jobs is equal to the number of cores, wait for them to finish
            if (( i % $(nproc) == 0 )); then
                wait
            fi
        done
    done
    # Wait for any remaining processes to complete
    wait
done
