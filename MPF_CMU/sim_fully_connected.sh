#!/bin/bash
dir="../simulation/data/fully_connected_mpf_big"

generate_random_string() {
    cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1
}

for file in "$dir"/*
do
    for i in {1..32}
    do
        random_id=$(generate_random_string)
        output_file="${file}_${random_id}_log.txt"
        ./mpf -l "$file" -2.0 2.0 > "$output_file" &

        # If the number of background jobs is equal to the number of cores, wait for them to finish
        if (( i % $(nproc) == 0 )); then
            wait
        fi
    done

    # Wait for any remaining processes to complete
    wait
done
