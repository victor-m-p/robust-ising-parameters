#!/bin/bash
dir="../simulation/data/fully_connected_mpf"

generate_random_string() {
    cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1
}

for file in "$dir"/*
do
    for i in {1..10}
    do
        random_id=$(generate_random_string)
        output_file="${file}_${random_id}_log.txt"
        mpf -c "$file" > "$output_file"
    done
done
