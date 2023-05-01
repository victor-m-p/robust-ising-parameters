#!/bin/bash
dir="../simulation/data/not_connected_mpf"
for file in "$dir"/*
do 
	mpf -c "$file"
done
