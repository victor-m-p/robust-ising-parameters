#!/bin/bash
dir="../simulation/data/fully_connected_mpf"
for file in "$dir"/*
do 
	mpf -c "$file"
done