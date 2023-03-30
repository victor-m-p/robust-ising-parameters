#!/bin/bash
dir="../data/sim/bias_mpf"
for file in "$dir"/*
do 
	mpf -c "$file"
done
