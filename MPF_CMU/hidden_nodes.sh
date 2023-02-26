#!/bin/bash
FILES="../data/hidden_nodes_0.25/questions*"

for f in $FILES
do
    nohup ./mpf -c "$f" 
done 