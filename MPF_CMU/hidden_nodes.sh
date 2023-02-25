#!/bin/bash
FILES="../data/hidden_nodes_0.25_logs0/questions*"

for f in $FILES
do
    nohup ./mpf -l "$f" 0
done 