#!/bin/bash
FILES="../data/hidden_nodes/questions*"
for f in $FILES
do
    nohup ./mpf -l "$f" 0
done 