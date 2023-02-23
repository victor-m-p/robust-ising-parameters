#!/bin/bash
FILES="../data/sample_questions/mdl_config/*"

for f in $FILES
do
    nohup ./mpf -c "$f" 
done 