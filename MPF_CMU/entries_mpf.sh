#!/bin/bash
FILES="../data/sample_questions/mdl_10/*"

for f in $FILES
do
    nohup ./mpf -c "$f" > "question_fit_10x.txt"
done 