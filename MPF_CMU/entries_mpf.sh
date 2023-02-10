#!/bin/bash
FILES="../data/sample_questions/mdl/*"

for f in $FILES
do
    nohup ./mpf -c "$f" > "question_fit_5.txt"
done 