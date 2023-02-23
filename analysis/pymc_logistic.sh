#!/bin/bash
for size in 3 5 7 9 11
    do
    for beta in 0.25 0.5 1.0
        do
        python pymc_logistic.py -q $size -s 500 -b $beta
    done
done