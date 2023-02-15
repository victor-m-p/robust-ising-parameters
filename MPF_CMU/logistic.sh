#!/bin/bash
for size in 3 5 7
    do
    for beta in 0.25 0.5 1.0
        do
        ./mpf -g "../data/logistic/$size._500._$beta." $size 500 $beta
        ./mpf -c "../data/logistic/$size._500._$beta._data.dat"
    done
done