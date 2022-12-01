#!/usr/bin/env bash

for nr in `echo 5 6 7`
do
        echo "nr:" ${nr}
        python train_gohr.py --nr ${nr} > log/log_gohr_${nr}r_d1_20e.txt
done