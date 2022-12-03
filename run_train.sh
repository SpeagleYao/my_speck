#!/usr/bin/env bash

for nr in `echo 5 6 7`
do
        echo "nr:" ${nr}
        python train_gohr.py --epochs 200 --nr ${nr} > log_my/log_res_gohr_${nr}r_mse_200e.txt
done