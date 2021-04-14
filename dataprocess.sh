#!/bin/bash
for c in 1 2 3 4 5 6 7 8 9 10
do
    for n in 4  
    do
        python datasetsplit.py \
        --dataset bloodcell1-3 \
        --savepath ./bloodcell1-3/Split/$c/ \
        --ntrain $n\
        --nvalid $n\
        --patchsize 11 &
    done
done
