#!/bin/bash
for c in 1 2 3 4 5 6 7 8 9 10
do
    for n in 5 10 20 40 60 80  
    do
        python datasetsplit.py \
        --dataset bloodcell2-2 \
        --savepath ./bloodcell2-2/Split/$c/ \
        --ntrain $n\
        --nvalid $n\
        --patchsize 11 &
    done
done
