#!/bin/bash
for c in 1 2 3 4 5 6 7 8 9 10
    do
        for k in 4 
        do
            # k=$k+2
           python compare.py  \
           --model 1DCNN \
           --n 5 \
           --r 9 \
           --k $k \
           --savepath "./compared/2-2/1DCNN/${c}" \
           --dataset "./bloodcell2-2/Split/${c}/patch_11" \
           &
        
    done
done
