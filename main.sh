#!/bin/bash
#for r in 9
#do

run(){     
        c=$1
        python train.py  \
            --train_number 4 \
            --test_number 2000 \
            --valid_number 4 \
            --modelname PURE3DCNN \
            --patchsize 11 \
            --gpu_ids 5 \
            --dim 30 \
            --kw 9 \
            --kb 8 \
            --dataset "./bloodcell1-3/Split/$c/" &
}

#   for kw in  9
    # do 
        for c in 1 2 3 4 5 6 7 8 9 10 
            do
            run $c  
        done
    # done

 


