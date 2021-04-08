#!/bin/bash
#for r in 9
#do

run(){     
        c=$1
        python train.py  \
            --train_number 80 \
            --test_number 2000 \
            --valid_number 80 \
            --modelname 2DCNN \
            --patchsize 11 \
            --gpu_ids 1 \
            --dim 33 \
            --kw 4 \
            --kb 26 \
            --dataset "./bloodcell2-2/Split/$c/" &
}

#   for kw in  9
    # do 
        for c in 1 2 3 4 5 6 7 8 9 10
            do
            run $c 
        done
    # done

 


