#!/bin/bash
#for r in 9
#do

run(){     
        c=$1
        python train.py  \
            --trial_number $c \
            --train_number 70 \
            --test_number 2000 \
            --valid_number 70 \
            --modelname 2DCNN \
            --patchsize 11 \
            --gpu_ids 6 \
            --dim 33 \
            --kw 9 \
            --kb 8 &
}

#   for kw in  9
    # do 
        for c in 1 2 3 4 5 6 7 8 9 10 
            do
            run $c  
        done
    # done

 


