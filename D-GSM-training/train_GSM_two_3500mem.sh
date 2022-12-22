# !/bin/bash
echo " Running Training EXP"

#two continuous scenarios: MA(past)->FT(current)
CUDA_VISIBLE_DEVICES=0 python3 train_GSM.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 2-FT --tag social-stgcnn-FT --use_lrschd --num_epochs 250 --tasks 2 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 1 && echo "two continuous scenarios training Launched." &
P0=$!

#three continuous scenarios: MA(past)->FT(past)->ZS(current)
#CUDA_VISIBLE_DEVICES=0 python3 train_GSM.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 3-ZS --tag social-stgcnn-ZS --use_lrschd --num_epochs 250 --tasks 3 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 2 && echo "three continuous scenarios training Launched." &
#P1=$!

#four continuous scenarios: MA(past)->FT(past)->ZS(past)->EP(current)
#CUDA_VISIBLE_DEVICES=0 python3 train_GSM.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 4-EP --tag social-stgcnn-EP --use_lrschd --num_epochs 250 --tasks 4 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 3 && echo "four continuous scenarios training Launched." &
#P2=$!

#five continuous scenarios: MA(past)->FT(past)->ZS(past)->EP(past)->SR(current)
#CUDA_VISIBLE_DEVICES=0 python3 train_GSM.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5  --dataset 5-SR --tag social-stgcnn-SR --use_lrschd --num_epochs 250 --tasks 5 --mem_size 3500 --margin 0.5 --eps 0.001 --cur_task 4 && echo "five continuous scenarios training Launched." &
#P3=$!

wait $P0 
#$P1 $P2 $P3 
