#CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python train.py --resume checkpoint/checkpoint.pth.tar >> log.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python train.py > log.txt 2>&1 &
