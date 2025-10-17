#!/bin/bash

# 要测试的temperature值
for temp1 in 64 48; do
    echo "Testing temperature1 = $temp1"
    
    python train_fsl.py --max_epoch 200 --model_class ProtoNet --backbone_class Res12 --dataset MiniImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.01 --temperature $temp1 --temperature2 16 --lr 0.0005 --lr_mul 40 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 0 --init_weights ./saves/initialization/miniimagenet/protonet-5-shot.pth --eval_interval 1 --use_euclidean --augment --seed 3407
    
    echo "Completed temperature1 = $temp1"
done