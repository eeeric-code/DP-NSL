python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 12345 main.py \
    --config_yaml=yaml/dpnsl/dpnsl.yaml \
    --upscale 2,3,4 \
    --upscale_test 2,3,4,5 \
    --ckpt_dir train_dpnsl \
    --batch_size 8 \
    --num_workers 4 \
    --lr 1e-4 \
    --test_period=100 \
    --amp \

