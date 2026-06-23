# DP-NSL: Dual-Prior Guided Null-Space Learning with Mixture-of-Splines for Arbitrary Medical Slice Super-Resolution (ECCV2026)

Haofei Song, Siyuan Xu, Xintian Mao, Shaojie Guo, Qingli Li, Yan Wang*

**Paper Link**: coming soon


## Environment

Install PyTorch following the official instructions for your CUDA version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The data preprocessing protocol follows the slice-based setting used by [I3Net](https://github.com/eeeric-code/I3Net). For the original preprocessing reference, please see the `data_prepare/` folder in the I3Net repository.

This repository already includes a small debug set under `data/`, so users can run the training and testing scripts without preparing the full dataset first.

```text
data/
  data_slice/
    Task10_Colon/
      imagesTr/
        colon_001/
          colon_001_slice_000.pt
          colon_001_slice_001.pt
          ...
  data_volume/
    Task10_Colon/
      imagesTs/
        colon_018.pt
```

For full experiments, prepare your dataset with the same directory structure, then update the paths in `config.py`, `yaml/dpnsl/dpnsl.yaml`, or the command line arguments as needed.

## Checkpoint

An official checkpoint is provided at:

```text
ckpt/1000.pth
```

## Configuration

The default model configuration is:

```bash
yaml/dpnsl/dpnsl.yaml
```


## Quick Run

Evaluate the provided checkpoint on the included debug test volume:

```bash
bash test.sh
```

Run training on the included debug training slices:

```bash
bash train.sh
```

For a shorter smoke test, you can run one epoch directly:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 12345 main.py \
    --config_yaml=yaml/dpnsl/dpnsl.yaml \
    --upscale 2,3,4 \
    --upscale_test 2,3,4 \
    --ckpt_dir debug_train \
    --batch_size 1 \
    --num_workers 4 \
    --lr 1e-4 \
    --test_period=1 \
    --save_ckpt_period=1 \
    --amp \
    --kwargs max_epoch=1
```

Training logs and checkpoints are written under:

```text
log/
experiments/
```

## Repository Structure

```text
.
├── main.py                  # training entry point
├── test.py                  # evaluation entry point
├── config.py                # CLI and default config
├── data.py                  # dataset loader
├── val.py                   # validation / evaluation loop
├── model_zoo/
│   ├── dpnsl/               # DP-NSL model
│   └── encoder/             # encoder backbones
├── data_prepare/            # preprocessing notes, adapted from I3Net
├── yaml/dpnsl/dpnsl.yaml    # model config
├── train.sh                 # training launcher
└── test.sh                  # evaluation launcher
```

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{
  coming soon
}
```

## Contact

If you have any question, please contact hfsong@stu.ecnu.edu.cn

