# fast-deep-photo-style-transfer-tf

Initial pipeline for deep photo style transfer [Luan et al., 2017](https://arxiv.org/abs/1703.07511) using
[Louie Yang's](https://github.com/LouieYang/deep-photo-styletransfer-tf) implementation, with optional
fast style transfer using [Logan Engstrom's](https://github.com/lengstrom/fast-style-transfer) implementation.

## Usage

```
python run_fpst.py --in-path inputPhotos/insightCorner.jpg
--style-path stylePhotos/leopard.jpg --checkpoint-path checkpoints/udnie.ckpt
--out-path test.jpg --slow
```

`--in-path` Path to the input `image.jpg`
`--style-path` Path to the style image `style.jpg`
`--checkpoint-path` Path to checkpoint file for fast style transfer algorithm `style.ckpt`
`out-path` Output filename `output.jpg`
`--slow` Specifies deep photo style transfer algorithm. Default is `False`, which switches to fast style transfer.
