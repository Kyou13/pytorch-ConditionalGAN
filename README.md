# pytorch-ConditionalGAN
## Description
ConditionalGANのpytorch実装

### Conditional GAN
[papaer link](https://arxiv.org/abs/1411.1784)
- ラベルも入力とし学習を行う
  - ラベルを指定することで任意の画像を精製することができる

## Requirement
- Python 3.7
- pytorch 1.1.0
- torchvision 0.3.0
- Click

## Usage
### Training
```
$ pip install -r requirements.txt
$ python main.py train
# training log saved at ./samples/fake_images-[epoch].png, ./samples/loss.png
```

### Generate
```
$ python main.py generate
# saved at ./samples/fake_images_%y%m%d%H%M%S.png
```
