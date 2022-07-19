# jojo-gan

This repository aims to perform one-shot image stylization getting the facial stylistic details right. Given a reference style image, paired real data is approximated using GAN inversion and a pretrained StyleGAN is fine-tuned using that approximate paired data. The StyleGAN is then encouraged to generalize so that the learned style can be applied to all other images.

## Improvements

Added support for `restyle` GAN inverter which performs better than the already existing `e4e`. This also improves the overall performance of JoJoGAN. Also created well-documented and modular Python scripts to easily pre-train and fine-tune JoJoGAN. 

## Usage

1. Put the target image into <a href="test_input">test_input</a>

2. Download the required data and models using  
`python3 download_data.py`

3. 1. To use a pretrained JoJoGAN model, run    
`python3 pretrained_style.py`    
To use other pretrained styles, change the value of style <a href="https://github.com/007prateekd/jojo-gan/blob/c8d4bab4853355842f2259497d990c134a8befd4/pretrained_style.py#L105">here</a> from the set of options mentioned in the comment above it

   2. To fine-tune a StyleGAN2 model for a custom style, run    
`python3 finetune_style.py`   
The styles need to be present in the <a href="style_images">style_images</a> and mentioned <a href="https://github.com/007prateekd/jojo-gan/blob/c8d4bab4853355842f2259497d990c134a8befd4/finetune_style.py#L130">here</a> as a list. Note that the style must have a face in it for _dlib_ to detect. If it fails to detect, then manually crop the style image and put into the <a href="style_images_aligned">style_images_aligned</a>

## Acknowledgements
This code borrows from <a href="https://github.com/mchong6/JoJoGAN">JoJoGAN</a> and <a href="https://github.com/yuval-alaluf/restyle-encoder">ReStyle</a>.
