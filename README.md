# jojo-gan

## Usage
- First download necessary data using `python3 download_data.py`.
- The target image needs to present in the <a href="test_input"> test_input directory. </a>
- To use a pretrained model, run `python3 pretrained_style.py`. To use some other pretrained style, change the value of style parameter in line 105 from the set of options mentioned in the comment above it.
- Otherwise, to fine-tune the model for a custom style, run `python3 finetune_style.py`. The styles need to be present in the <a href="style_images"> style_images </a> directory and mentioned in line 117 as a list.