# picture_segment-matting_method
Internship program----Infimind project to change color on pictures with SAM model and PP-matting model

This project is aim to change color at fasion domain, for example we intend to remain the pattern and design the same and only change color with CV algorithm.

## first try
- We attempt to identify the relevant color regions in the HSV space, apply a mask, and then adjust the HSV values in the masked area to modify the color
- run change_color.py

## second try
- We attempt to use the 'Segment Anything' model for image segmentation, obtaining the areas where we need to change the color, and then apply color modifications to these areas
- run repalce_color.py

## third try
- We found that at the edges of the image, the SAM model does not provide a good mask, with some missing parts at the masked edges. Motivated by this, we attempted edge optimization using matting methods
- ```python
python tools/predict.py \
    --config configs/ppmatting/ppmatting-hrnet_w18-human_1024.yml \
    --model_path pretrained_models/ppmatting-hrnet_w18-human_1024.pdparams \
    --image_path demo/human.jpg \
    --save_dir ./ \
    --fg_estimate True \
![image](https://github.com/DengRay/picture_segment-matting_method/assets/76397795/f6af8215-d5c7-4024-8776-0a55d873e557)


