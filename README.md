# image_segment-matting_method
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
```python
python tools/predict.py \
    --config configs/ppmatting/ppmatting-hrnet_w18-human_1024.yml \
    --model_path pretrained_models/ppmatting-hrnet_w18-human_1024.pdparams \
    --image_path demo/human.jpg \
    --save_dir ./ \
    --fg_estimate True \
```  
## forth try
- Since the matting method can only segment the human part, the mask obtained at the edge includes parts of the person, causing noise. Motivated by this, we used the trimap method to preprocess the image, enabling the matting algorithm to work only in areas needing optimization.
- run repalce_color.py
```python
python tools/predict.py \
    --config configs/ppmatting/ppmatting-hrnet_w18-human_1024.yml \
    --model_path pretrained_models/ppmatting-hrnet_w18-human_1024.pdparams \
    --image_path demo/human.jpg \
    --save_dir ./ \
    --fg_estimate True \
    --trimap_path /home/dengyiru/change_color/trimap.jpg
 ```
 ## fifth try
 - I find there are some specific figure like lace and button which will be dismissed by the segmentation or matting method,so I do a KNN clustering for color on the trimap to optimize the fine granularity in the specific area.
 - run focus_color.py to get the main color cluster.

 ## tip
 - SAM OPEN SOURCE:[https://github.com/facebookresearch/segment-anything]
 - PP-MATTING OPEN SOURCE:[https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/Matting/README.md/#Models]
 - my project report is in the docx file which show more detailed description of my work
