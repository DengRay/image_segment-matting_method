import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import os
 
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
 
 
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def load_model(model_path):
    sam_checkpoint = model_path
    device = "cuda"
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def show_image_axis(image,axis_path):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    #plt.imsave("/home/dengyiru/segment-anything-main/test/1.jpg",image)
    #show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.savefig(axis_path)

def get_mask(mask,score,dic):
    totle_mask = []
    totle_save_path = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        temp_num = i+10
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        #plt.imsave(f"/home/dengyiru/segment-anything-main/test/3.jpg",image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        sa_path = dic + f"{temp_num}.jpg"
        plt.savefig(sa_path)
        totle_mask.append(mask)
        totle_save_path.append(sa_path)
    return totle_mask,totle_save_path

def dilate_and_erode(mask_data, struc="ELLIPSE", size=(10, 10)):
    """
    膨胀侵蚀作用，得到粗略的trimap图
    :param mask_data: 读取的mask图数据
    :param struc: 结构方式
    :param size: 核大小
    :return:
    """
    if struc == "RECT":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    elif struc == "CORSS":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    msk = mask_data / 255

    dilated = cv2.dilate(msk, kernel, iterations=1) * 255
    eroded = cv2.erode(msk, kernel, iterations=1) * 255
    res = dilated.copy()
    res[((dilated == 255) & (eroded == 0))] = 128
    return res

nn = 19
model_path = "/home/dengyiru/segment-anything-main/check_point/sam_vit_h_4b8939.pth"
input_image_path =f"/home/dengyiru/change_color/test1/{nn}.jpg"
input_image_axis_path = "/home/dengyiru/segment-anything-main/test/2.jpg"
save_mask_dic = "/home/dengyiru/segment-anything-main/test/"
final_mask_save_path = '/home/dengyiru/change_color/mask.jpg'
final_mask_save_path_2 = '/home/dengyiru/change_color/mask2.jpg'
final_mask_save_path_3 = '/home/dengyiru/change_color/trimap4.jpg'
output_image_path = f'/home/dengyiru/change_color/result/3_{nn}.jpg'

image = cv2.imread(input_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
show_image_axis(image,input_image_axis_path)
predictor = load_model(model_path)
predictor.set_image(image)
input_label = np.array([1])
print(f"the axis image is save in path:{input_image_axis_path}\nPlease choose the point you want split")
x1 = float(input("Enter the x-coordinate of the first point: "))
y1 = float(input("Enter the y-coordinate of the first point: "))
input_point = np.array([[x1, y1]])#选择的标注点

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

totle_mask,image_files = get_mask(masks,scores,save_mask_dic)
for i, image_file in enumerate(image_files):   # 显示可选图像的列表
    print(f"{i}: {image_file}")
selected_index = int(input("Enter the index of the image you want to select: "))
temp = totle_mask[selected_index]
new_mask = [[255 if item else 0 for item in sublist] for sublist in temp]
new_mask = np.array(new_mask)
new_mask = new_mask.astype(np.uint8)
#print(type(new_mask))
#print(new_mask.shape)
#print(list(new_mask))

cv2.imwrite(final_mask_save_path_2, new_mask)
image = cv2.imread(input_image_path) # 加载图像
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # 转换到 HSV 空间
result = cv2.bitwise_and(image, image, mask=new_mask)
cv2.imwrite(final_mask_save_path, result)

# 颜色转变
h, s, v = cv2.split(hsv)
h[new_mask > 0] = 120# 改变色调值为50
s[new_mask > 0] = 80  # 改变饱和度值为10
#v[new_mask > 0] += 50
hsv_mod = cv2.merge([h, s, v])
image_bright = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR) # 转回 RGB 空间
cv2.imwrite(output_image_path, image_bright) # 保存处理后的图像
print(f"color change has done! the output save in {output_image_path}")

size = 20
trimap = dilate_and_erode(new_mask, size=(size, size))
cv2.imwrite(final_mask_save_path_3, trimap)

# 打印 HSV 矩阵
#print(hsv[25][30])
#print(hsv.shape)
# 对 HSV 矩阵进行处理
# 例如，我们可以增加 V 通道（亮度）的值以使图像更亮
# 找到青色的区域
'''
lower_cyan = np.array([17, 43, 46])  # 这些值可能需要调整
upper_cyan = np.array([18, 100, 255])  # 这些值可能需要调整
cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
'''
