import cv2
import numpy as np

num = 4
# 加载图像
image = cv2.imread(f'/home/dengyiru/change_color/test1/{num}.jpg')
#resized_image = cv2.resize(image, (50, 50))
# 转换到 HSV 空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 打印 HSV 矩阵

#print(hsv[25][30])
print(hsv.shape)

# 对 HSV 矩阵进行处理
# 例如，我们可以增加 V 通道（亮度）的值以使图像更亮
# 找到青色的区域
lower_cyan = np.array([100, 43, 46])  # 这些值可能需要调整
upper_cyan = np.array([118, 255, 255])  # 这些值可能需要调整
cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
print(cyan_mask)
print(type(cyan_mask))
print(cyan_mask.shape)

result = cv2.bitwise_and(image, image, mask=cyan_mask)
cv2.imwrite('/home/dengyiru/change_color/mask.jpg', result)
# 将青色区域变为白色
h, s, v = cv2.split(hsv)
h[cyan_mask > 0] = 30# 改变色调值为50
s[cyan_mask > 0] =100  # 改变色调值为50
#v[cyan_mask > 0] = 6
hsv_mod = cv2.merge([h, s, v])
#hsv[cyan_mask][0] = 0  # 在 HSV 中，白色的表示为 (0, 0, 255)

# 转回 RGB 空间
image_bright = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)
#esize = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# 保存处理后的图像
#cv2.imwrite('/home/dengyiru/change_color/image_origin.jpg', resized_image)
cv2.imwrite(f'/home/dengyiru/change_color/result/{num}.jpg', image_bright)
