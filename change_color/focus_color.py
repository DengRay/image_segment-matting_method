import time
from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from skimage import transform
from skimage.io import imread
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import cv2
from PIL import Image

# calculating time
def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Spend time:{time.time() - start} s")
        return result

    return wrapper


@time_it
def get_best_k(src):
    K_list = [i for i in range(2, 8)]
    scores = []
    for i in K_list:
        mbk = MiniBatchKMeans(n_clusters=i, random_state=0)
        scores.append(silhouette_score(src, mbk.fit_predict(src), sample_size=int(src.shape[0] / 128)))
    index = scores.index(max(scores))
    best_k = K_list[index]
    print(f'best_k:', best_k)
    return best_k


def get_main_Color(src):
    # to_HSV
    src_temp = src[:, :, :3]
    img = rgb2hsv(src_temp)
    #rgb_img = cv2.cvtColor(src, cv2.COLOR_RGBA2RGB)
    #img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    h, w, d = img.shape
    img = np.reshape(img, (h * w, d))
    #k = get_best_k(img)
    bk = MiniBatchKMeans(n_clusters=4, random_state=0)
    bk.fit(img)
    result = bk.cluster_centers_
    #print(f"hh{result}")
    # to_RGB
    maincolor = hsv2rgb(result) * 255
    return maincolor ,result


def maincolor_show(maincolor):
    N = len(maincolor)
    img = np.zeros((300, 100 * N, 3), np.uint8)

    for i in range(N):
        img[:, 100 * i:100 * (i + 1)] = [maincolor[i]]
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.savefig('/home/dengyiru/change_color/focus_result/temp.jpg')

def mask_maincolor(maincolor,pic):
    N = len(maincolor)

    main_oringin = np.array(maincolor[3])
    temp = np.array([20/180,60/255,40/255])
    main_max = main_oringin  + temp
    main_min = main_oringin - temp
    src_temp = pic[:, :, :3]
    #img = cv2.cvtColor(src_temp, cv2.COLOR_BGR2HSV)
    img = rgb2hsv(src_temp)
    #img = np.array(img_hsv)
    #img = img * 255
    #pp = np.array(img)
    #print(pp.shape)
    h, w, d = img.shape
    mask =  [[0 for _ in range(w)] for _ in range(h)]
    h_channel = img[:, :, 0]
    s_channel = img[:, :, 1]
    v_channel = img[:, :, 2]
    #print(main_max)
    #print(main_min)
    #print(h_channel)
    for i in range(h):
        for j in range(w):
            if main_min[0]<h_channel[i][j]<= main_max[0] and main_min[1]<s_channel[i][j]<= main_max[1] and main_min[2]<v_channel[i][j]<= main_max[2]:
                mask[i][j] = 255

    #data = [[255, 0, 255, 0], [0, 255, 0, 255], [255, 0, 255, 0], [0, 255, 0, 255]]
    #print(mask)
    np_data = np.array(mask, dtype=np.uint8)
    #img = np.dstack([np_data,np_data,np_data])
    # 创建一个图像对象
    img = Image.fromarray(np_data,"L")

    # 保存图像
    img.save('/home/dengyiru/change_color/focus_result/mask.jpg')




def main():
    #img = Image.open("/home/dengyiru/change_color/test2/18.jpg")
    #background = Image.new('RGBA', img.size, (255,255,255))
    #alpha_composite = Image.alpha_composite(background, img)
    #rgb_img=alpha_composite.convert('RGB')
    # 使用 RGB 模式将图像转换为 JPG
    #rgb_img = img.convert('RGB')
    #src = np.array(rgb_img)
    # 将图像保存为 JPG 格式
    #rgb_img.save("output.jpg")
    src = imread("/home/dengyiru/change_color/test2/18.jpg")
    plt.imshow(src)
    plt.axis('off')
    plt.show()
    plt.savefig('/home/dengyiru/change_color/focus_result/temp2.jpg')
    print(type(src))
    t = np.array(src)
    print(t.shape)

    #src = transform.rescale(src, [0.4, 0.4, 1])
    #src = transform.rescale(src, [1, 1, 1])
    maincolor,maincolor_hsv = get_main_Color(src)
    print(maincolor_hsv)
    # visualization
    maincolor_show(maincolor)
    mask_maincolor(maincolor_hsv,src)


if __name__ == '__main__':
    main()


