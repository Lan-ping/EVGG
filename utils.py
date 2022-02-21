import cv2
import imgaug as ia
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from imgaug import augmenters as iaa

types = os.listdir('./data')
print("数据类别:",len(types))
def generator(Imgs, Labels, batch_size=8, precessFunc=None):
    seq = iaa.Sequential([

        iaa.Fliplr(0.5),  # 0.5的概率水平翻转
        iaa.Crop(percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1])),  # random crops
        # sigma在0~0.5间随机高斯模糊，且每张图纸生效的概率是0.5
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),
        # 增大或减小每张图像的对比度
        iaa.ContrastNormalization((0.75, 1.5)),
        # 高斯噪声
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5),
        # 给每个像素乘上0.8-1.2之间的数来使图片变暗或变亮
        # 20%的图片在每个channel上乘以不同的因子
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # 对每张图片进行仿射变换，包括缩放、平移、旋转、修剪
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)  # 随机应用以上的图片增强方法

    N = len(Imgs)
    # N1 = len(Imgs1)

    index = np.arange(N)
    # index1 = np.arange(N1)

    while True:
        np.random.shuffle(index)
        for i in range(0, N, batch_size):
            if i+batch_size >= N:
                break
            rImgs = []
            rLabel = []
            for j in range(i, i+batch_size):
                # print(Imgs[index[j]], Labels[index[j]])
                try:
                    # img = cv2.imread(Imgs[index[j]])
                    img = cv2.imdecode(np.fromfile(Imgs[index[j]], dtype=np.uint8), cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (256, 256)).astype(np.float32)
                    img = precessFunc(img)
                    # MIN, MAX = img.min(), img.max()
                    # img = (img - MIN)/(MAX - MIN)
                    # img = keras.preprocessing.image.load_img(Imgs[index[j]])
                    # img = keras.preprocessing.image.img_to_array(img)
                    # img = img.reshape((1, )+img.shape)
                except:
                    continue

                rImgs.append(img)
                l = np.zeros(shape=[5])
                l[Labels[index[j]]] = 1
                rLabel.append(l)
            rImgs = np.array(rImgs)
            rLabel = np.array(rLabel)
            # it = datagen.flow(rImgs, batch_size=8)
            # img = it.next()

            rImgs = seq.augment_images(rImgs)

            yield rImgs, rLabel


def getImgs():
    Imgs, Labels = [], []
    Imgs1, Labels1 = [], []
    for idx, sick in enumerate(types):
        files = os.listdir(os.path.join('./split_data/train', sick))
        # print(files)
        # print(len(files))
        for file in files:
            Imgs.append(os.path.join('./split_data/train', sick, file))
            Labels.append(idx)
    # return  Imgs,Labels

    for idx1, sick1 in enumerate(types):
        files1 = os.listdir(os.path.join('./split_data/test', sick1))
        # print(files1)
        # print(len(files1))
        for file1 in files1:
            Imgs1.append(os.path.join('./split_data/test', sick1, file1))
            Labels1.append(idx1)
    return Imgs, Labels,Imgs1,Labels1