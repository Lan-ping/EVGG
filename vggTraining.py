
import keras
import numpy as np
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import utils
import random
import shutil

def precessFunc(img):
    img =tensorflow.keras.applications.vgg16.preprocess_input(img)
    return img

if __name__ == '__main__':
    def makedir(new_dir):
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
    random.seed(1)  # 随机种子
    # 1.确定原图像数据集路径
    dataset_dir = os.path.join("C:/Users/kuan/Desktop/tomato/data")
    # 2.确定数据集划分后保存的路径
    split_dir = os.path.join("C:/Users/kuan/Desktop/tomato/split_data")
    train_dir = os.path.join(split_dir, "train")
    test_dir = os.path.join(split_dir, "test")
    # 3.确定将数据集划分为训练集，测试集的比例
    train_pct = 0.8
    test_pct = 0.2
    # 4.划分
    for root, dirs, files in os.walk(dataset_dir):
        for sub_dir in dirs:  # 遍历0，1，2，3，4文件夹
            imgs = os.listdir(os.path.join(root, sub_dir))  # 展示目标文件夹下所有的文件名
            imgs = list(filter(lambda x: x.endswith('.JPG'), imgs))  # 取到所有以.jpg结尾的文件
            random.shuffle(imgs)  # 乱序图片路径
            img_count = len(imgs)  # 计算图片数量

            train_point = int(img_count * train_pct)  # 0:train_pct
            for i in range(img_count):
                if i < train_point:  # 保存0-train_point的图片到训练集
                    out_dir = os.path.join(train_dir, sub_dir)

                else:  # 保存valid_point-结束的图片到测试集
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)  # 创建文件夹
                target_path = os.path.join(out_dir, imgs[i])  # 指定目标保存路径
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])  # 指定目标原图像路径
                shutil.copy(src_path, target_path)  # 复制图片

    trainImgs, trainLabels, valImgs, valLabels = utils.getImgs()
    N = len(trainImgs)
    N1 = len(valImgs)
    print("数据增强前训练集图像共：", N, "张")
    print("测试集数据增强前图像共：", N1, "张")

    trainGenerator = utils.generator(Imgs=trainImgs, Labels=trainLabels, precessFunc=precessFunc)
    valGenerator = utils.generator(Imgs=valImgs, Labels=valLabels, precessFunc=precessFunc)
    regl1 = 0.000
    regl2 = 0.001
    basemodel = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(256, 256, 3),
    )
    for layer in basemodel.layers:
        layer.trainable = False
    x = basemodel.output
    x = Flatten()(x)
    x = Dense(2048, activation='relu', kernel_regularizer=tensorflow.keras.regularizers.l1_l2(regl1, regl2))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu', kernel_regularizer=tensorflow.keras.regularizers.l1_l2(regl1, regl2))(x)
    x = Dropout(0.5)(x)
    x = Dense(5, activation='softmax')(x)
    model = Model(inputs=basemodel.input, outputs=x)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(1e-4),
        metrics=['accuracy']
    )
    tensorboard = tensorflow.keras.callbacks.TensorBoard(log_dir='./recordVgg')
    checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(filepath='./recordVgg/bestModel.hdf5', save_best_only=True,
                                                 save_weights_only=True)
    data = trainGenerator.__next__()
    p = model.predict(data[0])
    history=model.fit_generator(
        generator=trainGenerator,
        steps_per_epoch=200,
        epochs=200,
        validation_data=valGenerator,
        validation_steps=100,
        callbacks=[tensorboard, checkpoint]
    )

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('VGG Training and Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('VGG Training and Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.title('VGG Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('VGG Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()