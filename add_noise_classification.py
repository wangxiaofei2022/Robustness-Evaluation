#####对分类数据集（jpg格式图像分别在以类别命名的文件下）进行处理#####
import os
import cv2
import numpy as np
import random
from imgaug import augmenters as iaa


# /////////////// Corruptions ///////////////


#######################  Rain  #######################
# Generate noisy image
# value： control the number of raindrops
def get_noise_rain(img, value):
    noise = np.random.uniform(0, 256, img.shape[0:2])  # Generate random numbers
    # Control the noise level, take a floating point number, and only keep the largest part as noise
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    # Noise for initial blurring
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)

    # cv2.imshow('img',noise)
    # cv2.waitKey()
    # cv2.destroyWindow('img')
    return noise


# Add noise to motion blur to imitate raindrops
# noise： input Noise Plot，shape = img.shape[0:2]
# length: diagonal matrix size, representing the length of the raindrop
# angle： angle of inclination
# w: raindrop size
# Output: blurred noise
def rain_blur(noise, length=10, angle=5, w=3):
    # Diagonal array comes with 45 degree tilt. Add an error of -45 degrees to ensure that the start is positive
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # Generate focus matrix
    k = cv2.warpAffine(dig, trans, (length, length))  # Generate blur kernel
    k = cv2.GaussianBlur(k, (w, w), 0)  # Gaussian blurs this rotated diagonal kernel

    # k = k / length

    blurred = cv2.filter2D(noise, -1, k)

    # 0-255
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    # cv2.imshow('img',blurred)
    # cv2.waitKey()
    # cv2.destroyWindow('img')
    return blurred


def alpha_rain(rain, img, beta=0.8):
    # beta = 0.8   #results weight

    # expand dimensin
    # 2D --> 3D
    # 4 channels with alpha

    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel

    rain_result = img.copy()
    rain = np.array(rain, dtype=np.float32)
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]

    return rain_result
    # cv2.imshow('rain_effct_result', rain_result)
    #     # cv2.waitKey()
    #     # cv2.destroyAllWindows('rain_effct_result')


#######################  Random Noise  #######################
def get_noise_random(image, noise_num):
    # image，noise_num
    img_noise = image
    # cv2.imshow("src", img)
    rows, cols, chn = img_noise.shape
    # 加噪声
    for i in range(noise_num):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255
    return img_noise


#######################  Gaussian Noise  #######################
def get_noise_gaussian(img, mean, sigma):
    if img != '.DS_Store':
        # Grayscale normalization
        img = img / 255
        # Generate gaussian noise
        noise = np.random.normal(mean, sigma, img.shape)
        # add
        gaussian_out = img + noise
        gaussian_out = np.clip(gaussian_out, 0, 1)
        # 0-255
        gaussian_out = np.uint8(gaussian_out * 255)
        # 0-255
        # noise = np.uint8(noise*255)
        return gaussian_out


#######################  Poisson Noise  #######################
def get_noise_poisson(img, lam):
    # lam>=0 Expectations of Noise Appearance
    # The smaller the value, the less the noise frequency

    # 产生泊松噪声
    noise = np.random.poisson(lam, img.shape).astype(dtype='uint8')
    # 噪声和图片叠加
    poisson_out = img + noise
    return poisson_out


#######################  SP Noise  #######################
def get_noise_sp(noise_img, proportion):
    height, width = noise_img.shape[0], noise_img.shape[1]
    num = int(height * width * proportion)
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img


#######################  Fog  #######################
def get_noise_fog(input_dir, output_dir, severity):
    input_dir = input_dir + "/" ###报错AttributeError: 'NoneType' object has no attribute 'ndim'  对图像加雪、雾时路径应加上“/”
    output_dir = output_dir + "/"
    seq = iaa.Sequential([
        iaa.imgcorruptlike.Fog(severity=severity)])
    imglist = []
    filelist = os.listdir(input_dir)
    for item in filelist:
        img = cv2.imread(input_dir + item)
        imglist.append(img)
    for count in range(1):
        images_aug = seq.augment_images(imglist)
        for index in range(len(images_aug)):
            # filename = str(count) + str(index) + '.jpg'

            filename = str(filelist[index])
            cv2.imwrite(output_dir + filename, images_aug[index])


#######################  Snow  #######################
def get_noise_snow(input_dir, output_dir, severity):
    input_dir = input_dir + "/" ###报错AttributeError: 'NoneType' object has no attribute 'ndim'  对图像加雪、雾时路径应加上“/”
    output_dir = output_dir + "/"
    snow_dict = {1: (0.3, 0.4), 2: (0.37, 0.47), 3: (0.44, 0.54), 4: (0.51, 0.61), 5: (0.58, 0.68)}
    flake_size = snow_dict.get(severity)
    seq = iaa.Sequential([
        iaa.Snowflakes(flake_size=flake_size, speed=(0.007, 0.03))])
    imglist = []
    filelist = os.listdir(input_dir)
    for item in filelist:
        img = cv2.imread(input_dir + item)
        imglist.append(img)
    for count in range(1):
        images_aug = seq.augment_images(imglist)
        for index in range(len(images_aug)):
            # filename = str(count) + str(index) + '.jpg'

            filename = str(filelist[index])
            cv2.imwrite(output_dir + filename, images_aug[index])


def convert(input_dir, output_dir, model: 'str', severity):
    for filename in os.listdir(input_dir):
        path = input_dir + "/" + filename
        print("doing... ", path)
        noise_img = cv2.imread(path)
        if model == 'rain':
            rain_dict = {1: 100, 2: 500, 3: 1000, 4: 3000, 5: 6000}
            value = rain_dict.get(severity)
            img_noise = get_noise_rain(noise_img, value)
            rain = rain_blur(img_noise, length=10, angle=5, w=3)
            img_noise = alpha_rain(rain, noise_img, beta=0.8)
            cv2.imwrite(output_dir + '/' + filename, img_noise)

        elif model == 'gaussian':
            gaussian_dict = {1: 0.020, 2: 0.050, 3: 0.100, 4: 0.150, 5: 0.200}
            value = gaussian_dict.get(severity)
            img_noise = get_noise_gaussian(noise_img, 0, value)
            cv2.imwrite(output_dir + '/' + filename, img_noise)

        elif model == 'poisson':
            poisson_dict = {1: 0.5, 2: 10, 3: 50, 4: 70, 5: 100}
            value = poisson_dict.get(severity)
            img_noise = get_noise_poisson(noise_img, value)
            cv2.imwrite(output_dir + '/' + filename, img_noise)

        elif model == 'sp':
            sp_dict = {1: 0.003, 2: 0.006, 3: 0.015, 4: 0.050, 5: 0.100}
            value = sp_dict.get(severity)
            img_noise = get_noise_sp(noise_img, value)
            cv2.imwrite(output_dir + '/' + filename, img_noise)

        elif model == 'random':
            random_dict = {1: 1000, 2: 10000, 3: 20000, 4: 50000, 5: 100000}
            value = random_dict.get(severity)
            img_noise = get_noise_random(noise_img, value)
            cv2.imwrite(output_dir + '/' + filename, img_noise)

        elif model == 'fog':
            get_noise_fog(input_dir, output_dir, severity)

        elif model == 'snow':
            get_noise_snow(input_dir, output_dir, severity)

        else:
            print('Model input error, please re-enter')



if __name__ == '__main__':
    aid_path = "/xxx/xxx/xxx/xxx/xxx/test"
    test_path = "/xxx/xxx/xxx/xxx/xxx/test_fog1"

    # 遍历test文件夹中的30个子文件夹
    for subdir in os.listdir(aid_path):
        # 创建输出路径，与test文件夹中的子文件夹相对应
        output_subdir = os.path.join(test_path, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        input_dir = os.path.join(aid_path, subdir)
        output_dir = os.path.join(output_subdir)
        convert(input_dir, output_dir, model='fog', severity=1)
