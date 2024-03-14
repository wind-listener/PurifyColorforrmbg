
import cv2
import numpy as np

def adjust_foreground_color(image, alpha_channel):
    # 将RGB图像转换到HSV色彩空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 获取背景色和前景色的平均值
    background_color_hsv = np.mean(hsv_image[alpha_channel < 128], axis=0)
    foreground_color_hsv = np.mean(hsv_image[alpha_channel >= 128], axis=0)

    # 计算背景色和前景色之间的差异
    color_diff = foreground_color_hsv - background_color_hsv

    # 调整HSV图像中的前景色
    # 这里简单地减少饱和度，实际应用中可能需要更复杂的调整
    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            if alpha_channel[i, j] >= 128:  # 前景区域
                # 示例：减少饱和度
                hsv_image[i, j, 1] = max(0, hsv_image[i, j, 1] - color_diff[1] * 0.5)

    # 将HSV图像转换回RGB色彩空间
    adjusted_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    # 将调整后的RGB图像和原始的Alpha通道重新组合成RGBA图像
    adjusted_rgba = cv2.merge([adjusted_rgb[:, :, 0], adjusted_rgb[:, :, 1], adjusted_rgb[:, :, 2], alpha_channel])
    return adjusted_rgba

# 读取图像和alpha通道
image_path = '/data/code-deploy/ZZMatte/testimages/cleansecolor/2024-03-12-11-39-15-663838-4-removed_bg_image.png'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
alpha_channel = image[:, :, 3]

# 调整颜色
adjusted_image = adjust_foreground_color(image[:, :, :3], alpha_channel)

# 保存调整后的图像
cv2.imwrite('/data/code-deploy/ZZMatte/testimages/cleansecolor/adjusted_image.png', adjusted_image)
