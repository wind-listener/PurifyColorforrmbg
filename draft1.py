import cv2
import numpy as np

def process_image(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[2] != 4:
        raise ValueError("图像必须是RGBA通道格式")

    # 获取A通道，并进行二值分割
    alpha_channel = image[:, :, 3]
    _, binary_mask = cv2.threshold(alpha_channel, int(0.05 * 255), 255, cv2.THRESH_BINARY)

    # 执行闭运算以连通区域
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('/data/code-deploy/ZZMatte/testimages/cleansecolor/closed_mask.png', closed_mask)
    # 计算分割边界
    edges = cv2.Canny(closed_mask, 100, 200)
    cv2.imwrite('/data/code-deploy/ZZMatte/testimages/cleansecolor/edges.png', edges)
    # 使用滑窗提取颜色分量
    kernel_radius = 10
    for y in range(kernel_radius, image.shape[0] - kernel_radius):
        for x in range(kernel_radius, image.shape[1] - kernel_radius):
            if edges[y, x] == 255:  # 边界上的像素
                # 定义滑窗区域
                window = image[y-kernel_radius:y+kernel_radius+1, x-kernel_radius:x+kernel_radius+1]
                window_alpha = alpha_channel[y-kernel_radius:y+kernel_radius+1, x-kernel_radius:x+kernel_radius+1]

                # 在滑窗内提取0.5以下的区域颜色
                color_to_reduce = np.mean(window[window_alpha < int(0.5 * 255)], axis=0)

                # 在0.5以上的区域减少这个颜色分量
                above_half_alpha_indices = np.where(window_alpha >= int(0.5 * 255))
                for i, j in zip(*above_half_alpha_indices):
                    image[y-kernel_radius+i, x-kernel_radius+j, :3] = np.clip(image[y-kernel_radius+i, x-kernel_radius+j, :3] - color_to_reduce[:3], 0, 255)

    # 保存修改后的图像
    cv2.imwrite('/data/code-deploy/ZZMatte/testimages/cleansecolor/processed_image.png', image)

# 替换为实际的图像路径
image_path = '/data/code-deploy/ZZMatte/testimages/cleansecolor/2024-03-12-11-39-15-663838-4-removed_bg_image.png'
process_image(image_path)
