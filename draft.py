import cv2
import numpy as np

def purify_colors_with_trimap(image_path, image_rmbg_path, trimap_path):
    # 读取图像和trimap
    image = cv2.imread(image_path)
    trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)
    image_rmbg = cv2.imread(image_rmbg_path,cv2.IMREAD_UNCHANGED)

    # 将trimap转换为0, 0.5, 1
    trimap = trimap.astype(np.float32) / 255

    # 创建滑窗核
    kernel_radius = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kernel_radius+1, 2*kernel_radius+1))

    # 提取0和0.5之间的边界（背景和不确定区域之间）
    background_boundary = cv2.dilate(trimap, kernel, iterations=1) * (1 - trimap)

    # 提取0.5和1之间的边界（不确定区域和前景之间）
    foreground_boundary = cv2.dilate(1 - trimap, kernel, iterations=1) * trimap

    # 估计背景颜色
    background_color = np.mean(image[background_boundary > 0], axis=0)

    # 净化颜色：在0.5和1的边界上消除背景颜色分量
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if foreground_boundary[y, x] > 0:
                # 减去估计的背景颜色，并进行裁剪以确保值在合法范围内
                image[y, x] = np.clip(image[y, x] - background_color, 0, 255)

    return image

# 路径
image_path = '/data/code-deploy/ZZMatte/testimages/cleansecolor/2024-03-12-11-39-15-663838-4-uploaded_sambg_image.png'
image_rmbg_path = "image_path = '/data/code-deploy/ZZMatte/testimages/cleansecolor/2024-03-12-11-39-15-663838-4-removed_bg_image.png'"
trimap_path = '/data/code-deploy/ZZMatte/testimages/cleansecolor/trimap_unresized.png'

# 处理图像
purified_image = purify_colors_with_trimap(image_path, image_rmbg_path, trimap_path)

# # 显示和保存结果
# cv2.imshow('Purified Image', purified_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('purified_image_with_trimap.jpg', purified_image)
