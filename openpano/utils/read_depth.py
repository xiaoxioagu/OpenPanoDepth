import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取深度图片
depth_image_path = 'E:\dataset\\2D3Ds\\area_1\pano\depth\camera_14c620a723e54e8cb6847b4a0c532dca_office_1_frame_equirectangular_domain_depth.png'
# depth_image_path = 'E:\dataset\2D3Ds\area_1\pano\depth'# 替换为你的深度图片路径
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
depth_image = cv2.resize(depth_image, (1024, 512), interpolation = cv2.INTER_AREA)
depth_image = depth_image/65535*128

# 检查图片是否成功读取
if depth_image is None:
    print("无法读取深度图片")
else:
    # 显示深度图片的值
    print("深度图片的值：")
    print(depth_image)

    # 将深度图片转换为8位图像以便显示
    depth_image_8bit = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 显示深度图片
    plt.figure(figsize=(10, 10))
    plt.imshow(depth_image_8bit)
    plt.title("深度图片")
    plt.colorbar()
    plt.show()

    # 如果你想使用OpenCV的窗口显示图片
    cv2.imshow('Depth Image', depth_image_8bit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
