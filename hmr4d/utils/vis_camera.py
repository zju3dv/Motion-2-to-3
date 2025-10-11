import matplotlib.pyplot as plt
import numpy as np
def vis_camera_list(T_list, R_list):
    # 创建一个新的图形
    fig = plt.figure()
    # 添加一个3D子图
    ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    print("T_list")
    print(T_list)
    #print("R_list")
    #print(R_list)
    for T, R in zip(T_list, R_list):
        # 计算相机的方向
        direction = R @ np.array([0, 0, 1])
        print("T:")
        # 绘制相机位置
        print(T)
        ax.scatter(*T, color='b')

        # 绘制相机方向
        ax.quiver(*T, *direction, color='r')

    # 设置图形的标题和坐标轴标签
    ax.set_title('Camera')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # 显示图形
    plt.show()

#T_list = [np.array([0, 0, 0]), np.array([1, 1, 1])]
#R_list = [np.eye(3), np.eye(3)]
#vis_camera_list(T_list, R_list)