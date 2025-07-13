# TODO 用ctypes调用c代码进行验证对比

# import ctypes
# import numpy as np
# import pinocchio as pin

# # 加载 C 库
# lib = ctypes.CDLL('./spinor.so')
# lib.spatial_rotate_z.argtypes = [ctypes.c_double, np.ctypeslib.ndpointer(dtype=np.float64, shape=(6, 6))]

# # 调用 C 函数
# c_result = np.zeros((6, 6), dtype=np.float64)
# lib.spatial_rotate_z(np.pi / 4, c_result)  # 45°

# # Pinocchio 计算
# model = pin.Model()
# joint = pin.JointModelRX()  # 示例关节
# data = model.createData()
# pin.spatialRotation(joint.rotation())  # 对比旋量

# print("C 计算结果:\n", c_result)
# print("Pinocchio 结果:\n", data.oMi[1].rotation())  # 假设第一个关节
###################################################################################

# 物体初始在Pw=[0,0,0]T
# 运动指令：1、绕固定坐标系是z轴旋转30度
#          2、沿固定坐标系的x轴移动+3米

# 初始R:  I

# 过程：绕z轴R是
# [ cos  -sin 0 ]
# [ sin   cos 0 ]
# [ 0     0   1 ]

# 移动： [3 0 0]

import numpy as np
from math import cos, sin, radians
import pinocchio as pin

def pin_by_fixed_frame_test():
    M_init = pin.SE3.Identity()
    R_z = pin.AngleAxis(radians(30), np.array([0., 0., 1.])).toRotationMatrix()
    M_rot = pin.SE3(R_z, np.zeros(3))

    M = M_rot * M_init

    M_trans = pin.SE3(np.eye(3), np.array([3., 0., 0.]))
    M = M_trans * M

    p_w = M.translation
    R_wb = M.rotation

    print("最终位置  p_w:", p_w)
    print("旋转矩阵  R_wb:\n", R_wb)

def move_by_fixed_frame_test():
    p_init = np.array([0, 0, 0])
    R_init = np.eye(3)

    theta = radians(30)
    R_z = np.array([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0,          0,          1]
    ])
    R_new = R_z @ R_init

    delta_p = np.array([3, 0, 0])
    p_new = p_init + delta_p

    print("最终位置p_new:", p_new)
    print("最终旋转矩阵 R_new:\n", R_new)

# 绕问题坐标系运动
# 物体初始在Pw=[0,0,0]T
# 运动指令：1、绕物体坐标系是z轴旋转45度
#          2、沿物体坐标系的x轴移动+2米

# 旋转： [ cos  -sin 0 ]
# [ sin   cos 0 ]
# [ 0     0   1 ]

# 移动： [2 0 0]  在新的物体左边系移动   求的是相对固定坐标系的

def move_by_body_frame_test():
    p_init = np.array([0, 0, 0])
    R_init = np.eye(3)

    theta = radians(45)
    R_z = np.array([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0,          0,          1]
    ])
    R_new = R_init @ R_z

    delta_p = np.array([2, 0, 0])
    p_new = R_new @ delta_p # 旋转后，再移动，需要乘以旋转矩阵 p老s = r老s新s*p新s  消元 消除掉末尾的

    print("最终位置p_new:", p_new)
    print("最终旋转矩阵 R_new:\n", R_new)

def pin_by_body_frame_test():
    M_init = pin.SE3.Identity()
    R_z = pin.AngleAxis(radians(45), np.array([0., 0., 1.])).toRotationMatrix()
    M_rot = pin.SE3(R_z, np.zeros(3))

    M = M_init * M_rot

    M_trans = pin.SE3(np.eye(3), np.array([2., 0., 0.]))
    M = M * M_trans

    p_w = M.translation
    R_w = M.rotation

    print("最终位置p_w:", p_w)
    print("最终旋转矩阵 R_w:\n", R_w)


if __name__ == "__main__":
    move_by_fixed_frame_test()
    pin_by_fixed_frame_test()
    move_by_body_frame_test()
    pin_by_body_frame_test()
