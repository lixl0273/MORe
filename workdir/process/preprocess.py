from typing import Union
import numpy as np
import pandas as pd
import cv2
import copy


def TVL1_optical_flow(prev_frame: np.array, next_frame: np.array):
    """Compute the TV-L1 optical flow and normalized the result"""
    # Transform the image from BGR to Gray
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Create TV-L1 optical flow
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create(scaleStep=0.5)
    flow = optical_flow.calc(prev_frame, next_frame, None)

    return flow


def TVL1_magnitude(flow: np.array):
    """Compute the magnitude of the frame"""
    flow = copy.deepcopy(flow)
    mag = np.sqrt(np.sum(flow ** 2, axis=-1))
    mag = normalized_channel(mag)

    return mag


def normalized_channel(frame):
    min_value = np.amin(frame, axis=(0, 1), keepdims=True)
    max_value = np.amax(frame, axis=(0, 1), keepdims=True)

    frame = (frame - min_value) / (max_value - min_value + 1e-8) * 255
    frame = np.minimum(frame, 255)
    frame = np.maximum(frame, 0)
    return frame.astype("uint8")

def normalized(frame: np.array,
               g_min: float = -128,
               g_max: float = 128,
               lambda_: int = 16):
    # Do the normalization
    if len(frame.shape) > 2:
        f_min = np.amin(frame, axis=(0, 1))
        f_max = np.amax(frame, axis=(0, 1))
    else:
        f_min = np.min(frame)
        f_max = np.max(frame)

    frame = lambda_ * (frame - f_min) * (g_max - g_min) / (f_max - f_min + 1e-8) + g_min

    return frame


def optical_strain(flow: np.array) -> np.array:
    """Compute the optical strain for the given u, v
    Refer to: https://github.com/mariaoliverparera/mod-opticalStrain/blob/master/get_contours.py

    Parameters
    ----------
    flow : np.array
        Normalized horizontal and vertical optical flow fields

    Returns
    -------
    np.array
        Return the optical strain magnitude for u, v
    """
    flow = copy.deepcopy(normalized(flow))
    u = flow[..., 0]
    v = flow[..., 1]

    # Compute the gradient
    u_x = u - np.roll(u, 1, axis=1)
    u_y = u - np.roll(u, 1, axis=0)
    v_x = v - np.roll(v, 1, axis=1)
    v_y = v - np.roll(v, 1, axis=0)

    e_xy = 0.5 * (u_y + v_x)

    e_mag = np.sqrt(u_x**2 + 2 * (e_xy**2) + v_y**2)
    # e_mag = normalized_channel(e_mag)
    e_mag = minmax_norm(e_mag)*255

    return e_mag


def compute_features(onset_frame: np.array, apex_frame: np.array):
    # 1. 光流计算
    flow = TVL1_optical_flow(prev_frame=onset_frame, next_frame=apex_frame)

    # 2. 拆分光流 u 和 v，并归一化
    # u_flow = normalized_channel(flow[..., 0])
    # v_flow = normalized_channel(flow[..., 1])
    u_flow = (minmax_norm(flow[..., 0]) * 255).astype(np.uint8)
    v_flow = (minmax_norm(flow[..., 1]) * 255).astype(np.uint8)

    # 3. 计算应变张量
    strain_mag = optical_strain(flow)  # shape: (H, W), already uint8

    # 4. 拼接为三通道图像：(H, W, 3)
    combined = np.concatenate((
        u_flow[..., np.newaxis],      # 通道 0：水平光流
        v_flow[..., np.newaxis],      # 通道 1：垂直光流
        strain_mag[..., np.newaxis],  # 通道 2：光流应变
    ), axis=2)

    return combined.astype(np.uint8)

import cv2
import numpy as np

def tvl1_ofcalc(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flow = cv2.optflow.DualTVL1OpticalFlow_create()
    
    of = flow.calc(img1, img2, None)
    return of

def minmax_norm(x):
    x_flat = x.reshape(-1)
    x_max = np.max(x)
    x_min = np.min(x)
    if x_max == x_min:
        x_flat *= 0
    else:
        x_flat = (x_flat - x_min)/(x_max - x_min)
    return x_flat.reshape(x.shape)

def calc_os_flow(path1, path2):
    flow = tvl1_ofcalc(path1, path2)
    u_flow = minmax_norm(flow[:, :, 0])*255
    v_flow = minmax_norm(flow[:, :, 1])*255
    
    ux, uy = np.gradient(flow[:, :, 0])
    vx, vy = np.gradient(flow[:, :, 1])
    
    os_flow = np.sqrt(ux ** 2 + vy ** 2 + 0.25 * (uy + vx) ** 2)
    os_flow = minmax_norm(os_flow)*255
    
    # return np.concatenate((os_flow.reshape(*os_flow.shape, 1), v_flow.reshape(*v_flow.shape, 1), u_flow.reshape(*u_flow.shape, 1)), axis=2)
    return np.concatenate((u_flow.reshape(*u_flow.shape, 1), 
                      v_flow.reshape(*v_flow.shape, 1), 
                      os_flow.reshape(*os_flow.shape, 1)), axis=2)

