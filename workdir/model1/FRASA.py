import cv2
import numpy as np
import torch
from scipy.spatial import ConvexHull
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import dlib
except ImportError:
    dlib = None

def create_default_landmarks(mean_landmarks_path):
    """
    从保存的平均关键点创建默认返回值
    :param mean_landmarks_path: 平均关键点npy文件路径
    :return: 与get_facial_keypoints未检测到人脸时相同格式的返回值
    """
    # 加载平均关键点
    mean_landmarks = np.load(mean_landmarks_path)
    
    # 转换为坐标字典格式
    facial_landmarks = {i: (int(x), int(y)) for i, (x, y) in enumerate(mean_landmarks)}
    
    # 创建五官区域划分
    facial_regions = {
        "left_eyebrow": [facial_landmarks[i] for i in range(17, 22)],   # 17-21
        "right_eyebrow": [facial_landmarks[i] for i in range(22, 27)],  # 22-26
        "left_eye": [facial_landmarks[i] for i in range(36, 42)],       # 36-41
        "right_eye": [facial_landmarks[i] for i in range(42, 48)],      # 42-47
        "nose": [facial_landmarks[i] for i in range(27, 36)],           # 27-35
        "mouth": [facial_landmarks[i] for i in range(48, 68)],          # 48-67
        "left_cheek": [facial_landmarks[i] for i in [1, 2, 3, 4]] + 
                     [facial_landmarks[i] for i in [31, 32]] + 
                     [facial_landmarks[48]],  
        "right_cheek": [facial_landmarks[i] for i in [12, 13, 14, 15]] + 
                      [facial_landmarks[i] for i in [34, 35]] + 
                      [facial_landmarks[54]]
    }
    
    # 获取区域名称列表（保持顺序）
    region_names = list(facial_regions.keys())
    
    # 找到最大点数
    max_points = max(len(region) for region in facial_regions.values())
    
    # 填充所有区域到相同长度
    padded_regions = []
    for region_points in facial_regions.values():
        padded = region_points + [(-100, -100)] * (max_points - len(region_points))
        padded_regions.append(padded)
    
    # 转换为张量 [区域数量, 最大点数, 2]
    regions_tensor = torch.tensor(padded_regions, dtype=torch.float32)
    
    return region_names, regions_tensor

def get_facial_keypoints(detector,predictor,apex_orig_single_image,default_return=None,failed_log_file=None,file_path=None):
    """
    读取人脸图像并提取 68 个人脸关键点坐标
    :param image_path: 图像路径
    :return: 关键点坐标字典 {区域名: [(x1, y1), (x2, y2), ...]}
    """
    def log_failure(image_path, reason):
        """记录失败情况到日志文件"""
        if failed_log_file:
            with open(failed_log_file, 'a') as f:
                f.write(f"{image_path}|{reason}\n")

    # 确保输入是 NumPy 数组并具有正确的形状
    if torch.is_tensor(apex_orig_single_image):
        # 检查维度并转换为 HWC 格式的 NumPy 数组
        if apex_orig_single_image.dim() == 3:
            # 如果是CHW格式 [3, H, W]
            if apex_orig_single_image.shape[0] == 3:
                # 转换为HWC格式 [H, W, 3]
                apex_orig_single_image = apex_orig_single_image.permute(1, 2, 0).cpu().numpy()
            # 如果是HWC格式 [H, W, 3]
            elif apex_orig_single_image.shape[2] == 3:
                apex_orig_single_image = apex_orig_single_image.cpu().numpy()
            else:
                raise ValueError(f"Expected tensor with shape [3,H,W] or [H,W,3], got {apex_orig_single_image.shape}")
        else:
            raise ValueError(f"Expected 3D tensor, got {apex_orig_single_image.dim()}D tensor")
    
    # 确保数据类型是 uint8
    if apex_orig_single_image.dtype != np.uint8:
        if apex_orig_single_image.max() <= 1.0:  # 假设是 [0,1] 范围的 float
            apex_orig_single_image = (apex_orig_single_image * 255).astype(np.uint8)
        else:
            apex_orig_single_image = apex_orig_single_image.astype(np.uint8)

    # 保存原始尺寸
    orig_height, orig_width = apex_orig_single_image.shape[:2]    

    gray_apex = cv2.cvtColor(apex_orig_single_image, cv2.COLOR_BGR2GRAY)

    # debug_path = "debug_gray.jpg"  # Linux临时目录
    # cv2.imwrite(debug_path, gray)
    # print(f"Debug image saved to: {debug_path}")

    # 检测人脸
    faces = detector(gray_apex)
    if len(faces) == 0:
        # 尝试不同的检测参数
        faces = detector(gray_apex, 1)  # 尝试upsample=1
        
        if len(faces) == 0:
            # print(f"No faces detected in apex frame. Using default landmarks.")
            if failed_log_file and file_path:
                log_failure(file_path, "apex frame failed")
            return default_return
    rect=faces[0]

    shape = predictor(gray_apex, rect)

    # orig_height, orig_width = gray_apex.shape[:2]
    orig_width=28
    orig_height=28

    # 计算缩放比例（强制拉伸的缩放因子）
    scale_x = orig_width / 224
    scale_y = orig_height / 224

    # 生成坐标字典，收集关键点并映射到224x224尺寸并确保坐标在图像范围内
    facial_landmarks = {}
    for i in range(68):
        # 原始坐标
        x_orig = shape.part(i).x
        y_orig = shape.part(i).y
        # 映射到224x224后的坐标
        x_scaled = int(x_orig * scale_x)
        y_scaled = int(y_orig * scale_y)
        # 确保坐标在合法范围内
        x_scaled = max(0, min(x_scaled, 27))
        y_scaled = max(0, min(y_scaled, 27))
        facial_landmarks[i] = (x_scaled, y_scaled)

    # 五官区域划分
    facial_regions = {
        "left_eyebrow": [facial_landmarks[i] for i in range(17, 22)],   # 17-21
        "right_eyebrow": [facial_landmarks[i] for i in range(22, 27)],  # 22-26
        "left_eye": [facial_landmarks[i] for i in range(36, 42)],       # 36-41
        "right_eye": [facial_landmarks[i] for i in range(42, 48)],      # 42-47
        "nose": [facial_landmarks[i] for i in range(27, 36)],           # 27-35
        "mouth": [facial_landmarks[i] for i in range(48, 68)],          # 48-67
        # 扩展左脸颊
        "left_cheek": [facial_landmarks[i] for i in [1, 2, 3, 4]] + [facial_landmarks[i] for i in [31, 32]] + [facial_landmarks[48]],  
        # 扩展右脸颊
        "right_cheek": [facial_landmarks[i] for i in [12, 13, 14, 15]] + [facial_landmarks[i] for i in [34, 35]] + [facial_landmarks[54]]
    }
    # 获取区域名称列表（保持顺序）
    region_names = list(facial_regions.keys())
    
    # 找到最大点数
    max_points = max(len(region) for region in facial_regions.values())
    
    # 填充所有区域到相同长度
    padded_regions = []
    for region_points in facial_regions.values():
        padded = region_points + [(-100, -100)] * (max_points - len(region_points))
        padded_regions.append(padded)
    
    # 转换为张量 [区域数量, 最大点数, 2]
    regions_tensor = torch.tensor(padded_regions, dtype=torch.float32)
    
    return region_names, regions_tensor


# 五官区域划分矩阵
def construct_component_matrix(region_names, regions_tensor, H, W, patch_size):
    """
    构建五官区域划分矩阵 A_Component
    :param region_names: 区域名称列表
    :param regions_tensor: 关键点坐标张量 [区域数量, 最大点数, 2]
    :param H, W: 图像尺寸
    :param patch_size: Patch 大小
    :return: A_Component (M_C, N)
    """

    N_h, N_w = H // patch_size, W // patch_size
    N = N_h * N_w
    M_C = len(region_names)
    A_Component = np.zeros((M_C, N))

    # 将张量转换为numpy数组
    regions_np = regions_tensor.numpy()
    
    for i, region_name in enumerate(region_names):
        region_points = regions_np[i]
        # 移除填充的(0,0)点
        valid_points = [p for p in region_points if not (p[0] == -100 and p[1] == -100)]
        region_points = np.array(valid_points)
        
        if "eyebrow" in region_name:  # 眉毛（线性）
            for x, y in region_points:
                patch_x, patch_y = int(x // patch_size), int(y // patch_size)
                patch_idx = patch_y * N_w + patch_x
                A_Component[i, patch_idx] = 1
        else:  # 其他封闭区域
            hull = ConvexHull(region_points)
            hull_points = region_points[hull.vertices]
            
            for py in range(N_h):
                for px in range(N_w):
                    patch_center_x = (px + 0.5) * patch_size
                    patch_center_y = (py + 0.5) * patch_size
                    
                    if cv2.pointPolygonTest(hull_points, (patch_center_x, patch_center_y), False) >= 0:
                        patch_idx = py * N_w + px
                        A_Component[i, patch_idx] = 1

    return torch.tensor(A_Component, dtype=torch.float32)


def construct_region_matrix(H, W, patch_size):
    """
    生成基于图片长宽对半划分的面部区域矩阵 A_Region
    :param H: 图像高度
    :param W: 图像宽度
    :param P: Patch 大小
    :return: A_Region (4 × N) 矩阵，其中 4 是面部区域（左上、右上、左下、右下），N 是 Patch 数量
    """
    N_h, N_w = H // patch_size, W // patch_size  # Patch 网格数量
    N = N_h * N_w  # Patch 总数
    M_R = 4  # 面部区域数目（左上、右上、左下、右下）

    # 计算面部中心线
    mid_w = W // 2
    mid_h = H // 2

    # 初始化 A_Region 矩阵
    A_Region = np.zeros((M_R, N))

    # 遍历每个 Patch，确定其属于哪个面部区域
    for py in range(N_h):
        for px in range(N_w):
            patch_center_x = (px + 0.5) * patch_size  # Patch 中心 x 坐标
            patch_center_y = (py + 0.5) * patch_size  # Patch 中心 y 坐标
            patch_idx = py * N_w + px        # Patch 索引

            # 判断区域归属
            if patch_center_x < mid_w and patch_center_y < mid_h:
                A_Region[0, patch_idx] = 1  # 左上 (LU)
            elif patch_center_x >= mid_w and patch_center_y < mid_h:
                A_Region[1, patch_idx] = 1  # 右上 (RU)
            elif patch_center_x < mid_w and patch_center_y >= mid_h:
                A_Region[2, patch_idx] = 1  # 左下 (LD)
            else:
                A_Region[3, patch_idx] = 1  # 右下 (RD)

    return torch.tensor(A_Region,dtype=torch.float32)

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # 生成 A_Component 矩阵
# H, W, patch_size = 224, 224, 16
# image = cv2.imread("example.jpg")
# #图像缩放到224x224
# image_resized = cv2.resize(image, (224, 224))
# # 运行关键点检测
# region_names, regions_tensor = get_facial_keypoints(detector,predictor,image)
# A_Component = construct_component_matrix(region_names, regions_tensor, H, W, patch_size)
# print(A_Component.shape)  # (8, N)
# # 生成 A_Region 矩阵
# A_Region = construct_region_matrix(H,W,patch_size)
# print(A_Region.shape)  # (4, N)


class FRASA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,M_R=4, N=196):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.w_c = nn.Linear(dim, dim)  # Component attention transformation
        self.w_r = nn.Linear(dim, dim)  # Region attention transformation
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # 可训练参数 A_init
        # self.A_init = nn.Parameter(torch.zeros((M_R, N)))  # 初始化为 0，训练时优化

    def forward(self, x, A_Component, A_Region):
        # print('FRASA')
        # print('x.shape:',x.shape)  #torch.Size([32, 196, 768])
        # print('A_Component.shape:',A_Component.shape)  #torch.Size([32, 8, 196])
        # print('A_Region.shape:',A_Region.shape)  #torch.Size([32, 4, 196])
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Q, K, V shape: (B, num_heads, N, C // num_heads)
        # print('q.shape:',q.shape) #torch.Size([32, 4, 196, 192])
        # print('k.shape:',k.shape)
        # print('v.shape:',v.shape)
        
        if A_Component is not None:
            F_Component = torch.matmul(A_Component.transpose(1, 2), torch.matmul(A_Component, x))  
        else: 
            F_Component=x.clone()

        Component_K = self.w_c(F_Component)  
        # print('Component_K.shape:',Component_K.shape) #torch.Size([32, 196, 768])
        # 将 Component_K [32, 196, 768] 拆分成 4 heads [32, 4, 196, 192]
        head_dim_C=Component_K.size(-1) //self.num_heads
        Component_K = Component_K.view(Component_K.size(0), Component_K.size(1), self.num_heads, head_dim_C).transpose(1, 2)  # [32, 4, 196, 192]

        if A_Region is not None:
            # Compute A_(A_Region) and F_Region
            # A_A_Region = A_Region + self.A_init  # Adaptive region matrix
            F_Region = torch.matmul(A_Region.transpose(1, 2), torch.matmul(A_Region, x)) 
        else: 
            F_Region=x.clone()
               
        Region_K = self.w_r(F_Region)  
        # print('Region_K.shape:',Region_K.shape) #torch.Size([32, 196, 768])
        head_dim_K=Region_K.size(-1) //self.num_heads
        Region_K =Region_K.view(Region_K.size(0), Region_K.size(1), self.num_heads, head_dim_K).transpose(1, 2)

        # print('Component_K',Component_K)
        # print('Region_K',Region_K)


        # 计算 FRASA 注意力
        attn = (q @ k.transpose(-2, -1) +  # 基础 Patch 层
                q @ Component_K.transpose(-2, -1) +  # 五官区域层
                q @ Region_K.transpose(-2, -1)) * self.scale  # 面部区域层

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


