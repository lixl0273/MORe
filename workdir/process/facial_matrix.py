import os
import cv2
import numpy as np
import torch
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import hashlib
try:
    import dlib
except ImportError:
    dlib = None
import argparse
import pandas as pd
from typing import List, Dict, Tuple
import traceback

class FacialMatrixPrecomputer:
    """面部矩阵预计算器"""
    
    def __init__(self, 
                 dlib_shape_predictor: str = 'shape_predictor_68_face_landmarks.dat',
                 target_size: int = 224,
                 patch_size: int = 16,
                 mean_landmarks_path: str = 'mean_landmarks_224x224.npy'):
        """
        初始化预计算器
        
        Args:
            dlib_shape_predictor: dlib形状预测器路径
            target_size: 目标图像尺寸
            patch_size: patch大小
            mean_landmarks_path: 平均关键点文件路径
        """
        self.target_size = target_size
        self.patch_size = patch_size
        
        # 初始化dlib
        if dlib is None:
            raise ImportError(
                "dlib is required only for recomputing facial matrices. "
                "For training with existing precomputed_matrices, use PrecomputedMatrixLoader."
            )
        print("Initializing dlib...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dlib_shape_predictor)
        
        # 加载平均关键点作为默认值
        self.mean_landmarks = np.load(mean_landmarks_path)
        print(f"Loaded mean landmarks from {mean_landmarks_path}")
        
        # 计算默认矩阵
        self.default_A_Component, self.default_A_Region = self._create_default_matrices()
        
        # 存储缓存信息
        self.cache_dir = None
        self.index_data = {}
    
    def _create_default_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """使用平均关键点创建默认矩阵"""
        N_h, N_w = self.target_size // self.patch_size, self.target_size // self.patch_size
        N = N_h * N_w
        M_C = 8
        
        # 从平均关键点创建区域划分
        facial_regions = {
            "left_eyebrow": [self.mean_landmarks[i] for i in range(17, 22)],
            "right_eyebrow": [self.mean_landmarks[i] for i in range(22, 27)],
            "left_eye": [self.mean_landmarks[i] for i in range(36, 42)],
            "right_eye": [self.mean_landmarks[i] for i in range(42, 48)],
            "nose": [self.mean_landmarks[i] for i in range(27, 36)],
            "mouth": [self.mean_landmarks[i] for i in range(48, 68)],
            "left_cheek": [self.mean_landmarks[i] for i in [1, 2, 3, 4, 31, 32, 48]],
            "right_cheek": [self.mean_landmarks[i] for i in [12, 13, 14, 15, 34, 35, 54]]
        }
        
        # 计算默认A_Component矩阵
        default_A_Component = np.zeros((M_C, N), dtype=np.float32)
        from scipy.spatial import ConvexHull
        
        region_names = list(facial_regions.keys())
        for r_idx, (region_name, region_points) in enumerate(zip(region_names, facial_regions.values())):
            points_array = np.array(region_points)
            
            if "eyebrow" in region_name or len(points_array) < 3:
                # 对于眉毛或点数不足的区域，使用点覆盖
                for x, y in region_points:
                    patch_x = int(x // self.patch_size)
                    patch_y = int(y // self.patch_size)
                    if 0 <= patch_x < N_w and 0 <= patch_y < N_h:
                        patch_idx = patch_y * N_w + patch_x
                        default_A_Component[r_idx, patch_idx] = 1.0
            else:
                # 使用凸包
                try:
                    hull = ConvexHull(points_array)
                    hull_points = points_array[hull.vertices]
                    
                    # 检查每个patch中心是否在凸包内
                    for py in range(N_h):
                        for px in range(N_w):
                            patch_center_x = (px + 0.5) * self.patch_size
                            patch_center_y = (py + 0.5) * self.patch_size
                            
                            # 判断点是否在凸包内
                            if self._is_point_in_polygon((patch_center_x, patch_center_y), hull_points):
                                patch_idx = py * N_w + px
                                default_A_Component[r_idx, patch_idx] = 1.0
                except:
                    # 凸包失败，使用点覆盖
                    for x, y in region_points:
                        patch_x = int(x // self.patch_size)
                        patch_y = int(y // self.patch_size)
                        if 0 <= patch_x < N_w and 0 <= patch_y < N_h:
                            patch_idx = patch_y * N_w + patch_x
                            default_A_Component[r_idx, patch_idx] = 1.0
        
        # 计算默认A_Region矩阵
        M_R = 4
        default_A_Region = np.zeros((M_R, N), dtype=np.float32)
        mid_w = self.target_size // 2
        mid_h = self.target_size // 2
        
        for py in range(N_h):
            for px in range(N_w):
                patch_center_x = (px + 0.5) * self.patch_size
                patch_center_y = (py + 0.5) * self.patch_size
                patch_idx = py * N_w + px
                
                if patch_center_x < mid_w and patch_center_y < mid_h:
                    default_A_Region[0, patch_idx] = 1.0  # 左上
                elif patch_center_x >= mid_w and patch_center_y < mid_h:
                    default_A_Region[1, patch_idx] = 1.0  # 右上
                elif patch_center_x < mid_w and patch_center_y >= mid_h:
                    default_A_Region[2, patch_idx] = 1.0  # 左下
                else:
                    default_A_Region[3, patch_idx] = 1.0  # 右下
        
        return default_A_Component, default_A_Region
    
    def _is_point_in_polygon(self, point, polygon):
        """判断点是否在多边形内（射线法）"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _extract_frame_number(self, image_path: str) -> int:
        """从图像路径中提取帧号"""
        # 尝试从文件名中提取数字，例如 "img46.jpg" -> 46
        filename = Path(image_path).stem
        # 移除非数字字符，提取数字
        import re
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[-1])
        return -1
    
    def _find_apex_frame_image(self, video_dir: str, apex_frame: int) -> str:
        """
        根据视频目录和apex帧号找到对应的图像文件
        
        Args:
            video_dir: 视频目录（包含多帧图像）
            apex_frame: apex帧号
            
        Returns:
            图像文件路径
        """
        video_path = Path(video_dir)
        
        # 首先尝试查找匹配帧号的图像
        possible_patterns = [
            f"*{apex_frame:03d}*",  # 如 "img046.jpg"
            f"*{apex_frame:02d}*",  # 如 "img46.jpg"
            f"*{apex_frame}*",      # 如 "img46.jpg"
            f"*reg_img{apex_frame}.jpg",  # 你的格式
            f"*{apex_frame}.jpg"
        ]
        
        for pattern in possible_patterns:
            matches = list(video_path.glob(pattern))
            if matches:
                # 选择第一个匹配的文件
                return str(matches[0])
        
        # 如果未找到，尝试读取目录下所有图像，通过文件名提取帧号
        all_images = list(video_path.glob("*.jpg")) + list(video_path.glob("*.png"))
        
        # 按帧号排序
        image_frame_pairs = []
        for img_path in all_images:
            frame_num = self._extract_frame_number(str(img_path))
            if frame_num >= 0:
                image_frame_pairs.append((frame_num, str(img_path)))
        
        if not image_frame_pairs:
            raise FileNotFoundError(f"No images found in {video_dir}")
        
        # 按帧号排序
        image_frame_pairs.sort(key=lambda x: x[0])
        
        # 找到最接近apex_frame的图像
        closest = min(image_frame_pairs, key=lambda x: abs(x[0] - apex_frame))
        print(f"Warning: Using closest frame {closest[0]} instead of apex frame {apex_frame}")
        return closest[1]
    
    def compute_matrices_for_image(self, image_path: str, use_default_on_fail: bool = True):
        """
        计算单个图像的矩阵
        
        Args:
            image_path: 图像路径
            use_default_on_fail: 检测失败时是否使用默认矩阵
            
        Returns:
            (success, A_Component, A_Region, image_hash)
        """
        try:
            # 计算图像hash
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
                image_hash = hashlib.md5(image_bytes).hexdigest()
            
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # 缩放到目标尺寸
            if image.shape[:2] != (self.target_size, self.target_size):
                image = cv2.resize(image, (self.target_size, self.target_size))
            
            # 获取关键点
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)
            
            if len(faces) == 0:
                faces = self.detector(gray, 1)  # 尝试upsample
            
            if len(faces) == 0:
                if use_default_on_fail:
                    print(f"⚠️ No face detected, using default: {image_path}")
                    return False, self.default_A_Component, self.default_A_Region, image_hash
                else:
                    raise ValueError("No face detected")
            
            rect = faces[0]
            shape = self.predictor(gray, rect)
            
            # 提取68个关键点
            landmarks = {}
            for i in range(68):
                x, y = shape.part(i).x, shape.part(i).y
                x = max(0, min(x, self.target_size-1))
                y = max(0, min(y, self.target_size-1))
                landmarks[i] = (x, y)
            
            # 计算A_Component矩阵
            N_h, N_w = self.target_size // self.patch_size, self.target_size // self.patch_size
            N = N_h * N_w
            M_C = 8
            A_Component = np.zeros((M_C, N), dtype=np.float32)
            
            from scipy.spatial import ConvexHull
            
            # 定义区域划分
            facial_regions = {
                "left_eyebrow": [landmarks[i] for i in range(17, 22)],
                "right_eyebrow": [landmarks[i] for i in range(22, 27)],
                "left_eye": [landmarks[i] for i in range(36, 42)],
                "right_eye": [landmarks[i] for i in range(42, 48)],
                "nose": [landmarks[i] for i in range(27, 36)],
                "mouth": [landmarks[i] for i in range(48, 68)],
                "left_cheek": [landmarks[i] for i in [1, 2, 3, 4, 31, 32, 48]],
                "right_cheek": [landmarks[i] for i in [12, 13, 14, 15, 34, 35, 54]]
            }
            
            region_names = list(facial_regions.keys())
            for r_idx, (region_name, region_points) in enumerate(zip(region_names, facial_regions.values())):
                points_array = np.array(region_points)
                
                if "eyebrow" in region_name or len(points_array) < 3:
                    # 对于眉毛或点数不足的区域，使用点覆盖
                    for x, y in region_points:
                        patch_x = int(x // self.patch_size)
                        patch_y = int(y // self.patch_size)
                        if 0 <= patch_x < N_w and 0 <= patch_y < N_h:
                            patch_idx = patch_y * N_w + patch_x
                            A_Component[r_idx, patch_idx] = 1.0
                else:
                    # 使用凸包
                    try:
                        hull = ConvexHull(points_array)
                        hull_points = points_array[hull.vertices]
                        
                        # 检查每个patch中心是否在凸包内
                        for py in range(N_h):
                            for px in range(N_w):
                                patch_center_x = (px + 0.5) * self.patch_size
                                patch_center_y = (py + 0.5) * self.patch_size
                                
                                if self._is_point_in_polygon((patch_center_x, patch_center_y), hull_points):
                                    patch_idx = py * N_w + px
                                    A_Component[r_idx, patch_idx] = 1.0
                    except:
                        # 凸包失败，使用点覆盖
                        for x, y in region_points:
                            patch_x = int(x // self.patch_size)
                            patch_y = int(y // self.patch_size)
                            if 0 <= patch_x < N_w and 0 <= patch_y < N_h:
                                patch_idx = patch_y * N_w + patch_x
                                A_Component[r_idx, patch_idx] = 1.0
            
            # 计算A_Region矩阵（固定计算）
            M_R = 4
            A_Region = np.zeros((M_R, N), dtype=np.float32)
            mid_w = self.target_size // 2
            mid_h = self.target_size // 2
            
            for py in range(N_h):
                for px in range(N_w):
                    patch_center_x = (px + 0.5) * self.patch_size
                    patch_center_y = (py + 0.5) * self.patch_size
                    patch_idx = py * N_w + px
                    
                    if patch_center_x < mid_w and patch_center_y < mid_h:
                        A_Region[0, patch_idx] = 1.0
                    elif patch_center_x >= mid_w and patch_center_y < mid_h:
                        A_Region[1, patch_idx] = 1.0
                    elif patch_center_x < mid_w and patch_center_y >= mid_h:
                        A_Region[2, patch_idx] = 1.0
                    else:
                        A_Region[3, patch_idx] = 1.0
            
            return True, A_Component, A_Region, image_hash
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
            if use_default_on_fail:
                return False, self.default_A_Component, self.default_A_Region, hashlib.md5(image_path.encode()).hexdigest()
            else:
                raise
    
    def precompute_from_csv(self, 
                           csv_path: str, 
                           output_dir: str,
                           image_base_dir: str = None):
        """
        从CSV文件预计算矩阵
        
        Args:
            csv_path: CSV文件路径
            output_dir: 输出目录
            image_base_dir: 图像基础目录（如果CSV中的路径是相对的）
        """
        # 创建输出目录
        self.cache_dir = Path(output_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取CSV文件
        print(f"Loading CSV from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records")
        
        # 检查必要的列
        # required_columns = ['Dataset', 'Subject', 'Filename', 'Label', 'ApexFrame']
        required_columns = ['Subject', 'Filename', 'Estimated Emotion', 'ApexFrame']

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV缺少必要列: {col}")
        
        # 五分类直接指定固定数据集名称
        # df['Dataset'] = 'casme2'
        # label_mapping = {
        # 'happiness': 0,
        # 'repression': 1,
        # 'disgust': 2,
        # 'surprise': 3,
        # 'others':4
        # }
            
        df['Dataset'] = 'samm'
        label_mapping = {
        'happiness': 0,
        'anger': 1,
        'contempt': 2,
        'surprise': 3,
        'other':4
        }
    
        # 统计各类别数量
        counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
        # 转换标签
        labels = []
        for emotion in df['Estimated Emotion']:
            emotion_lower = str(emotion).lower().strip()
            if emotion_lower in label_mapping:
                label = label_mapping[emotion_lower]
                counts[label] += 1
            labels.append(label)
        # 添加Label列
        df['Label'] = labels
        
        # 创建索引文件
        index_file = self.cache_dir / "matrix_index.json"
        self.index_data = {
            'metadata': {
                'csv_path': csv_path,
                'total_samples': len(df),
                'target_size': self.target_size,
                'patch_size': self.patch_size,
                'created_at': pd.Timestamp.now().isoformat()
            },
            'samples': {}
        }
        
        # 进度条
        successful = 0
        failed = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Precomputing matrices"):
            try:
                # 构建图像路径
                if 'ApexPath' in df.columns and pd.notna(row['ApexPath']):
                    # 如果CSV中有直接的图像路径
                    image_path = row['ApexPath']
                    if image_base_dir and not Path(image_path).is_absolute():
                        image_path = Path(image_base_dir) / image_path
                else:
                    # 根据视频目录和apex帧号查找图像
                    # 假设视频目录结构为: {dataset}/{subject}/{video_name}/
                    video_dir = Path(row['Dataset']) / row['Subject'] / row['Filename']
                    if image_base_dir:
                        video_dir = Path(image_base_dir) / video_dir
                    
                    apex_frame = int(row['ApexFrame'])
                    image_path = self._find_apex_frame_image(str(video_dir), apex_frame)
                
                # 计算矩阵
                success, A_Component, A_Region, image_hash = self.compute_matrices_for_image(
                    image_path, use_default_on_fail=True
                )
                
                # 保存结果
                cache_filename = f"{image_hash}.pkl"
                cache_file = self.cache_dir / cache_filename
                
                result = {
                    'image_path': str(image_path),
                    'image_hash': image_hash,
                    'A_Component': A_Component,
                    'A_Region': A_Region,
                    'success': success,
                    'csv_row': {
                        'Dataset': row['Dataset'],
                        'Subject': row['Subject'],
                        'Filename': row['Filename'],
                        'Label': row['Label'],
                        'ApexFrame': row['ApexFrame']
                    }
                }
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # 更新索引
                sample_key = f"{row['Dataset']}_{row['Subject']}_{row['Filename']}"
                self.index_data['samples'][sample_key] = {
                    'hash': image_hash,
                    'cache_file': cache_filename,
                    'success': success,
                    'label': row['Label'],
                    'apex_frame': row['ApexFrame']
                }
                
                successful += 1
                
            except Exception as e:
                print(f"❌ Failed to process row {idx} ({row['Dataset']}/{row['Subject']}/{row['Filename']}): {str(e)}")
                traceback.print_exc()
                failed += 1
        
        # 保存索引
        with open(index_file, 'w') as f:
            json.dump(self.index_data, f, indent=2, default=str)
        
        print(f"\nPrecomputation complete:")
        print(f"  Successful: {successful}/{len(df)}")
        print(f"  Failed: {failed}/{len(df)}")
        print(f"  Output directory: {output_dir}")
        print(f"  Index file: {index_file}")
        
        return successful, failed


class PrecomputedMatrixLoader:
    """加载预计算的矩阵"""
    
    def __init__(self, cache_dir: str, default_on_fail: bool = True, 
                 mean_landmarks_path: str = None, target_size: int = 224, patch_size: int = 16):
        """
        初始化加载器
        
        Args:
            cache_dir: 缓存目录
            default_on_fail: 加载失败时是否使用默认矩阵
            mean_landmarks_path: 平均关键点文件路径
            target_size: 目标图像尺寸
            patch_size: patch大小
        """
        self.cache_dir = Path(cache_dir)
        self.index_file = self.cache_dir / "matrix_index.json"
        self.default_on_fail = default_on_fail
        self.target_size = target_size
        self.patch_size = patch_size
        
        # 加载索引
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index_data = json.load(f)
            print(f"Loaded index with {len(self.index_data.get('samples', {}))} samples")
        else:
            self.index_data = {'samples': {}}
            print(f"Warning: Index file not found at {self.index_file}")
        
        # 创建默认矩阵
        if default_on_fail:
            # 尝试加载平均关键点文件
            if mean_landmarks_path and os.path.exists(mean_landmarks_path):
                try:
                    mean_landmarks = np.load(mean_landmarks_path)
                    print(f"Loaded mean landmarks from {mean_landmarks_path}")
                    self.default_A_Component, self.default_A_Region = self._create_default_matrices_from_landmarks(
                        mean_landmarks, target_size, patch_size
                    )
                except Exception as e:
                    print(f"Error loading mean landmarks from {mean_landmarks_path}: {str(e)}")
                    print("Using uniform default matrices instead.")
                    self.default_A_Component, self.default_A_Region = self._create_uniform_default_matrices(
                        target_size, patch_size
                    )
            else:
                print(f"Mean landmarks file not found at {mean_landmarks_path}, using uniform default matrices.")
                self.default_A_Component, self.default_A_Region = self._create_uniform_default_matrices(
                    target_size, patch_size
                )
    
    def _create_default_matrices_from_landmarks(self, mean_landmarks: np.ndarray, target_size: int, patch_size: int):
        """从平均关键点创建默认矩阵"""
        N_h, N_w = target_size // patch_size, target_size // patch_size
        N = N_h * N_w
        M_C = 8
        
        # 从平均关键点创建区域划分
        facial_regions = {
            "left_eyebrow": [mean_landmarks[i] for i in range(17, 22)],
            "right_eyebrow": [mean_landmarks[i] for i in range(22, 27)],
            "left_eye": [mean_landmarks[i] for i in range(36, 42)],
            "right_eye": [mean_landmarks[i] for i in range(42, 48)],
            "nose": [mean_landmarks[i] for i in range(27, 36)],
            "mouth": [mean_landmarks[i] for i in range(48, 68)],
            "left_cheek": [mean_landmarks[i] for i in [1, 2, 3, 4, 31, 32, 48]],
            "right_cheek": [mean_landmarks[i] for i in [12, 13, 14, 15, 34, 35, 54]]
        }
        
        # 计算默认A_Component矩阵
        default_A_Component = np.zeros((M_C, N), dtype=np.float32)
        from scipy.spatial import ConvexHull
        
        region_names = list(facial_regions.keys())
        for r_idx, (region_name, region_points) in enumerate(zip(region_names, facial_regions.values())):
            points_array = np.array(region_points)
            
            if "eyebrow" in region_name or len(points_array) < 3:
                # 对于眉毛或点数不足的区域，使用点覆盖
                for x, y in region_points:
                    patch_x = int(x // patch_size)
                    patch_y = int(y // patch_size)
                    if 0 <= patch_x < N_w and 0 <= patch_y < N_h:
                        patch_idx = patch_y * N_w + patch_x
                        default_A_Component[r_idx, patch_idx] = 1.0
            else:
                # 使用凸包
                try:
                    hull = ConvexHull(points_array)
                    hull_points = points_array[hull.vertices]
                    
                    # 检查每个patch中心是否在凸包内
                    for py in range(N_h):
                        for px in range(N_w):
                            patch_center_x = (px + 0.5) * patch_size
                            patch_center_y = (py + 0.5) * patch_size
                            
                            # 判断点是否在凸包内
                            if self._is_point_in_polygon((patch_center_x, patch_center_y), hull_points):
                                patch_idx = py * N_w + px
                                default_A_Component[r_idx, patch_idx] = 1.0
                except:
                    # 凸包失败，使用点覆盖
                    for x, y in region_points:
                        patch_x = int(x // patch_size)
                        patch_y = int(y // patch_size)
                        if 0 <= patch_x < N_w and 0 <= patch_y < N_h:
                            patch_idx = patch_y * N_w + patch_x
                            default_A_Component[r_idx, patch_idx] = 1.0
        
        # 计算默认A_Region矩阵
        M_R = 4
        default_A_Region = np.zeros((M_R, N), dtype=np.float32)
        mid_w = target_size // 2
        mid_h = target_size // 2
        
        for py in range(N_h):
            for px in range(N_w):
                patch_center_x = (px + 0.5) * patch_size
                patch_center_y = (py + 0.5) * patch_size
                patch_idx = py * N_w + px
                
                if patch_center_x < mid_w and patch_center_y < mid_h:
                    default_A_Region[0, patch_idx] = 1.0  # 左上
                elif patch_center_x >= mid_w and patch_center_y < mid_h:
                    default_A_Region[1, patch_idx] = 1.0  # 右上
                elif patch_center_x < mid_w and patch_center_y >= mid_h:
                    default_A_Region[2, patch_idx] = 1.0  # 左下
                else:
                    default_A_Region[3, patch_idx] = 1.0  # 右下
        
        return default_A_Component, default_A_Region
    
    def _create_uniform_default_matrices(self, target_size: int, patch_size: int):
        """创建均匀分布的默认矩阵（当没有平均关键点时使用）"""
        N_h, N_w = target_size // patch_size, target_size // patch_size
        N = N_h * N_w
        
        # 均匀分布的A_Component矩阵
        default_A_Component = np.ones((8, N)) * 0.5
        
        # 简单的A_Region矩阵（四个象限）
        default_A_Region = np.zeros((4, N), dtype=np.float32)
        mid_w = target_size // 2
        mid_h = target_size // 2
        
        for py in range(N_h):
            for px in range(N_w):
                patch_center_x = (px + 0.5) * patch_size
                patch_center_y = (py + 0.5) * patch_size
                patch_idx = py * N_w + px
                
                if patch_center_x < mid_w and patch_center_y < mid_h:
                    default_A_Region[0, patch_idx] = 1.0  # 左上
                elif patch_center_x >= mid_w and patch_center_y < mid_h:
                    default_A_Region[1, patch_idx] = 1.0  # 右上
                elif patch_center_x < mid_w and patch_center_y >= mid_h:
                    default_A_Region[2, patch_idx] = 1.0  # 左下
                else:
                    default_A_Region[3, patch_idx] = 1.0  # 右下
        
        return default_A_Component, default_A_Region
    
    def _is_point_in_polygon(self, point, polygon):
        """判断点是否在多边形内（射线法）"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_matrices(self, dataset: str, subject: str, filename: str, return_tensor: bool = True):
        """
        根据数据集信息获取矩阵
        
        Args:
            dataset: 数据集名
            subject: 受试者ID
            filename: 文件名
            return_tensor: 是否返回张量
            
        Returns:
            (A_Component, A_Region, success)
        """
        sample_key = f"{dataset}_{subject}_{filename}"
        
        if sample_key in self.index_data['samples']:
            sample_info = self.index_data['samples'][sample_key]
            cache_file = self.cache_dir / sample_info['cache_file']
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    A_Component = data['A_Component']
                    A_Region = data['A_Region']
                    
                    if return_tensor:
                        A_Component = torch.from_numpy(A_Component).float()
                        A_Region = torch.from_numpy(A_Region).float()
                    
                    return A_Component, A_Region, sample_info['success']
                except Exception as e:
                    print(f"Error loading cache file {cache_file}: {str(e)}")
        
        # 使用默认矩阵
        if self.default_on_fail:
            if return_tensor:
                return (torch.from_numpy(self.default_A_Component).float(),
                       torch.from_numpy(self.default_A_Region).float(),
                       False)
            else:
                return self.default_A_Component, self.default_A_Region, False
        else:
            raise ValueError(f"No precomputed matrix for {sample_key}")
    
    def get_matrices_by_csv_row(self, csv_row: pd.Series, return_tensor: bool = True):
        """
        根据CSV行获取矩阵
        
        Args:
            csv_row: pandas Series对象（CSV的一行）
            return_tensor: 是否返回张量
            
        Returns:
            (A_Component, A_Region, success)
        """
        return self.get_matrices(
            csv_row['Dataset'],
            csv_row['Subject'],
            csv_row['Filename'],
            return_tensor
        )
    
    def batch_get_matrices(self, sample_keys: List[str], return_tensor: bool = True):
        """
        批量获取矩阵
        
        Args:
            sample_keys: 样本键列表
            return_tensor: 是否返回张量
            
        Returns:
            (batch_A_Component, batch_A_Region, batch_success)
        """
        batch_A_Component = []
        batch_A_Region = []
        batch_success = []
        
        for key in sample_keys:
            # 解析样本键
            parts = key.split('_')
            if len(parts) >= 3:
                dataset, subject, filename = parts[0], parts[1], '_'.join(parts[2:])
                A_Comp, A_Reg, success = self.get_matrices(dataset, subject, filename, False)
                batch_A_Component.append(A_Comp)
                batch_A_Region.append(A_Reg)
                batch_success.append(success)
            else:
                print(f"Invalid sample key: {key}")
                if self.default_on_fail:
                    batch_A_Component.append(self.default_A_Component)
                    batch_A_Region.append(self.default_A_Region)
                    batch_success.append(False)
        
        if return_tensor:
            batch_A_Component = torch.stack([torch.from_numpy(arr).float() for arr in batch_A_Component])
            batch_A_Region = torch.stack([torch.from_numpy(arr).float() for arr in batch_A_Region])
        else:
            batch_A_Component = np.stack(batch_A_Component)
            batch_A_Region = np.stack(batch_A_Region)
        
        return batch_A_Component, batch_A_Region, batch_success


def main():
    parser = argparse.ArgumentParser(description='Precompute facial matrices from CSV')
    parser.add_argument('--csv_path', type=str, 
                       default='data/SAMM/precomputed_flows/dataset_with_flows.csv',
                       help='Path to the CSV file')
    parser.add_argument('--output_dir', type=str, default='data/SAMM/precomputed_matrices',
                       help='Output directory for precomputed matrices')
    parser.add_argument('--image_base_dir', type=str, default='data',
                       help='Base directory for images (if paths in CSV are relative)')
    parser.add_argument('--dlib_shape_predictor', type=str, 
                       default='shape_predictor_68_face_landmarks.dat',
                       help='Path to dlib shape predictor')
    parser.add_argument('--mean_landmarks', type=str,
                       default='data/3DB/mean_landmarks_224x224.npy',
                       help='Path to mean landmarks file')
    parser.add_argument('--target_size', type=int, default=224,
                       help='Target image size for processing')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size for matrix construction')
    
    args = parser.parse_args()
    
    # 初始化预计算器
    precomputer = FacialMatrixPrecomputer(
        dlib_shape_predictor=args.dlib_shape_predictor,
        target_size=args.target_size,
        patch_size=args.patch_size,
        mean_landmarks_path=args.mean_landmarks
    )
    
    # 执行预计算
    successful, failed = precomputer.precompute_from_csv(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        image_base_dir=args.image_base_dir
    )
    
    # # 测试加载器 - 传递mean_landmarks路径
    # print("\nTesting loader...")
    # loader = PrecomputedMatrixLoader(
    #     cache_dir=args.output_dir,
    #     default_on_fail=True,
    #     mean_landmarks_path=args.mean_landmarks,
    #     target_size=args.target_size,
    #     patch_size=args.patch_size
    # )
    
    # # 读取CSV并测试几个样本
    # df = pd.read_csv(args.csv_path)
    # test_samples = df.iloc[:3] if len(df) >= 3 else df

    
    # for _, row in test_samples.iterrows():
    #     try:
    #         A_Comp, A_Reg, success = loader.get_matrices_by_csv_row(row)
    #         status = "✓" if success else "⚠️ (default)"
    #         print(f"  {status} {row['Dataset']}/{row['Subject']}/{row['Filename']}: "
    #               f"A_Component {A_Comp.shape}, A_Region {A_Reg.shape}")
    #     except Exception as e:
    #         print(f"  ✗ {row['Dataset']}/{row['Subject']}/{row['Filename']}: {str(e)}")


if __name__ == "__main__":
    main()