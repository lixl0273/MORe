import torch
from torchvision import transforms
import os
from os.path import join
import glob
import cv2
import numpy as np
import pandas as pd
from math import floor

from PIL import Image
import torch.utils.data as data
from process.preprocess import *
from pathlib import Path
from process.facial_matrix import PrecomputedMatrixLoader


class RafDataSet(data.Dataset):
    def __init__(self,raf_path, phase, num_loso, transform_flow=None,transform_apex=None,transform_aug=None,num_classes=5,db='smic',combined=False,use_facial_matrices=False, matrix_cache_dir=None, matrix_loader=None):
        self.phase = phase
        self.raf_path = raf_path
        self.transform_flow = transform_flow
        self.transform_apex=transform_apex
        self.transform_aug=transform_aug
        self.db = db
        self.use_facial_matrices = use_facial_matrices
        self.matrix_cache_dir = matrix_cache_dir
        # 如果使用面部矩阵，初始化加载器
        if use_facial_matrices and matrix_loader is None:
            self.matrix_loader = PrecomputedMatrixLoader(
                cache_dir=matrix_cache_dir,
                default_on_fail=True,
                mean_landmarks_path=os.path.join(raf_path, '3DB/mean_landmarks_224x224.npy'),
                target_size=224,
                patch_size=16
            )
            print(f"初始化面部矩阵加载器，缓存目录: {matrix_cache_dir}")
        elif use_facial_matrices:
            self.matrix_loader = matrix_loader
        else:
            self.matrix_loader = None

        self.db_name=[]
        self.file_paths_onset = []
        self.file_paths_apex = []
        self.file_paths_offset=[]
        self.file_names = []
        self.label= []
        self.sub = []
        # self.label_au=[]
        self.label_db=[]
        self.onset_paths=[]
        self.apex_paths=[]
        self.farneback_flow_paths=[]
        self.tvl1_flow_paths=[]

        # 新增：存储CSV行信息，用于查找矩阵
        self.csv_rows = []

        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
       
        if combined:
            df = pd.read_csv(os.path.join(self.raf_path, '3DB/precomputed_tvl1_flows_224/dataset_with_tvl1_flows_224.csv'))
            # df.columns = ['Dataset','Subject','Filename','Label','OnsetFrame','ApexFrame','OffsetFrame','OnsetPath','ApexPath','FlowPath_Farneback','FlowPath_TVL1']
            df.columns=['Dataset','Subject','Filename','Label','OnsetFrame','ApexFrame','OffsetFrame','OnsetPath','ApexPath','FlowPath']

            if self.phase == 'train':
                dataset = df.loc[df['Subject'] != num_loso]
            else:
                dataset = df.loc[df['Subject'] == num_loso]      

            Database_name = dataset['Dataset'].values
            Label_all = dataset['Label'].values
            Subject = dataset['Subject'].values
            File_names = dataset['Filename'].values
            Onset_num = dataset['OnsetFrame'].values
            Apex_num = dataset['ApexFrame'].values
            Offset_num = dataset['OffsetFrame'].values
            Onset_path = dataset['OnsetPath'].values
            Apex_path = dataset['ApexPath'].values
            # Flow_path_farneback = dataset['FlowPath_Farneback'].values
            # Flow_path_tvl1 = dataset['FlowPath_TVL1'].values
            Flow_path_tvl1=dataset['FlowPath'].values          



            # for (f, sub, db_name, onset, apex, offset, label_all,onset_path,apex_path,flow_path_tvl1) in zip(File_names, Subject, Database_name, Onset_num, Apex_num,Offset_num, Label_all,Onset_path,Apex_path,flow_path_tvl1):
            # # for (label_all,onset_path,apex_path, flow_path_tvl1) in zip(Label_all,Onset_path,Apex_path,Flow_path_tvl1):
            for (f, sub, db_name, label_all,onset_path,apex_path,flow_path_tvl1) in zip(File_names, Subject, Database_name, Label_all,Onset_path,Apex_path,Flow_path_tvl1):
                #three classes
                if num_classes == 3:

                    self.onset_paths.append(onset_path)
                    self.apex_paths.append(apex_path)
                    # self.farneback_flow_paths.append(flow_path_farneback)
                    self.tvl1_flow_paths.append(flow_path_tvl1)

                    # [single3-fix] populate canonical aligned fields for combined CSV branch
                    # Some downstream LOSO/filtering code relies on these lists rather than
                    # onset_paths/apex_paths directly.
                    self.file_names.append(f)
                    self.sub.append(sub)
                    self.db_name.append(db_name)
                    self.label_db.append(db_name)
                    self.file_paths_onset.append(onset_path)
                    self.file_paths_apex.append(apex_path)

                    # 保存CSV行信息
                    csv_row = {
                        'Dataset': db_name,
                        'Subject': sub,
                        'Filename': f,
                    }
                    self.csv_rows.append(csv_row)                   
                    

                    # (‘0‘ for negative, ‘1‘ for positive, ‘2‘ for surprise).
                    if label_all == 0:
                        self.label.append(0)
                        a=a+1
                    elif label_all == 1:
                        self.label.append(1)
                        b=b+1
                    else:
                        self.label.append(2)
                        c=c+1

                else:
                    print('wrong')

        
        
        elif self.db == 'casme2':
            df = pd.read_csv(os.path.join(self.raf_path, 'CASMEII','precomputed_dual_flows/dataset_with_dual_flows.csv'), usecols=[0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12])
            df.columns = ['Subject', 'Filename', 'OnsetFrame', 'ApexFrame', 'OffsetFrame', 'Label_AU', 'Label','OnsetPath','ApexPath','FlowPath_Farneback', 'FlowPath_TVL1']
            df['Subject'] = df['Subject'].apply(str)
            if self.phase == 'train':
                dataset = df.loc[df['Subject'] != num_loso]
            else:
                dataset = df.loc[df['Subject'] == num_loso]

            Label_all = dataset['Label'].values
            # Subject = dataset['Subject'].values
            # File_names = dataset['Filename'].values
            # Onset_num = dataset['OnsetFrame'].values
            # Apex_num = dataset['ApexFrame'].values
            # Offset_num = dataset['OffsetFrame'].values

            Onset_path = dataset['OnsetPath'].values
            Apex_path = dataset['ApexPath'].values
            # Flow_path = dataset['FlowPath_Farneback'].values
            Flow_path_farneback = dataset['FlowPath_Farneback'].values  # Farneback光流路径
            Flow_path_tvl1 = dataset['FlowPath_TVL1'].values 




            # for (f, sub, onset, apex, offset, label_all) in zip(File_names,Subject, Onset_num, Apex_num,Offset_num, Label_all):
            for (label_all,onset_path,apex_path,flow_path_farneback,flow_path_tvl1) in zip(Label_all,Onset_path,Apex_path,Flow_path_farneback,Flow_path_tvl1):
            #     #three classes
                if num_classes == 3:
                    if label_all == 'happiness' or label_all == 'repression' or label_all == 'disgust' or label_all == 'surprise' or label_all == 'fear' or label_all == 'sadness' :#or label_all=='others':

                        # self.file_paths_onset.append(onset)
                        # self.file_paths_offset.append(offset)
                        # self.file_paths_apex.append(apex)
                        # self.sub.append(sub)
                        # self.file_names.append(f)

                        self.onset_paths.append(onset_path)
                        self.apex_paths.append(apex_path)
                        self.farneback_flow_paths.append(flow_path_farneback)
                        self.tvl1_flow_paths.append(flow_path_tvl1)
                        
                        if label_all == 'happiness':
                            self.label.append(0)
                            a=a+1
                        elif label_all == 'surprise':
                            self.label.append(1)
                            b=b+1
                        else:
                            self.label.append(2)
                            c=c+1

                elif num_classes==5:
                #five classes
                    if label_all == 'happiness' or label_all == 'repression' or label_all == 'disgust' or label_all == 'surprise' or label_all == 'others':

                        # self.file_paths_onset.append(onset)
                        # self.file_paths_offset.append(offset)
                        # self.file_paths_apex.append(apex)
                        # self.sub.append(sub)
                        # self.file_names.append(f)

                        self.onset_paths.append(onset_path)
                        self.apex_paths.append(apex_path)
                        self.farneback_flow_paths.append(flow_path_farneback)
                        self.tvl1_flow_paths.append(flow_path_tvl1)

                        if label_all == 'happiness':
                            self.label.append(0)

                            a = a + 1
                        elif label_all == 'repression':
                            self.label.append(1)

                            b = b + 1
                        elif label_all == 'disgust':
                            self.label.append(2)

                            c = c + 1
                        elif label_all == 'surprise':
                            self.label.append(3)

                            d = d + 1
                        else:
                            self.label.append(4)
                            e = e + 1

                else:
                    print('wrong')


        elif self.db == 'samm':
            
            df = pd.read_excel(os.path.join(self.raf_path, "SAMM","SAMM_Micro_FACS_Codes_v2.xlsx"), usecols=[0, 1, 3, 4, 5, 8, 9])
            df.columns = ['Subject', 'Filename', 'OnsetFrame', 'ApexFrame', 'OffsetFrame', 'Label_AU', 'Label']

            if self.phase == 'train':
                dataset = df.loc[df['Subject'] != num_loso]
                # print('train_dataset:',len(dataset))
            else:
                dataset = df.loc[df['Subject'] == num_loso] 
                # print('test_dataset:',len(dataset))   


            Label_all = dataset['Label'].values
            Subject = dataset['Subject'].values
            File_names = dataset['Filename'].values
            Onset_num = dataset['OnsetFrame'].values
            Apex_num = dataset['ApexFrame'].values
            Offset_num = dataset['OffsetFrame'].values

            for (f, sub, onset, apex, offset, label_all) in zip(File_names,Subject, Onset_num, Apex_num,Offset_num, Label_all):
                # Anger,Sadness,Surprise,Fear,Other,Happiness,Disgust,Contempt
                # try:
                #     label_all = str(label_all).strip().lower()  # 统一格式
                #     # print("当前标签:", label_all)  # 调试输出
                #     if num_classes == 3:
                #         if label_all in ['happiness', 'anger', 'contempt', 'disgust', 'fear', 'sadness', 'surprise']:
                #             self.file_paths_onset.append(onset)
                #             self.file_paths_offset.append(offset)
                #             self.file_paths_apex.append(apex)
                #             self.sub.append(sub)
                #             self.file_names.append(f)

                #             if label_all == 'happiness':
                #                 self.label.append(0)
                #                 a += 1
                #             elif label_all == 'surprise':
                #                 self.label.append(1)
                #                 b += 1
                #             else:
                #                 self.label.append(2)
                #                 c += 1
                # except Exception as e:
                #     print(f"处理数据时出错: {e}, 数据: {label_all}")
                
                
                #three classes
                if num_classes == 3:
                    if label_all == 'Happiness' or label_all == 'Anger' or label_all == 'Contempt' or label_all == 'Disgust' or label_all == 'Fear' or label_all == 'Sadness' or label_all == 'Surprise':
                        self.file_paths_onset.append(onset)
                        self.file_paths_offset.append(offset)
                        self.file_paths_apex.append(apex)
                        self.sub.append(sub)
                        self.file_names.append(f)
                        
                        if label_all == 'Happiness':
                            self.label.append(0)
                            a=a+1
                        elif label_all == 'Surprise':
                            self.label.append(1)
                            b=b+1
                        else:
                            self.label.append(2)
                            c=c+1

                elif num_classes==5:
                #five classes
                    if label_all == 'Happiness' or label_all == 'Anger' or label_all == 'Contempt' or label_all == 'Surprise' or label_all == 'Other':

                        self.file_paths_onset.append(onset)
                        self.file_paths_offset.append(offset)
                        self.file_paths_apex.append(apex)
                        self.sub.append(sub)
                        self.file_names.append(f)

                        if label_all == 'Happiness':
                            self.label.append(0)

                            a = a + 1
                        elif label_all == 'Anger':
                            self.label.append(1)

                            b = b + 1
                        elif label_all == 'Contempt':
                            self.label.append(2)

                            c = c + 1
                        elif label_all == 'Surprise':
                            self.label.append(3)

                            d = d + 1
                        else:
                            self.label.append(4)
                            e = e + 1
        
        else:
            print('wrong') 


    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):

        onset_path=self.onset_paths[idx]
        apex_path=self.apex_paths[idx]
        # farneback_flow_path=self.farneback_flow_paths[idx]
        tvl1_flow_path=self.tvl1_flow_paths[idx]


        image_onset_bgr = cv2.imread(onset_path)
        image_apex_bgr = cv2.imread(apex_path) #BGR


        if image_apex_bgr is None:
            raise ValueError(f"无法读取图像: {apex_path}")
            
        # 检查图像尺寸
        if image_apex_bgr.size == 0:
            raise ValueError(f"空图像: {apex_path}")

        image_apex_bgr=cv2.resize(image_apex_bgr,(224,224))

        # farneback_flow=np.load(farneback_flow_path)
        tvl1_flow=np.load(tvl1_flow_path)
        flow = tvl1_flow.transpose(2, 0, 1)
        flow = torch.from_numpy(flow).float()

        # flow=np.concatenate((farneback_flow,tvl1_flow),axis=2)
        # flow = flow.transpose(2, 0, 1)  # (224,224,3) -> (3,224,224)

        # flow=torch.from_numpy(flow).float()

        # mean_6channel = [
        #     0.017, 0.003, 0.586,  # Farneback: u, v, strain
        #     0.474, 0.515, 0.025    # TV-L1: u, v, strain
        # ]

        # std_6channel = [
        #     0.994, 0.918, 0.759,   # Farneback: u, v, strain
        #     0.203, 0.239, 0.046    # TV-L1: u, v, strain
        # ]

        # transform_flow = transforms.Compose([
        #     transforms.Normalize(mean=mean_6channel, std=std_6channel)
        # ])

        # flow=transform_flow(flow)


        # if farneback_flow.ndim == 3 and farneback_flow.shape[2] == 3:
        #     farneback_flow = farneback_flow.transpose(2, 0, 1)  # [3, H, W]
        # if tvl1_flow.ndim == 3 and tvl1_flow.shape[2] == 3:
        #     tvl1_flow = tvl1_flow.transpose(2, 0, 1) 
        # weight_farneback = 0.3
        # weight_tvl1 = 0.7

        # total_weight = weight_farneback + weight_tvl1
        # weight_farneback /= total_weight
        # weight_tvl1 /= total_weight

        # flow = weight_farneback * farneback_flow + weight_tvl1 * tvl1_flow
        # flow = torch.from_numpy(flow).float()

        label = self.label[idx]

        # 新增：加载面部矩阵
        if self.use_facial_matrices and self.matrix_loader is not None:
            try:
                # 获取CSV行信息
                csv_row = self.csv_rows[idx]
                
                # 获取面部矩阵
                A_Component, A_Region, success = self.matrix_loader.get_matrices(
                    dataset=csv_row['Dataset'],
                    subject=csv_row['Subject'],
                    filename=csv_row['Filename'],
                    return_tensor=True
                )
                
                # 如果获取失败，使用默认矩阵
                # if not success:
                    # print(f"警告: 样本 {csv_row['Dataset']}/{csv_row['Subject']}/{csv_row['Filename']} 使用默认矩阵")
                
                # 扩展维度以适应批量处理 [8,196] -> [1,8,196]
                A_Component = A_Component.unsqueeze(0)  # [1, 8, 196]
                A_Region = A_Region.unsqueeze(0)        # [1, 4, 196]
                
            except Exception as e:
                print(f"加载面部矩阵失败: {str(e)}")
                # 创建空的占位符矩阵
                A_Component = torch.zeros((1, 8, 196), dtype=torch.float32)
                A_Region = torch.zeros((1, 4, 196), dtype=torch.float32)
                success = False
        else:
            # 不使用面部矩阵时，创建空的占位符
            A_Component = torch.zeros((1, 8, 196), dtype=torch.float32)
            A_Region = torch.zeros((1, 4, 196), dtype=torch.float32)
            success = False

        if self.phase!='test':  
            # flow_nor = self.transform_flow(flow)
            flow_nor=flow
            image_apex_nor = self.transform_apex(image_apex_bgr)
            if self.transform_aug is not None:
                flow_nor = self.transform_aug(flow_nor)
                image_apex_nor = self.transform_aug(image_apex_nor)

            if self.use_facial_matrices:
                return {
                    'flow': flow_nor,
                    'image_apex': image_apex_nor,
                    'label': label,
                    'A_Component': A_Component,
                    'A_Region': A_Region
                }
            return flow_nor,image_apex_nor,label
        else:
            # flow_nor = self.transform_flow(flow)
            flow_nor=flow
            image_apex_nor = self.transform_apex(image_apex_bgr)
            if self.use_facial_matrices:
                return {
                    'flow': flow_nor,
                    'image_apex': image_apex_nor,
                    'label': label,
                    'A_Component': A_Component,
                    'A_Region': A_Region
                }
            return flow_nor,image_apex_nor,label      
        


        