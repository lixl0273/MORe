import os
import torch

if os.environ.get("OFE_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("[env] cuDNN disabled by OFE_DISABLE_CUDNN=1")

import os
import sys
import torch

if os.environ.get("OFE_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("[env] OFE_DISABLE_CUDNN=1: cuDNN disabled for this run.")
from model1.theory.selective_flow_mix import selective_flow_mix, build_same_class_index
from model1.theory.info_losses import compute_full_information_loss
import torch.utils.data as data
from timm.loss import *
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.insert(0, project_root)
import metrics2
from model1.FRASA import *
from process.facial_matrix import PrecomputedMatrixLoader


torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)
# CUDA_VISIBLE_DEVICES is controlled by shell

try:
    import dlib
except ImportError:
    dlib = None
# 加载 dlib 的人脸检测器和关键点预测器
detector = None  # dlib disabled; using precomputed facial matrices
predictor = None  # dlib disabled; using precomputed facial matrices

from utils import *
from model1.dataset import *
from model1.model import *
from model1.cser_modules import MSMERWithCSER

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='./data/microemotion', help='Raf-DB dataset path.')
    parser.add_argument('--model_name', default='en_FRAT', help='the model architecture')
    parser.add_argument('--use_mpt', action='store_true', help='Enable Motion-Preserving Tokenization stem')
    parser.add_argument('--mpt_variant', type=str, default='full', choices=['none','s2d','local','full','dw_pw','amp','dir','add','mul'], help='MPT variant')
    parser.add_argument('--mpt_residual_scale', type=float, default=0.75, help='Residual scale of MPT stem')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0008, help='Initial learning rate for sgd.')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--loss_function', default='crossentropy', help='the loss functions')
    parser.add_argument('--optimizer_name', default='adam', help='the optimizer')
    parser.add_argument('--trans_depth',type=int,default=3, help='Transformer Layers')
    parser.add_argument('--trans_num_heads',type=int, default=2, help='Transformer Num Heads')
    parser.add_argument('--embed_dim',type=int, default=512, help='Embedding Dim')
    parser.add_argument('--num_classes',type=int, default=3, help='Classes') 
    parser.add_argument('--db',type=str, default='3DB', help='Dataset')    #casme2,samm,
    parser.add_argument('--combined', default=True, dest='combined', action='store_true',
                    help='if the label of dataset is the combined 3 classes. ')
    parser.add_argument('--gamma', default=0.987, type=float)
    
    parser.add_argument('--image_size', default=224, type=int,help='input image size (default: 224)')
    parser.add_argument('--patch_size', default=16, type=int,help='patch size (default: 16)')
    parser.add_argument('--use_facial_matrices', default=True)


    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=1000,
                        help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    parser.add_argument('--patchup_prob', type=float, default=.7, help='PatchUp probability')
    

    parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

    parser.add_argument('--magnitude_percentile',type=float, default=0.6, help='magnitude_percentile')
    parser.add_argument('--direction_percentile',type=float, default=0.6, help='direction_percentile')
    parser.add_argument('--ostu',default=False, help='ostu')
    parser.add_argument('--mix_alpha',default=1.0, help='mix_alpha')
    parser.add_argument('--theory_full', action='store_true', help='Enable full information-theoretic EN-FRAT objective')
    parser.add_argument('--info_alpha', type=float, default=1.0, help='Weight for enhanced CE loss')
    parser.add_argument('--info_beta', type=float, default=0.05, help='Weight for structural compression loss')
    parser.add_argument('--info_mu', type=float, default=0.05, help='Weight for structural information gain loss')
    parser.add_argument('--info_nu', type=float, default=0.1, help='Weight for prediction consistency loss')
    parser.add_argument('--sfm_alpha', type=float, default=0.4, help='Beta distribution alpha for selective flow-mix')
    parser.add_argument('--sfm_q', type=float, default=0.6, help='Quantile threshold for selective flow-mix mask')
    parser.add_argument('--sfm_same_class', action='store_true', help='Use same-class selective flow-mix')
    parser.add_argument('--sfm_temperature', type=float, default=0.0, help='Soft mask temperature; <=0 means hard mask')
    parser.add_argument('--max-folds', '--max_folds', dest='max_folds', type=int, default=0, help='Debug: only run first K LOSO folds; 0 means all')
    parser.add_argument('--rho_com', type=float, default=0.75, help='Motion calibration ratio for component prior')
    parser.add_argument('--rho_reg', type=float, default=0.75, help='Motion calibration ratio for region prior')
    parser.add_argument('--motion_tau', type=float, default=0.7, help='Temperature for motion affinity')
    parser.add_argument('--prior_type', type=str, default='nonparam', choices=['static', 'nonparam', 'learned_relation'], help='Structural prior type')
    parser.add_argument('--branch_mode', type=str, default='both', choices=['none', 'component', 'region', 'both'], help='Structural branch ablation mode')
    parser.add_argument('--gate_mode', type=str, default='learned', choices=['learned', 'fixed'], help='Gate ablation mode')
    parser.add_argument('--fixed_gamma_com', type=float, default=1.0, help='Fixed component structural weight')
    parser.add_argument('--fixed_gamma_reg', type=float, default=1.0, help='Fixed region structural weight')
    parser.add_argument('--gate_type', type=str, default='sigmoid', choices=['sigmoid', 'softmax'], help='Granularity gate type')
    parser.add_argument('--mix_strategy', type=str, default='selective', choices=['selective', 'global'], help='Flow mixing strategy for theory objective')


    # ---- CSER: causal structural evidence reasoning ----
    parser.add_argument('--experiment', type=str, default='E0',
                        help='Experiment tag, e.g., E0/E1/E2/E3/E4/E5.')
    parser.add_argument('--run_name', type=str, default='',
                        help='Optional run name used to avoid log overwrite.')
    parser.add_argument('--use_cser', action='store_true',
                        help='Enable CSER wrapper on top of original MSMER.')
    parser.add_argument('--cser_mode', type=str, default='residual',
                        choices=['residual', 'direct'],
                        help='residual keeps original MSPA path; direct uses CSER representation for classification.')
    parser.add_argument('--cser_graph_type', type=str, default='soft',
                        choices=['prior', 'learn', 'soft'],
                        help='Graph source: anatomical prior, learnable graph, or soft mixture.')
    parser.add_argument('--cser_prior_beta', type=float, default=0.7,
                        help='Mixture weight for prior graph in soft graph.')
    parser.add_argument('--cser_alpha', type=float, default=0.1,
                        help='Residual scale for CSER residual branch.')
    parser.add_argument('--use_cser_reliability', action='store_true',
                        help='Enable region/token-wise evidence reliability modulation.')
    parser.add_argument('--cser_rel_mode', type=str, default='naive',
                        choices=['naive', 'residual', 'bounded'],
                        help='Reliability modulation mode. naive may suppress weak evidence; residual/bounded are safer.')
    parser.add_argument('--cser_rel_gamma', type=float, default=0.5,
                        help='Scale for residual reliability modulation.')
    parser.add_argument('--cser_rel_min', type=float, default=0.5,
                        help='Lower bound for bounded reliability modulation.')
    parser.add_argument('--cser_update_bound', type=float, default=0.0,
                        help='Per-token relative bound for the final CSER residual update.')
    parser.add_argument('--use_cser_counterfactual', action='store_true',
                        help='Enable counterfactual evidence auxiliary loss.')
    parser.add_argument('--cser_cf_weight', type=float, default=0.03,
                        help='Weight for counterfactual auxiliary loss.')
    parser.add_argument('--cser_cf_margin', type=float, default=0.02,
                        help='Margin for counterfactual confidence drop loss.')
    parser.add_argument('--cser_dropout', type=float, default=0.0,
                        help='Dropout inside CSER graph reasoning.')

    return parser.parse_args()


def maybe_wrap_cser(net_all, args, image_size, patch_size, embed_dim, num_classes):
    """Wrap original MSMER with CSER only when --use_cser is enabled."""
    if not getattr(args, 'use_cser', False):
        return net_all
    num_tokens = (int(image_size) // int(patch_size)) ** 2
    print(
        f"[CSER] enabled: experiment={getattr(args, 'experiment', '')}, "
        f"mode={args.cser_mode}, graph={args.cser_graph_type}, "
        f"alpha={args.cser_alpha}, reliability={getattr(args, 'use_cser_reliability', False)}, "
        f"rel_mode={getattr(args, 'cser_rel_mode', 'naive')}, "
        f"counterfactual={getattr(args, 'use_cser_counterfactual', False)}, "
        f"cf_weight={getattr(args, 'cser_cf_weight', 0.0)}"
    )
    return MSMERWithCSER(
        base_model=net_all,
        embed_dim=embed_dim,
        num_tokens=num_tokens,
        num_classes=num_classes,
        mode=args.cser_mode,
        graph_type=args.cser_graph_type,
        prior_beta=args.cser_prior_beta,
        residual_alpha=args.cser_alpha,
        use_reliability=getattr(args, 'use_cser_reliability', False),
        reliability_mode=getattr(args, 'cser_rel_mode', 'naive'),
        reliability_gamma=getattr(args, 'cser_rel_gamma', 0.5),
        reliability_min=getattr(args, 'cser_rel_min', 0.5),
        relative_update_bound=getattr(args, 'cser_update_bound', 0.0),
        use_counterfactual=getattr(args, 'use_cser_counterfactual', False),
        cf_margin=getattr(args, 'cser_cf_margin', 0.02),
        dropout=getattr(args, 'cser_dropout', 0.0),
    )


def initialize_weight_goog(m, n=''):
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()



def global_flow_mix(flow, labels, alpha=0.4, same_class=True):
    """
    Global Mixup in optical-flow domain.
    Unlike selective_flow_mix, this mixes the entire flow field.
    """
    B = flow.size(0)
    device = flow.device

    if same_class:
        index = build_same_class_index(labels)
    else:
        index = torch.randperm(B, device=device)

    beta = torch.distributions.Beta(alpha, alpha)
    lam = beta.sample((B,)).to(device=device, dtype=flow.dtype).view(B, 1, 1, 1)

    mixed = lam * flow + (1.0 - lam) * flow[index]
    aux = {"mix_index": index, "lambda": lam}
    return mixed, labels, aux

def criterion2(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat((y_pred_neg, zeros), dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return torch.mean(neg_loss + pos_loss)

def main():

    args = parse_args()
    raf_path=args.raf_path
    batch_size=args.batch_size
    optimizer_name=args.optimizer_name
    lr=args.lr
    epochs=args.epochs
    model_name=args.model_name
    db=args.db

    # Canonicalize dataset name for single-dataset protocols.
    # Keep single-dataset names lowercase because transform branches use
    # db == 'casme2' / 'samm' / 'smic'.
    db_raw = str(db)
    db_key = db_raw.lower()
    if db_key in {"casme2", "casmeii", "casme ii"}:
        db = "casme2"
    elif db_key == "samm":
        db = "samm"
    elif db_key == "smic":
        db = "smic"
    elif db_key in {"3db", "all", "combined", "combined3"}:
        db = "3DB"
    combined=args.combined
    if db in {"casme2", "samm", "smic"}:
        combined = True
        print(f"[single3] force combined=True for db={db}")    
    loss_function=args.loss_function
    num_classes=args.num_classes
    workers=args.workers

    trans_depth=args.trans_depth
    trans_num_heads=args.trans_num_heads
    embed_dim=args.embed_dim
        
    ostu=args.ostu
    magnitude_percentile=args.magnitude_percentile
    direction_percentile=args.direction_percentile
    mix_alpha=args.mix_alpha

    patch_size=args.patch_size
    image_size=args.image_size
    use_facial_matrices=args.use_facial_matrices

    # 初始化矩阵加载器（全局使用同一个）
    matrix_loader = PrecomputedMatrixLoader(
        cache_dir=os.path.join(raf_path, '3DB/precomputed_matrices'),
        default_on_fail=True,
        mean_landmarks_path=os.path.join(raf_path, '3DB/mean_landmarks_224x224.npy'),
        target_size=image_size,
        patch_size=patch_size
    )


    if model_name=='vit':
        log_name = f"{model_name}_L{trans_depth}_H{trans_num_heads}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_embed{embed_dim}"
    elif model_name=='en_vit':
        if ostu:
            log_name = f"{model_name}_ostu_L{trans_depth}_H{trans_num_heads}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_embed{embed_dim}_tvl1_nonnor"
        else:
            log_name = f"{model_name}_mp{magnitude_percentile}_dp{direction_percentile}_L{trans_depth}_H{trans_num_heads}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_embed{embed_dim}_mix1"
    elif model_name=='FRAT':
        log_name = f"{model_name}_L{trans_depth}_H{trans_num_heads}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_embed{embed_dim}"
    elif model_name=='en_FRAT':
        log_name = f"{model_name}_L{trans_depth}_H{trans_num_heads}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_embed{embed_dim}_ostu{ostu}_mp{magnitude_percentile}_dp{direction_percentile}_new_noaug_nosch_noval_mixnew"
 
    if getattr(args, 'run_name', ''):
        log_name = f"{args.run_name}_{log_name}"
    log_filename = f"logs/class{num_classes}/{model_name}/{db}/{image_size}/{log_name}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    sys.stdout = Logger(log_filename)
    sys.stderr = sys.stdout

    start_time = time.time()
    data_loading_time = 0


    ### data augmentation for training set only
    data_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(4),
        transforms.RandomCrop(224, padding=4),
    ])

    if db=='casme2':
        if num_classes==5:
            LOSO = ['17', '26', '16', '9', '5', '24', '2', '13', '4', '23', '11', '12', '8', '14', '3',
                '19', '1','18','10','20', '21', '22', '15', '6', '25', '7']
            flow_normal = transforms.Compose([
            transforms.ToTensor(),
            # 通道0 (OS Flow): 光流应变（Optical Strain）通道1 (V Flow): 垂直方向光流 通道2 (U Flow): 水平方向光流
            transforms.Normalize(mean=[0.482,0.480,0.024],
                                std=[0.199,0.239,0.045]),
        ])
            apex_normal = transforms.Compose([
                transforms.ToTensor(),
                # 图像为BGR格式，与OpenCV默认读取一致
                transforms.Normalize(mean=[0.296,0.358,0.500],
                                    std=[0.123,0.136,0.185]),
            ])
        elif num_classes==3:
            LOSO = ['17', '26', '16', '9', '5', '24', '2', '13', '4', '23', '11', '12', '8', '14', '3', '19', '1', '10',
                '20', '21', '22', '15', '6', '25', '7']
            flow_normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.474,0.515,0.025],   
                                std=[0.203,0.239,0.046]),
            ])
            apex_normal = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.303,0.365,0.508],   
                                    std=[0.119,0.135,0.187]),
            ])

    elif db=='samm':
        LOSO=[6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,30,31,32,33,34,35,36,37]
        
        if num_classes==5:
            flow_normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.018,0.507,0.496],
                                std=[0.032, 0.230,0.243]),
        ])
            apex_normal = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.504,0.504,0.504],
                                    std=[0.209, 0.209,0.209]),
            ])
        elif num_classes==3:
            flow_normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.018, 0.505,0.491],   
                                std=[0.033,0.232,0.238]),
            ])
            apex_normal = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.501,0.501,0.501],   
                                    std=[0.210,0.210,0.210]),
            ])
    elif db=='smic':
        LOSO=['s1','s2','s3','s4','s5','s6','s8','s9','s11','s12','s13','s14','s15','s18','s19','s20']
    else:
        LOSO=['sub01','sub02','sub03','sub04','sub05','sub06','sub07','sub08','sub09','sub11','sub12','sub13','sub14','sub15','sub16','sub17','sub19','sub20','sub21','sub22','sub23','sub24','sub25','sub26',
              's01','s02','s03','s04','s05','s06','s08','s09','s11','s12','s13','s14','s15','s18','s19','s20',
              '6','7','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','26','28','30','31','32','33','34','35','36','37']
        # flow_normal = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.020,0.520,0.496],   
        #                         std=[0.041,0.221,0.207]),
        #     ])
        # apex_normal = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.395,0.386,0.472],   
        #                             std=[0.187,0.182,0.200]),
        #     ])
        flow_normal = transforms.Compose([
            transforms.ToTensor(), ##3个通道对应 flow_x, flow_y, strain # R, G, B
            transforms.Normalize(mean=[0.504,0.471,0.062],   
                                std=[0.245,0.265,0.108]),
            ])
        # apex_normal = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.395,0.386,0.472],   
        #                             std=[0.187,0.182,0.200]),
        #     ])
        
        apex_normal = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.395,0.386,0.472],   
                                    std=[0.187,0.182,0.200]),
            ])

    preds_db = {}
    preds_db['casme2'] = torch.tensor([])
    preds_db['smic'] = torch.tensor([])
    preds_db['samm'] = torch.tensor([])
    preds_db['all'] = torch.tensor([])
    labels_db = {}
    labels_db['casme2'] = torch.tensor([])
    labels_db['smic'] = torch.tensor([])
    labels_db['samm'] = torch.tensor([])
    labels_db['all'] = torch.tensor([])
    allRST = []

    all_accuracy_dict = {}
    total_gt = []
    total_pred = []
    best_total_pred = []

    val_now = 0
    num_sum = 0
    pos_pred_ALL = torch.zeros(num_classes)
    pos_label_ALL = torch.zeros(num_classes)
    TP_ALL = torch.zeros(num_classes)
    acc_list = []
    epoch_list = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    


    # For single-dataset protocols, discover valid LOSO keys from the dataset itself.
    # This avoids empty folds caused by using a fixed integer LOSO list.
    if db in {"casme2", "samm", "smic"}:
        _valid_loso = []
        _seen_loso = set()

        _candidates = []
        for _i in range(1, 301):
            _candidates.extend([
                _i,
                str(_i),
                f"{_i:02d}",
                f"{_i:03d}",
                f"sub{_i:02d}",
                f"sub{_i:03d}",
                f"s{_i:02d}",
                f"s{_i:03d}",
                f"Subject{_i}",
                f"subject{_i}",
            ])

        for _key in _candidates:
            _key_id = str(_key)
            if _key_id in _seen_loso:
                continue
            _seen_loso.add(_key_id)

            try:
                _probe = RafDataSet(
                    raf_path,
                    phase='test',
                    num_loso=_key,
                    transform_flow=flow_normal,
                    transform_apex=apex_normal,
                    num_classes=num_classes,
                    db=db,
                    combined=combined,
                    use_facial_matrices=use_facial_matrices,
                    matrix_cache_dir=os.path.join(raf_path, '3DB/precomputed_matrices'),
                    matrix_loader=matrix_loader
                )
                _n = len(_probe)
            except Exception:
                _n = 0

            if _n > 0:
                _valid_loso.append(_key)

        print(f"[single3] db={db}, valid_loso={_valid_loso}")
        print(f"[single3] db={db}, num_valid_folds={len(_valid_loso)}")

        if len(_valid_loso) == 0:
            raise RuntimeError(f"No valid LOSO fold discovered for db={db}. Check RafDataSet combined/db filtering.")

        LOSO = _valid_loso


    if getattr(args, 'max_folds', 0):

    

        LOSO = LOSO[:args.max_folds]

    

        print(f"[debug] max_folds={args.max_folds}, truncated LOSO={LOSO}")

    

    for subj in LOSO:
        # print('sub:',subj)

        # 构建路径字符串
        if model_name=='vit':
            params_model_path = f"L{trans_depth}_H{trans_num_heads}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_embed{embed_dim}"
        elif model_name=='en_vit':
            if ostu:
                params_model_path = f"ostu_L{trans_depth}_H{trans_num_heads}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_embed{embed_dim}_tvl1_nonnor"
            else:
                params_model_path = f"mp{magnitude_percentile}_dp{direction_percentile}_L{trans_depth}_H{trans_num_heads}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_embed{embed_dim}_mix1"
        elif model_name=='FRAT':
            params_model_path = f"L{trans_depth}_H{trans_num_heads}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_embed{embed_dim}"
        elif model_name=='en_FRAT':
            params_model_path = f"L{trans_depth}_H{trans_num_heads}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_embed{embed_dim}_ostu{ostu}_mp{magnitude_percentile}_dp{direction_percentile}_new_noaug_nosch_noval_mixnew"       
       
        # Keep CSER parallel experiments from overwriting each other or original MSMER checkpoints.
        ckpt_root = f"class{num_classes}_cser_runs" if getattr(args, 'run_name', '') else f"class{num_classes}"
        run_subdir = getattr(args, 'run_name', '') or "manual"
        weight_path = Path(ckpt_root) / run_subdir / model_name / db / str(image_size) / params_model_path / f"{subj}.pth"

        # ------------------------------------------------------------------
        # MMC patch: optionally redirect per-fold best checkpoints to a user-
        # specified directory. This keeps the original best-subject saving
        # logic unchanged, but makes checkpoints easy to find for visualization.
        #
        # Usage:
        #   export MMC_CKPT_DIR=/path/to/CSER/checkpoints/MMC_final_combined3_LOSO
        # ------------------------------------------------------------------
        _mmc_ckpt_dir = os.environ.get("MMC_CKPT_DIR", "").strip()
        if _mmc_ckpt_dir:
            weight_path = Path(_mmc_ckpt_dir) / f"fold_{str(subj)}_best.pth"
            print(f"[MMC_CKPT] Redirect checkpoint path to: {weight_path}")
        weight_path.parent.mkdir(parents=True, exist_ok=True)

        data_load_start = time.time()

        # Fallback transforms for datasets whose db branch did not initialize them.
        # This prevents UnboundLocalError and keeps single-dataset runs executable.
        if 'flow_normal' not in locals():
            flow_normal = transforms.Compose([transforms.ToTensor()])
        if 'apex_normal' not in locals():
            apex_normal = transforms.Compose([transforms.ToTensor()])
        train_dataset = RafDataSet(raf_path, phase='train', num_loso=subj,  transform_flow=flow_normal,transform_apex=apex_normal,transform_aug=None,num_classes=num_classes,db=db,combined=combined,use_facial_matrices=use_facial_matrices,matrix_cache_dir=os.path.join(raf_path, '3DB/precomputed_matrices'),matrix_loader=matrix_loader)
        val_dataset = RafDataSet(raf_path, phase='test', num_loso=subj,  transform_flow=flow_normal,transform_apex=apex_normal,num_classes=num_classes,db=db,combined=combined,use_facial_matrices=use_facial_matrices,matrix_cache_dir=os.path.join(raf_path, '3DB/precomputed_matrices'),matrix_loader=matrix_loader)
        # Skip empty validation subject/fold.
        try:
            _val_len = len(val_dataset)
        except Exception:
            _val_len = 0
        if _val_len == 0:
            print(f"[WARN] Empty validation fold: db={db}, subj={subj}. Skip this fold.")
            continue
        data_loading_time += time.time() - data_load_start
        print(f'数据加载时间: {time.time() - data_load_start:.2f}秒')

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=workers,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 num_workers=workers,
                                                 shuffle=False,
                                                 pin_memory=True)
        print('num_sub', subj)
        print('Train set size:', train_dataset.__len__())
        print('Validation set size:', val_dataset.__len__())

        max_epoch = 0
        max_corr = 0
        max_f1 = 0
        max_pos_pred = torch.zeros(num_classes)
        max_pos_label = torch.zeros(num_classes)
        max_TP = torch.zeros(num_classes)

        if model_name=='vit':
            net_all = MyNet_FRAT(in_channels=3, num_classes=num_classes,depth=trans_depth,img_size=image_size,patch_size=patch_size,num_heads=trans_num_heads,use_frasa=False,embed_dim=embed_dim, use_mpt=getattr(args, 'use_mpt', False), mpt_variant=getattr(args, 'mpt_variant', 'full'), mpt_residual_scale=getattr(args, 'mpt_residual_scale', 0.5))
            if hasattr(net_all, 'vit_pos'):
                net_all.vit_pos.rho_com = getattr(args, 'rho_com', 0.5)
                net_all.vit_pos.rho_reg = getattr(args, 'rho_reg', 0.5)
                net_all.vit_pos.motion_tau = getattr(args, 'motion_tau', 0.7)
                net_all.vit_pos.prior_type = getattr(args, 'prior_type', 'nonparam')
                net_all.vit_pos.branch_mode = getattr(args, 'branch_mode', 'both')
                net_all.vit_pos.gate_mode = getattr(args, 'gate_mode', 'learned')
                net_all.vit_pos.fixed_gamma_com = getattr(args, 'fixed_gamma_com', 1.0)
                net_all.vit_pos.fixed_gamma_reg = getattr(args, 'fixed_gamma_reg', 1.0)

                net_all.vit_pos.gate_type = getattr(args, 'gate_type', 'sigmoid')
                for blk in getattr(net_all.vit_pos, 'blocks', []):
                    if hasattr(blk, 'attn') and hasattr(blk.attn, 'gate_type'):
                        blk.attn.gate_type = getattr(args, 'gate_type', 'sigmoid')
                print(f"[MPT] use_mpt={getattr(args, 'use_mpt', False)}, variant={getattr(args, 'mpt_variant', 'none')}, residual_scale={getattr(args, 'mpt_residual_scale', 0.5)}")
                print(f"[theory] rho_com={net_all.vit_pos.rho_com}, rho_reg={net_all.vit_pos.rho_reg}, motion_tau={net_all.vit_pos.motion_tau}, prior_type={getattr(args, 'prior_type', 'nonparam')}, branch_mode={getattr(args, 'branch_mode', 'both')}, gate_mode={getattr(args, 'gate_mode', 'learned')}, gate_type={getattr(args, 'gate_type', 'sigmoid')}")
        elif model_name=='en_vit':
            net_all=MyNet_FRAT(in_channels=3, num_classes=num_classes,depth=trans_depth,img_size=image_size,patch_size=patch_size,num_heads=trans_num_heads,use_frasa=False,embed_dim=embed_dim, use_mpt=getattr(args, 'use_mpt', False), mpt_variant=getattr(args, 'mpt_variant', 'full'), mpt_residual_scale=getattr(args, 'mpt_residual_scale', 0.5))
            if hasattr(net_all, 'vit_pos'):
                net_all.vit_pos.rho_com = getattr(args, 'rho_com', 0.5)
                net_all.vit_pos.rho_reg = getattr(args, 'rho_reg', 0.5)
                net_all.vit_pos.motion_tau = getattr(args, 'motion_tau', 0.7)
                net_all.vit_pos.prior_type = getattr(args, 'prior_type', 'nonparam')
                net_all.vit_pos.branch_mode = getattr(args, 'branch_mode', 'both')
                net_all.vit_pos.gate_mode = getattr(args, 'gate_mode', 'learned')
                net_all.vit_pos.fixed_gamma_com = getattr(args, 'fixed_gamma_com', 1.0)
                net_all.vit_pos.fixed_gamma_reg = getattr(args, 'fixed_gamma_reg', 1.0)

                print(f"[theory] rho_com={net_all.vit_pos.rho_com}, rho_reg={net_all.vit_pos.rho_reg}, motion_tau={net_all.vit_pos.motion_tau}, prior_type={getattr(args, 'prior_type', 'nonparam')}, branch_mode={getattr(args, 'branch_mode', 'both')}, gate_mode={getattr(args, 'gate_mode', 'learned')}, gate_type={getattr(args, 'gate_type', 'sigmoid')}")
        elif model_name=='FRAT':
            net_all=MyNet_FRAT(in_channels=3, num_classes=num_classes,depth=trans_depth,img_size=image_size,patch_size=patch_size,num_heads=trans_num_heads,use_frasa=True,embed_dim=embed_dim, use_mpt=getattr(args, 'use_mpt', False), mpt_variant=getattr(args, 'mpt_variant', 'full'), mpt_residual_scale=getattr(args, 'mpt_residual_scale', 0.5))
            if hasattr(net_all, 'vit_pos'):
                net_all.vit_pos.rho_com = getattr(args, 'rho_com', 0.5)
                net_all.vit_pos.rho_reg = getattr(args, 'rho_reg', 0.5)
                net_all.vit_pos.motion_tau = getattr(args, 'motion_tau', 0.7)
                net_all.vit_pos.prior_type = getattr(args, 'prior_type', 'nonparam')
                net_all.vit_pos.branch_mode = getattr(args, 'branch_mode', 'both')
                net_all.vit_pos.gate_mode = getattr(args, 'gate_mode', 'learned')
                net_all.vit_pos.fixed_gamma_com = getattr(args, 'fixed_gamma_com', 1.0)
                net_all.vit_pos.fixed_gamma_reg = getattr(args, 'fixed_gamma_reg', 1.0)

                print(f"[theory] rho_com={net_all.vit_pos.rho_com}, rho_reg={net_all.vit_pos.rho_reg}, motion_tau={net_all.vit_pos.motion_tau}, prior_type={getattr(args, 'prior_type', 'nonparam')}, branch_mode={getattr(args, 'branch_mode', 'both')}, gate_mode={getattr(args, 'gate_mode', 'learned')}, gate_type={getattr(args, 'gate_type', 'sigmoid')}")
        elif model_name=='en_FRAT':
            net_all=MyNet_FRAT(in_channels=3, num_classes=num_classes,depth=trans_depth,img_size=image_size,patch_size=patch_size,num_heads=trans_num_heads,use_frasa=True,embed_dim=embed_dim, use_mpt=getattr(args, 'use_mpt', False), mpt_variant=getattr(args, 'mpt_variant', 'full'), mpt_residual_scale=getattr(args, 'mpt_residual_scale', 0.5))
            if hasattr(net_all, 'vit_pos'):
                net_all.vit_pos.rho_com = getattr(args, 'rho_com', 0.5)
                net_all.vit_pos.rho_reg = getattr(args, 'rho_reg', 0.5)
                net_all.vit_pos.motion_tau = getattr(args, 'motion_tau', 0.7)
                net_all.vit_pos.prior_type = getattr(args, 'prior_type', 'nonparam')
                net_all.vit_pos.branch_mode = getattr(args, 'branch_mode', 'both')
                net_all.vit_pos.gate_mode = getattr(args, 'gate_mode', 'learned')
                net_all.vit_pos.fixed_gamma_com = getattr(args, 'fixed_gamma_com', 1.0)
                net_all.vit_pos.fixed_gamma_reg = getattr(args, 'fixed_gamma_reg', 1.0)

                print(f"[theory] rho_com={net_all.vit_pos.rho_com}, rho_reg={net_all.vit_pos.rho_reg}, motion_tau={net_all.vit_pos.motion_tau}, prior_type={getattr(args, 'prior_type', 'nonparam')}, branch_mode={getattr(args, 'branch_mode', 'both')}, gate_mode={getattr(args, 'gate_mode', 'learned')}, gate_type={getattr(args, 'gate_type', 'sigmoid')}")
            
        else:
            net_all=None

        net_all = maybe_wrap_cser(net_all, args, image_size, patch_size, embed_dim, num_classes)
        
        params_all = net_all.parameters()

        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(params_all, lr=lr, momentum=args.momentum)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(params_all, lr=lr)
        elif optimizer_name == 'adamW':
            optimizer = torch.optim.AdamW(params_all, lr=lr, weight_decay=0.6)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

        if loss_function == 'crossentropy':
            criterion = torch.nn.CrossEntropyLoss()
            # 若wetghtedCE loss
            # class_weight = torch.FloatTensor([0.2857, 0.1329, 0.1695]).cuda(args.gpuid)  # half-oversampling  [1.0, 2.2936, 3.0120]和[0.2857, 0.1329,0.1695]
            # criterion = nn.CrossEntropyLoss(weight=class_weight).cuda(args.gpuid)
        # elif loss_function == 'focal':
        #     criterion = LossFunctions.FocalLoss(class_num=classes, device=device)
        # elif loss_function == 'balanced':
        #     criterion = LossFunctions.BalancedLoss(class_num=classes, device=device)
        # elif loss_function == 'cosine':
        #     criterion = LossFunctions.CosineLoss()
            
        net_all=net_all.to(device)

        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []
        best_each_subject_labels = []

        num_samples = len(val_loader.dataset)
        predVec = torch.zeros(num_samples)
        labelVec = torch.zeros(num_samples)
        start_idx = 0
        end_idx = 0     


        for epoch in range(1,epochs+1):
            # print('\tEpoch {}/{}'.format(epoch, epochs))
            running_loss = 0.0
            running_corrects = 0  #num_train_correct
            num_train_examples = 0
            iter_cnt = 0
            
            net_all.train()

            for batch_i in train_loader:
                if use_facial_matrices:
                    flow = batch_i['flow'].cuda()
                    apex = batch_i['image_apex'].cuda()
                    class_labels = batch_i['label'].cuda()
                    A_Component = batch_i['A_Component'].cuda()
                    A_Region = batch_i['A_Region'].cuda()

                            # 扩展矩阵维度以适应批量 [B,1,8,196] -> [B,8,196]
                    A_Component = A_Component.squeeze(1)  # [B, 8, 196]
                    A_Region = A_Region.squeeze(1)

                else:
                    flow,apex,class_labels = batch_i
                    class_labels=class_labels.to(device)
                    apex=apex.to(device)
                    flow=flow.to(device)


                iter_cnt+=1
                # print(f"  flow shape: {flow.shape}")
                H, W = flow.shape[2], flow.shape[3]

                if model_name=='vit':
                    output_class= net_all(flow)

                if model_name=='en_vit':
                    lam = np.random.beta(mix_alpha, mix_alpha)
                    flow1=flow
                    index = torch.randperm(flow.size()[0]).cuda()
                    flow2=flow[index]
                    tau_m1, tau_m2, tau_d=calculate_thresholds(flow1,flow2,magnitude_percentile, direction_percentile)
                    flow_height, flow_width = flow.shape[2], flow.shape[3]
                    mask=generate_flow_mask(flow1,flow2, tau_m1, tau_m2, tau_d)
                    mask_224=torch.nn.functional.interpolate(mask,size=(flow_height, flow_width),mode='nearest')              
                    mix_flow = mask_224 * (lam * flow1 + (1 - lam) * flow2) + (1 - mask_224) * flow1
                    output_class= net_all(mix_flow)

                if model_name=="FRAT":
                    output_class= net_all(flow,A_Component,A_Region)

                if model_name=='en_FRAT':
                    lam = np.random.beta(mix_alpha, mix_alpha)
                    flow1=flow
                    index = torch.randperm(flow.size()[0]).cuda()
                    flow2=flow[index]
                    tau_m1, tau_m2, tau_d=calculate_thresholds(flow1,flow2,magnitude_percentile, direction_percentile)
                    flow_height, flow_width = flow.shape[2], flow.shape[3]
                    mask=generate_flow_mask(flow1,flow2, tau_m1, tau_m2, tau_d)
                    mask_224=torch.nn.functional.interpolate(mask,size=(flow_height, flow_width),mode='nearest')
                    mix_flow = mask_224 * (lam * flow1 + (1 - lam) * flow2) + (1 - mask_224) * flow1
                    output_class= net_all(mix_flow,A_Component,A_Region)

                
                loss_class = criterion(output_class, class_labels)

                if getattr(args, 'use_cser', False) and getattr(args, 'use_cser_counterfactual', False):
                    cser_aux_loss = getattr(net_all, 'cser_aux_loss', None)
                    if cser_aux_loss is not None:
                        loss_class = loss_class + getattr(args, 'cser_cf_weight', 0.03) * cser_aux_loss

                if getattr(args, 'theory_full', False):
                    # Full information-theoretic EN-FRAT objective:
                    # L = Lce + alpha*Lenh + beta*Lcomp + mu*Lgain + nu*Lcons.
                    output_class, aux = net_all(flow, A_Component, A_Region, return_aux=True)

                    sfm_temp = getattr(args, 'sfm_temperature', 0.0)
                    sfm_temp = None if sfm_temp is None or sfm_temp <= 0 else sfm_temp

                    if getattr(args, 'mix_strategy', 'selective') == 'global':
                        mix_flow_theory, _, sfm_aux = global_flow_mix(
                            flow,
                            class_labels,
                            alpha=getattr(args, 'sfm_alpha', 0.4),
                            same_class=getattr(args, 'sfm_same_class', False),
                        )
                    else:
                        mix_flow_theory, _, sfm_aux = selective_flow_mix(
                            flow,
                            class_labels,
                            alpha=getattr(args, 'sfm_alpha', 0.4),
                            q=getattr(args, 'sfm_q', 0.6),
                            same_class=getattr(args, 'sfm_same_class', False),
                            temperature=sfm_temp,
                        )

                    output_enh, aux_enh = net_all(
                        mix_flow_theory,
                        A_Component,
                        A_Region,
                        return_aux=True,
                    )

                    ce_loss = criterion(output_class, class_labels)
                    enhanced_loss = criterion(output_enh, class_labels)

                    aux_enh['logits'] = output_enh

                    loss_pack = compute_full_information_loss(
                        ce_loss=ce_loss,
                        enhanced_loss=enhanced_loss,
                        aux=aux,
                        aux_enhanced=aux_enh,
                        alpha=getattr(args, 'info_alpha', 1.0),
                        beta=getattr(args, 'info_beta', 0.05),
                        mu=getattr(args, 'info_mu', 0.05),
                        nu=getattr(args, 'info_nu', 0.1),
                    )

                    loss_class = loss_pack['loss']
                
                optimizer.zero_grad()
                loss_class.backward()
                optimizer.step()

                running_loss += loss_class.item() * flow.size(0)
                _, predicted = torch.max(output_class, 1)
                running_corrects += torch.sum(predicted == class_labels)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / float(len(train_loader.dataset))
            print('[Epoch %d] Training accuracy: %.4f. Training Loss: %.3f' % (epoch, epoch_acc, epoch_loss))

            # batch_loss = running_loss / iter_cnt
            # print('\t{}Epoch Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))

            net_all.eval()
            all_predictions = []  # 当前epoch的所有预测
            all_labels = []       # 当前epoch的所有标签

            with torch.no_grad():  # 禁用梯度计算
                val_loss=0.0
                num_val_correct = 0
                num_val_examples = 0

                pos_label = torch.zeros(num_classes)
                pos_pred = torch.zeros(num_classes)
                TP = torch.zeros(num_classes)

                running_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                sample_cnt = 0


                for batch_i in val_loader:
                    if use_facial_matrices:
                        flow = batch_i['flow'].cuda()
                        apex = batch_i['image_apex'].cuda()
                        class_labels = batch_i['label'].cuda()
                        A_Component = batch_i['A_Component'].cuda()
                        A_Region = batch_i['A_Region'].cuda()
                        A_Component = A_Component.squeeze(1)  # [B, 8, 196]
                        A_Region = A_Region.squeeze(1)

                    else:
                        flow,apex,class_labels = batch_i
                        class_labels=class_labels.to(device)
                        apex=apex.to(device)
                        flow=flow.to(device)

                    if model_name=='vit':
                        output_class= net_all(flow)

                    if model_name=='en_vit':
                        output_class=net_all(flow)

                    if model_name=='FRAT':
                        output_class= net_all(flow,A_Component,A_Region)

                    if model_name=='en_FRAT':
                        output_class= net_all(flow,A_Component,A_Region)

                    

                    loss_class = criterion(output_class, class_labels)
                    val_loss+=loss_class.data.item()*flow.size(0)

                    running_loss += loss_class
                    iter_cnt += 1

                    _, predicts = torch.max(output_class, 1)

                    # 关键：收集所有batch的数据
                    all_predictions.extend(predicts.cpu().tolist())
                    all_labels.extend(class_labels.cpu().tolist())

                    num_val_correct += (torch.max(output_class, 1)[1] == class_labels).sum().item()
                    num_val_examples += class_labels.size(0)  

                    correct_num = torch.eq(predicts,class_labels)
                    bingo_cnt += correct_num.sum().cpu()
                    sample_cnt += output_class.size(0)

                    for cls in range(num_classes):

                        for element in predicts:
                            if element == cls:
                                pos_label[cls] = pos_label[cls] + 1
                        for element in class_labels:
                            if element == cls:
                                pos_pred[cls] = pos_pred[cls] + 1
                        for elementp, elementl in zip(predicts, class_labels):
                            if elementp == elementl and elementp == cls:
                                TP[cls] = TP[cls] + 1

                    count = 0
                    SUM_F1 = 0
                    for index in range(num_classes):
                        if pos_label[index] != 0 or pos_pred[index] != 0:
                            count = count + 1
                            SUM_F1 = SUM_F1 + 2 * TP[index] / (pos_pred[index] + pos_label[index])
                    if count == 0:
                        print(f"[WARN] Empty class support in validation epoch: db={db}, subj={subj}. Skip this epoch.")
                        continue
                    AVG_F1 = SUM_F1 / count

                if num_val_examples == 0:
                    print(f"[WARN] Empty validation fold: db={db}, subj={subj}. Skip this fold.")
                    continue
                val_acc = num_val_correct / num_val_examples
                val_loss = val_loss / len(val_loader.dataset) 

                temp_best_each_subject_pred = []
                if best_accuracy_for_each_subject<=val_acc:
                    best_accuracy_for_each_subject=val_acc
                    # temp_best_each_subject_pred.extend(torch.max(output_class, 1)[1].tolist())
                    best_each_subject_pred = all_predictions.copy()
                    best_each_subject_labels = all_labels.copy()

                    # 保存模型参数
                    torch.save(net_all.state_dict(), weight_path)
            
                print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (epoch, val_acc, val_loss))

                running_loss = running_loss / iter_cnt
                acc = bingo_cnt.float() / float(sample_cnt)
                acc = np.around(acc.numpy(), 4)
                if bingo_cnt > max_corr:
                    max_corr = bingo_cnt
                    max_epoch = epoch
                if AVG_F1 >= max_f1:
                    max_f1 = AVG_F1
                    max_pos_label = pos_label
                    max_pos_pred = pos_pred
                    max_TP = TP
                print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f, F1-score:%.3f" % (epoch, acc, running_loss, AVG_F1))
                if val_acc==1.:
                    print('achieve 100%acc, break')
                    break 

                # if epoch <= 50:
                #     scheduler.step()                    



        print("评价1：--------------------------------------------------------------------------------------------------")
        # For UF1 and UAR computation
        print('Best Predicted    :', best_each_subject_pred)
        accuracydict = {}
        accuracydict['pred'] = best_each_subject_pred
        accuracydict['truth'] = best_each_subject_labels
        all_accuracy_dict[subj] = accuracydict

        print('Ground Truth :', best_each_subject_labels)
        print('Evaluation until this subject: ')
        total_pred.extend(best_each_subject_pred)
        total_gt.extend(best_each_subject_labels)
        best_total_pred.extend(best_each_subject_pred)
        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, show=True)
        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4)) 

        print("评价2：--------------------------------------------------------------------------------------------------") 
        end_idx = start_idx + len(val_dataset)
        predVec[start_idx:end_idx] = torch.tensor(best_each_subject_pred).to(device)
        labelVec[start_idx:end_idx] = torch.tensor(best_each_subject_labels).to(device)
        start_idx =end_idx
        preds_np = np.array(predVec)
        labels_np = np.array(labelVec)
        allRST.append([preds_np, labels_np])
        acc = torch.sum(predVec == labelVec).double() / len(predVec)
        print('\tSubject {} has the accuracy:{:.4f}\n'.format(subj, acc))

        preds_db['all'] = torch.cat((preds_db['all'], predVec), 0)
        labels_db['all'] = torch.cat((labels_db['all'], labelVec), 0)
        if combined:
            if str(subj).find('sub') != -1:
                preds_db['casme2'] = torch.cat((preds_db['casme2'], predVec), 0)
                labels_db['casme2'] = torch.cat((labels_db['casme2'], labelVec), 0)
            else:
                if str(subj).find('s') != -1:
                    preds_db['smic'] = torch.cat((preds_db['smic'], predVec), 0)
                    labels_db['smic'] = torch.cat((labels_db['smic'], labelVec), 0)
                else:
                    preds_db['samm'] = torch.cat((preds_db['samm'], predVec), 0)
                    labels_db['samm'] = torch.cat((labels_db['samm'], labelVec), 0)


        print("评价3：--------------------------------------------------------------------------------------------------")
        num_sum = num_sum + max_corr
        pos_label_ALL = pos_label_ALL + max_pos_label
        pos_pred_ALL = pos_pred_ALL + max_pos_pred
        TP_ALL = TP_ALL + max_TP
        count = 0
        SUM_F1 = 0
        for index in range(num_classes):
            if pos_label_ALL[index] != 0 or pos_pred_ALL[index] != 0:
                count = count + 1
                SUM_F1 = SUM_F1 + 2 * TP_ALL[index] / (pos_pred_ALL[index] + pos_label_ALL[index])

        if count == 0:
            print(f"[WARN] No valid fold contributed to final metrics: db={db}.")
            return
        F1_ALL = SUM_F1 / count
        val_now = val_now + val_dataset.__len__()
        if len(val_dataset) > 0:
            acc_now=max_corr/val_dataset.__len__()
        else:
            acc_now=0

        # 确保 acc_now 是 Python 数字而不是 Tensor
        if isinstance(acc_now, torch.Tensor):
            acc_now_value = acc_now.item()
        else:
            acc_now_value = acc_now
        
        #writer.add_scalar('trans_acc', acc_now, subj)
        print("[subject %s] correct_num:%d sum:%d ACC: %.4f  " % (subj, max_corr, val_dataset.__len__(),acc_now))
        print("[ALL_corr]: %d [ALL_val]: %d [ALL_ACC]:%.4f" % (int(num_sum), int(val_now), num_sum / val_now))
        print("[F1_now]: %.4f [F1_ALL]: %.4f" % (max_f1, F1_ALL))
        print('max_epoch:', str(max_epoch))
        acc_list.append(round(acc_now_value,4))
        epoch_list.append(str(max_epoch))

    print("最终评价1：--------------------------------------------------------------------------------------------------")
    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    print(np.shape(total_gt))
    print(all_accuracy_dict)

    wv4_save_dir = os.environ.get("WV4_SAVE_DIR", "").strip()
    if wv4_save_dir:
        import csv
        import json
        from pathlib import Path as _Path

        out_dir = _Path(wv4_save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pred_csv = out_dir / "predictions.csv"
        rows = []
        global_index = 0

        for subj_key, acc_dict in all_accuracy_dict.items():
            subj_preds = acc_dict.get("pred", [])
            subj_truth = acc_dict.get("truth", [])

            for local_index, (yt, yp) in enumerate(zip(subj_truth, subj_preds)):
                rows.append({
                    "global_index": global_index,
                    "subject": str(subj_key),
                    "sample_in_subject": local_index,
                    "y_true": int(yt),
                    "y_pred": int(yp),
                })
                global_index += 1

        with pred_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["global_index", "subject", "sample_in_subject", "y_true", "y_pred"]
            )
            writer.writeheader()
            writer.writerows(rows)

        meta = {
            "num_samples": len(rows),
            "num_subjects": len(all_accuracy_dict),
            "note": "Predictions saved from best_each_subject_pred and best_each_subject_labels in main_theory.py."
        }
        (out_dir / "predictions_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print("[WV4_SAVE] predictions:", pred_csv)
        print("[WV4_SAVE] num_samples:", len(rows))

    print("最终评价2：--------------------------------------------------------------------------------------------------")
    eval_acc = metrics2.accuracy()
    eval_f1 = metrics2.f1score()
    acc_w, acc_uw = eval_acc.eval(preds_db['all'], labels_db['all'])
    f1_w, f1_uw = eval_f1.eval(preds_db['all'], labels_db['all'])
    print('\nThe dataset has the ACC:{:.4f} and UAR and UF1:{:.4f} and {:.4f}'.format(acc_w, acc_uw, f1_uw))
    if combined:
        # casme2
        if preds_db['casme2'].nelement() != 0:
            acc_w, acc_uw = eval_acc.eval(preds_db['casme2'], labels_db['casme2'])
            f1_w, f1_uw = eval_f1.eval(preds_db['casme2'], labels_db['casme2'])
            print('\nThe casme2 dataset has the ACC:{:.4f} and UAR and UF1:{:.4f} and {:.4f}'.format(acc_w, acc_uw, f1_uw))
            
        # smic
        if preds_db['smic'].nelement() != 0:
            acc_w, acc_uw = eval_acc.eval(preds_db['smic'], labels_db['smic'])
            f1_w, f1_uw = eval_f1.eval(preds_db['smic'], labels_db['smic'])
            print('\nThe smic dataset has the ACC:{:.4f} and UAR and UF1:{:.4f} and {:.4f}'.format(acc_w, acc_uw, f1_uw))
            
        # samm
        if preds_db['samm'].nelement() != 0:
            acc_w, acc_uw = eval_acc.eval(preds_db['samm'], labels_db['samm'])
            f1_w, f1_uw = eval_f1.eval(preds_db['samm'], labels_db['samm'])
            print('\nThe samm dataset has the ACC:{:.4f} and UAR and UF1:{:.4f} and {:.4f}'.format(acc_w, acc_uw, f1_uw))

    print("最终评价3：--------------------------------------------------------------------------------------------------")
    print("Acc_list:", acc_list)
    print('max epoches:', epoch_list)
    print("Total acc: %.4f"% (num_sum * 1.0 / val_now))
        
    # writing parameters into log file
    print("\n" + "="*40)
    print("🏷️ 实验参数配置".center(40))
    print("="*40)

    # 1. 路径参数
    print("\n📂 路径参数:")
    print(f"  RAF数据集路径: {args.raf_path}")
    print(f"  数据库选择: {args.db}")
    print(f"  检查点文件: {args.checkpoint}")
    print(f"  预训练权重: {args.pretrained}")

    # 2. 模型架构参数
    print("\n🏗️ 模型架构参数:")
    print(f"  模型名称: {args.model_name}")
    print(f"  分类数: {args.num_classes}")
    print(f"  Transformer深度: {args.trans_depth}")
    print(f"  Transformer头数: {args.trans_num_heads}")
    print(f"  嵌入维度: {args.embed_dim}")
    print(f"  图像尺寸: {args.image_size}")
    print(f"  Patch尺寸: {args.patch_size}")
    print(f"  Dropout率: {args.drop_rate}")
    print(f"  使用面部矩阵: {'是' if args.use_facial_matrices else '否'}")
    print(f"  多数据集: {'是' if args.combined else '否'}")

    # 3. 训练参数
    print("\n⚙️ 训练参数:")
    print(f"  批量大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  优化器: {args.optimizer_name}")
    print(f"  损失函数: {args.loss_function}")
    print(f"  Momentum: {args.momentum}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  Gamma: {args.gamma}")
    print(f"  数据加载线程数: {args.workers}")

    # 4. 数据增强参数
    print("\n🔄 数据增强参数:")
    print(f"  PatchUp概率: {args.patchup_prob}")
    print(f"  Mix Alpha: {args.mix_alpha}")
    print(f"  Theory Full Objective: {getattr(args, 'theory_full', False)}")
    print(f"  Info Loss Weights: alpha={getattr(args, 'info_alpha', None)}, beta={getattr(args, 'info_beta', None)}, mu={getattr(args, 'info_mu', None)}, nu={getattr(args, 'info_nu', None)}")
    print(f"  Selective Flow-Mix: alpha={getattr(args, 'sfm_alpha', None)}, q={getattr(args, 'sfm_q', None)}, same_class={getattr(args, 'sfm_same_class', False)}")
    print(f"  Ostu: {'是' if args.ostu else '否'}")
    print(f"  幅度百分位数: {args.magnitude_percentile}")
    print(f"  方向百分位数: {args.direction_percentile}")

    print("="*40 + "\n")

    end_time = time.time()
    print("start_time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print("end_time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))) 

if __name__ == '__main__':
    seed=2
    seed_torch(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
   
         
