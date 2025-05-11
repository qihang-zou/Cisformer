import pandas as pd
import random
import numpy as np
from torch.utils.data import Dataset
import torch
import os
import gc
from accelerate import Accelerator
from functools import partial
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

# helper

def select_least_used_gpu():
    """自动选择当前GPU占用最少的一张卡，无GPU则使用CPU

    Usages:
        device = select_least_used_gpu()
    """
    if torch.cuda.is_available():
        # 获取所有GPU的数量
        num_gpus = torch.cuda.device_count()
        # 获取每个GPU的已用内存
        memory_allocated = [torch.cuda.memory_allocated(device) for device in range(num_gpus)]
        # 选择已用内存最少的GPU
        least_used_gpu = memory_allocated.index(min(memory_allocated))
        device = torch.device(f'cuda:{least_used_gpu}')
        print(f'Selected GPU: {least_used_gpu}')
        return device
    else:
        device = torch.device('cpu')
        print('No GPU available, using CPU')
        return device

def setup_seed(seed):
    #--- Fix random seed ---#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def exists(val):
    return val is not None

def comb_sublist(lst):
    out_lst = []
    for sub_lst in lst:
        if isinstance(sub_lst, (list, tuple)):
            out_lst += comb_sublist(sub_lst)
        else:
            out_lst.append(sub_lst)
    return out_lst


# DDP
class New_Accelerator(Accelerator):
    def free_dataloader(self):
        self._dataloaders = []
        gc.collect() 
        torch.cuda.empty_cache() 
        
class Accelerate_early_stop_helper():
    """
    Early stop helper for Accelerate parrallel framework.
    
    Args:
        accelerator: Initialed Accelerator from accelerate package.
        patience: Patience for stop condition.
        
    Usage:
        stopper = Accelerate_early_stop_helper(accelerator)
        if (condition for early stop):
            stopper.set()
        if stopper.should_break():
            break
    """
    def __init__(self, accelerator, patience = 1):
        self.accelerator = accelerator
        self.should_stop = torch.tensor([False]).to(self.accelerator.device)
        self.patience = patience
        self.raw_patience = patience
   
    def set(self):
        self.should_stop = torch.tensor([True]).to(self.accelerator.device)
    
    def should_break(self):
        multi_process_should_stop = self.accelerator.gather(self.should_stop)
        self.should_stop = torch.tensor([False]).to(self.accelerator.device)
        if all(multi_process_should_stop):
            self.patience -= 1
        else:
            self.patience = self.raw_patience
        if self.patience <= 0:
            return True
        else:
            return False

########################## data process ###################################

def load_data_path(files_dir, endwith="", greed=False, without=None):
    """_summary_
    Load file with same extension

    Args:
        files_dir (str): The main directory
        endwith (str, optional): File extension. Defaults to "".
        greed (bool, optional): Load file form child directory. Defaults to False.
        without (str, optional): Files will be ignored if its name contain this parameter. Defaults to None.
    """
    
    def finding(files_dir, endwith, greed):
        data_paths = []
        items = os.listdir(files_dir)
        dirs = [item for item in items if os.path.isdir(os.path.join(files_dir, item))]
        files = [item for item in items if not os.path.isdir(os.path.join(files_dir, item))]

        for file in files:
            if file.endswith(endwith):
                data_paths.append(os.path.join(files_dir, file))
        if not greed:
            return data_paths

        for dir in dirs:
            data_paths += finding(os.path.join(files_dir, dir), endwith, greed)
        return data_paths

    def clear(lst, string):
        return [x for x in lst if string not in x]

    output = finding(files_dir, endwith, greed)
    if isinstance(without, (list, tuple)):
        for each in without:
            output = clear(output, each)
    elif isinstance(without, str):
        output = clear(output, without)
    return output

# 上限7位数
def list_digit(x, digit_number = 7):
    x = str(x)
    x = np.array([int(each) for each in x])
    x = np.pad(x, (digit_number - len(x), 0), mode = "constant", constant_values = 0)
    return x

def list_digit_back(x):
    out = ""
    for each in x:
        out += str(int(each))
    return int(out)

def binning(x, max_value, num_bins = 64):
    if x == 0:
        return 0
    elif x > 0:
        bin_width = max_value / num_bins
        bin_index = min(int(x / bin_width), num_bins - 1)
        return bin_index + 1

class PairDataset(Dataset): 
    """
    Special tokens: {<PAD>: 0}. Only used at enc modality. 
    Zero express token will be considered as <PAD> at enc modality.
    Random multiplication. Multiple for the dec modality.
    If input enc_len or dec_len is longer than gene or peak list, it will be set to the length of gene or peak list.
    
    Output:
    rna_sequence, rna_value, atac_sequence, atac_value, enc_pad_mask
    
    Shape of output:
    rna_sequence: multiple, sequence_len
    rna_value: multiple, sequence_len
    atac_sequence: multiple, sequence_len, digit(7)
    atac_value: multiple, sequence_len
    enc_pad_mask: multiple, sequence_len
    
    Args:
        rna_matrix (sparse matrix or np.array): Count matrix
        atac_matrix (sparse matrix or np.array): Count matrix
        enc_len (int or "whole"): Input length of enc modality
        dec_len (int or "whole"): Input length of dec modality. If set to "whole", multiple will be set to 1 automatically.
        max_express (int): The max expression
        multiple (int): The number of rate of random multiplication
        atac2rna (bool): True for ATAC->RNA, False for ATAC->RNA
        is_raw_count: If set False (which is for normalized data), value will be binning.
    """
    def __init__(self, rna_adata, atac_adata, enc_len, dec_len, max_express = 64, multiple = 100, atac2rna = True, is_raw_count = False):
        super().__init__()
        assert enc_len == "whole" or type(enc_len) == int, 'enc_len must be int or "whole"'
        assert dec_len == "whole" or type(dec_len) == int, 'dec_len must be int or "whole"'
        self.rna_adata = rna_adata
        self.atac_adata = atac_adata
        self.rna_matrix = self.rna_adata.X
        self.atac_matrix = self.atac_adata.X
        self.rna_len = dec_len if atac2rna else enc_len
        self.atac_len = enc_len if atac2rna else dec_len
        self.rna_len = min(self.rna_len, self.rna_matrix.shape[1]) if type(self.rna_len) == int else self.rna_len
        self.atac_len = min(self.atac_len, self.atac_matrix.shape[1]) if type(self.atac_len) == int else self.atac_len
        self.max_express = max_express
        self.multiple = 1 if dec_len == "whole" else multiple
        self.is_raw_count = is_raw_count
        self.atac2rna = atac2rna
        # self.obs = pd.merge(rna_adata.obs, atac_adata.obs, left_index=True, right_index=True, suffixes=("_rna", "_atac"))
        # self.idx = []
        
        # self.enc_rna_nozero_rate = 0.8
        # self.dec_rna_one_rate = 0.4
        # self.dec_rna_above_one_rate = 0.5
        # self.enc_atac_one_rate = 0.8
        self.dec_atac_one_rate = 0.5

    def __getitem__(self, index):
        # self.idx.append(index)
        try:
            rna_value = self.rna_matrix[index].toarray().squeeze()
        except:
            rna_value = self.rna_matrix[index].squeeze()
        try:
            atac_value = self.atac_matrix[index].toarray().squeeze().astype(int)
        except:
            atac_value = self.atac_matrix[index].squeeze().astype(int)
        # clip expression
        if self.is_raw_count:
            rna_value = rna_value.astype(int)
            rna_value = np.clip(rna_value, None, self.max_express)
        else:
            partial_binning = partial(binning, max_value = rna_value.max(), num_bins = self.max_express)
            rna_value = np.apply_along_axis(partial_binning, 0, np.array([rna_value])).squeeze().astype(int)
        
        rna_idx = np.array(range(len(rna_value)))
        rna_nonzero_idx = rna_idx[rna_value != 0].tolist()
        # rna_zero_idx = rna_idx[rna_value == 0].tolist()
        # rna_one_idx = rna_idx[rna_value == 1].tolist()
        # rna_above_one_idx = rna_idx[rna_value > 1].tolist()

        atac_idx = np.array(range(len(atac_value)))
        atac_one_idx = atac_idx[atac_value == 1].tolist()
        atac_zero_idx = atac_idx[atac_value == 0].tolist()
        
        out_rna_sequence = []
        out_rna_value = []
        out_atac_sequence = []
        out_atac_value = []
        out_enc_pad_mask = []
        
        # multiple_loop, only for dec modality
        for _ in range(self.multiple):
            # enc: ATAC| dec: RNA
            if self.atac2rna:
                # RNA
                if type(self.rna_len) == int:
                    # 方案一
                    # num_one = round(self.rna_len * self.dec_rna_one_rate)
                    # num_above_one = round(self.rna_len * self.dec_rna_above_one_rate)
                    # num_one = min(num_one, len(rna_one_idx))
                    # num_above_one = min(num_above_one, len(rna_above_one_idx))
                    # num_zero = self.rna_len - num_one - num_above_one
                    # new_rna_one_idx = random.sample(rna_one_idx, k=num_one)
                    # new_rna_above_one_idx = random.sample(rna_above_one_idx, k=num_above_one)
                    # if num_zero <= len(rna_zero_idx):
                    #     new_rna_zero_idx = random.sample(rna_zero_idx, k=num_zero)
                    # else:
                    #     new_rna_zero_idx = random.choices(rna_zero_idx, k=num_zero)
                    # new_rna_idx = new_rna_one_idx + new_rna_above_one_idx + new_rna_zero_idx
                    # 方案二：表达位点加随机0位点
                    # new_rna_nozero_idx = rna_nonzero_idx.copy()
                    # if len(new_rna_nozero_idx) >= self.rna_len:
                    #     new_rna_idx = random.sample(new_rna_nozero_idx, self.rna_len)
                    # else:
                    #     num_zero = self.rna_len - len(new_rna_nozero_idx)
                    #     if num_zero <= len(rna_zero_idx):
                    #         new_rna_zero_idx = random.sample(rna_zero_idx, k=num_zero)
                    #     else:
                    #         new_rna_zero_idx = random.choices(rna_zero_idx, k=num_zero)
                    #     new_rna_idx = new_rna_nozero_idx + new_rna_zero_idx
                    # 方案三: 完全随机抓
                    # new_rna_idx = rna_idx.tolist()
                    # if len(new_rna_idx) > self.rna_len:
                    #     new_rna_idx = random.sample(new_rna_idx, k=self.rna_len)
                    # 方案四: 仅有表达位点
                    new_rna_idx = rna_nonzero_idx.copy()
                    if len(new_rna_idx) > self.rna_len:
                        new_rna_idx = random.sample(new_rna_idx, self.rna_len)
                else:
                    new_rna_idx = rna_idx.tolist()
                    # self.rna_len = len(new_rna_idx)
                new_rna_value = rna_value[new_rna_idx].copy()
                new_rna_sequence = np.array(new_rna_idx)
                # 方案四：
                # new_rna_value += 1 # add special token <PAD>
                new_rna_sequence += 1 # add special token <PAD>
                if type(self.rna_len) == int:
                    # pad , consider 0 expression as <PAD>
                    new_rna_value = np.pad(new_rna_value, (0, self.rna_len - len(new_rna_value)), mode = "constant", constant_values = 0) # pad
                    new_rna_sequence = np.pad(new_rna_sequence, (0, self.rna_len - len(new_rna_sequence)), mode = "constant", constant_values = 0) # pad
                    
                # ATAC   
                # 方案一：只取有效位点 
                if type(self.atac_len) == int:
                    # # consider 0 expressions
                    # num_one = round(self.atac_len * self.enc_atac_one_rate)
                    # num_one = min(num_one, len(atac_one_idx))
                    # num_zero = self.atac_len - num_one
                    # new_atac_one_idx = random.sample(atac_one_idx, k = num_one)
                    # if num_zero <= len(atac_zero_idx):
                    #     new_atac_zero_idx = random.sample(atac_zero_idx, k = num_zero)
                    # else:
                    #     new_atac_zero_idx = random.choices(atac_zero_idx, k = num_zero)
                    # new_atac_idx = new_atac_one_idx + new_atac_zero_idx
                    
                    # consider only expressed tokens
                    new_atac_idx = atac_one_idx.copy()
                    if len(new_atac_idx) > self.atac_len:
                        new_atac_idx = random.sample(new_atac_idx, self.atac_len)
                else:
                    new_atac_idx = atac_idx.tolist()
                    # self.atac_len = len(new_atac_idx)
                new_atac_value = atac_value[new_atac_idx].copy()
                new_atac_sequence = np.array(new_atac_idx)
                # new_atac_value += 1 # add special token <PAD>
                new_atac_sequence += 1 # add special token <PAD>
                # pad, consider 0 expression as <PAD>
                if type(self.atac_len) == int:
                    new_atac_value = np.pad(new_atac_value, (0, self.atac_len - len(new_atac_value)), mode = "constant", constant_values = 0) # pad
                    new_atac_sequence = np.pad(new_atac_sequence, (0, self.atac_len - len(new_atac_sequence)), mode = "constant", constant_values = 0) # pad
                # pad mask
                enc_pad_mask = new_atac_value != 0
                
                # 方案二：一半0一半1
                # if type(self.atac_len) == int:
                #     num_one = round(self.atac_len * self.dec_atac_one_rate)
                #     num_one = min(num_one, len(atac_one_idx))
                #     num_zero = self.atac_len - num_one
                #     new_atac_one_idx = random.sample(atac_one_idx, k = num_one)
                #     if num_zero <= len(atac_zero_idx):
                #         new_atac_zero_idx = random.sample(atac_zero_idx, k = num_zero)
                #     else:
                #         new_atac_zero_idx = random.choices(atac_zero_idx, k = num_zero)
                #     new_atac_idx = new_atac_one_idx + new_atac_zero_idx
                # else:
                #     new_atac_idx = atac_idx.tolist()
                #     # self.atac_len = len(new_atac_idx)
                # new_atac_value = atac_value[new_atac_idx].copy()
                # new_atac_sequence = np.array(new_atac_idx)
                # # pad mask
                # enc_pad_mask = np.ones_like(new_atac_sequence, dtype=bool)
                
            # enc: RNA| dec: ATAC
            else:
                # RNA
                if type(self.rna_len) == int:
                    # # consider 0 expressions
                    # num_nonzero = round(self.rna_len * self.enc_rna_nozero_rate)
                    # num_nonzero = min(num_nonzero, len(rna_nonzero_idx))
                    # num_zero = self.rna_len - num_nonzero
                    # new_rna_nonzero_idx = random.sample(rna_nonzero_idx, k=num_nonzero)
                    # if num_zero <= len(rna_zero_idx):
                    #     new_rna_zero_idx = random.sample(rna_zero_idx, k=num_zero)
                    # else:
                    #     new_rna_zero_idx = random.choices(rna_zero_idx, k=num_zero)
                    # new_rna_idx = new_rna_nonzero_idx + new_rna_zero_idx
                    
                    # consider only expressed tokens
                    new_rna_idx = rna_nonzero_idx.copy()
                    if len(new_rna_idx) > self.rna_len:
                        new_rna_idx = random.sample(new_rna_idx, self.rna_len)
                else:
                    new_rna_idx = rna_idx.tolist()
                    # self.rna_len = len(new_rna_idx)
                new_rna_value = rna_value[new_rna_idx].copy()
                new_rna_sequence = np.array(new_rna_idx)
                # new_rna_value += 1 # add special token <PAD>
                new_rna_sequence += 1 # add special token <PAD>
                # pad , consider 0 expression as <PAD>
                if type(self.rna_len) == int:
                    # pad , consider 0 expression as <PAD>
                    new_rna_value = np.pad(new_rna_value, (0, self.rna_len - len(new_rna_value)), mode = "constant", constant_values = 0) # pad
                    new_rna_sequence = np.pad(new_rna_sequence, (0, self.rna_len - len(new_rna_sequence)), mode = "constant", constant_values = 0) # pad
                # pad mask
                enc_pad_mask = new_rna_value != 0
                
                # ATAC
                if type(self.atac_len) == int:
                    num_one = round(self.atac_len * self.dec_atac_one_rate)
                    num_one = min(num_one, len(atac_one_idx))
                    num_zero = self.atac_len - num_one
                    new_atac_one_idx = random.sample(atac_one_idx, k = num_one)
                    if num_zero <= len(atac_zero_idx):
                        new_atac_zero_idx = random.sample(atac_zero_idx, k = num_zero)
                    else:
                        new_atac_zero_idx = random.choices(atac_zero_idx, k = num_zero)
                    new_atac_idx = new_atac_one_idx + new_atac_zero_idx
                else:
                    new_atac_idx = atac_idx.tolist()
                    # self.atac_len = len(new_atac_idx)
                new_atac_value = atac_value[new_atac_idx].copy()
                new_atac_sequence = np.array(new_atac_idx)
                
            # listing digit | iConv for ATAC
            new_atac_sequence = np.array([list_digit(each) for each in new_atac_sequence])

            out_rna_sequence.append(new_rna_sequence)
            out_rna_value.append(new_rna_value)
            out_atac_sequence.append(new_atac_sequence)
            out_atac_value.append(new_atac_value)
            out_enc_pad_mask.append(enc_pad_mask)
            
        return np.array(out_rna_sequence), np.array(out_rna_value), np.array(out_atac_sequence), np.array(out_atac_value), np.array(out_enc_pad_mask)
    
    def __len__(self):
        return self.rna_matrix.shape[0]
    
class PreDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        # rna_value: b, n
        # rna_sequence: b, n, 6
        # rna_pad_mask: b, n
        # atac_sequence: b, n, 6
        # mask_pos: b, n * 0.2
        # atac_pad_mask: b, n
        
    def __getitem__(self, index):
        output = [each[index] for each in self.data]
        return tuple(output)
        
    def __len__(self):
        return len(self.data[0])
    
class Rna2atacDataset(Dataset):
    """
    Special tokens: {<PAD>: 0}. Only used at enc modality. 
    Zero express token will be considered as <PAD> at enc modality.
    If input enc_len is longer than gene list, it will be set to the length of gene list.
    
    Output:
    rna_sequence, rna_value, atac_sequence, enc_pad_mask
    
    Shape of output:
    rna_sequence: sequence_len
    rna_value: sequence_len
    atac_sequence: sequence_len, digit(7)
    enc_pad_mask: sequence_len
    """
    def __init__(self, rna_adata, enc_len, atac_sequence_idx, max_express=64, is_raw_count=True):
        super().__init__()
        self.rna_adata = rna_adata
        self.rna_matrix = self.rna_adata.X
        self.rna_len = enc_len
        self.max_express = max_express
        self.is_raw_count = is_raw_count
        self.atac_sequence_idx = atac_sequence_idx
        
    def __getitem__(self, index):
        try:
            rna_value = self.rna_matrix[index].toarray().squeeze()
        except:
            rna_value = self.rna_matrix[index].squeeze()
         # clip expression
        if self.is_raw_count:
            rna_value = rna_value.astype(int)
            rna_value = np.clip(rna_value, None, self.max_express)
        else:
            partial_binning = partial(binning, max_value = rna_value.max(), num_bins = self.max_express)
            rna_value = np.apply_along_axis(partial_binning, 0, np.array([rna_value])).squeeze().astype(int)
        rna_idx = np.array(range(len(rna_value)))
        rna_nonzero_idx = rna_idx[rna_value != 0].tolist()
        new_rna_idx = rna_nonzero_idx.copy()
        if len(new_rna_idx) > self.rna_len:
            new_rna_idx = random.sample(new_rna_idx, self.rna_len)
        new_rna_value = rna_value[new_rna_idx].copy()
        new_rna_sequence = np.array(new_rna_idx)
        # new_rna_value += 1 # add special token <PAD>
        new_rna_sequence += 1 # add special token <PAD>
        # pad
        new_rna_value = np.pad(new_rna_value, (0, self.rna_len - len(new_rna_value)), mode = "constant", constant_values = 0) # pad
        new_rna_sequence = np.pad(new_rna_sequence, (0, self.rna_len - len(new_rna_sequence)), mode = "constant", constant_values = 0) # pad
        # pad mask
        enc_pad_mask = new_rna_value != 0
        # atac
        new_atac_sequence = np.array([list_digit(each) for each in self.atac_sequence_idx])
        return new_rna_sequence, new_rna_value, new_atac_sequence, enc_pad_mask

    def __len__(self):
        return self.rna_matrix.shape[0]

############ plot #############
def plot_attn_weight(attns, cell_names=None, xlabels=None, ylabels=None, up_percentile=95, down_percentile=5):
    """This func is used to plot heatmap for attention weights generated from M2M.generate_attn_weight()

    Args:
        attn (list or tensor): _description_
        cell_names (list of str): Names for each cell.
        xlabels (list of lists, optional): _description_. Defaults to None.
        ylabels (list of lists, optional): _description_. Defaults to None.
        up_percentile (int, optional): 最大值取到第几个分位数. Defaults to 95.
        down_percentile (int, optional): 最小值取到第几个分位数. Defaults to 5.
    """
    if xlabels is None:
        xlabels = [xlabels] * len(attns)
    if ylabels is None:
        ylabels = [ylabels] * len(attns)
    if cell_names is None or type(cell_names) == str:
        cell_names = [cell_names] * len(attns)
    if type(attns) != list:
        attns = [attns]
        xlabels = [xlabels]
        ylabels = [ylabels]
    items = zip(attns, cell_names, xlabels, ylabels)
    for attn, cell_name, xlabel, ylabel in items:
        numpy_matrix = attn.numpy() * (1/(attn.mean().item())) # normalize到均值为1
        # 设置数值范围，通过clip思想使用数据的分位数来去除极端值
        vmin = np.percentile(numpy_matrix, down_percentile)  # 下边界设为第5个分位数
        vmax = np.percentile(numpy_matrix, up_percentile)  # 上边界设为第95个分位数
        # 使用Seaborn绘制热图
        plt.figure(figsize=(8, 6))
        sns.heatmap(numpy_matrix, annot=False, cmap='viridis', vmin=vmin, vmax=vmax)
        title = 'Attention Weight' if cell_name is None else f'Attention Weight of {cell_name}'
        plt.title(title)
        # 添加刻度
        if exists(xlabel):
            plt.xticks(ticks=np.arange(len(xlabel)), labels=xlabel, rotation=60)
        if exists(ylabel):
            plt.yticks(ticks=np.arange(len(ylabel)), labels=ylabel, rotation=0)

        plt.show()