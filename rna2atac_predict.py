import sys
import tqdm
import argparse
import os
import yaml

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

sys.path.append("M2Mmodel")

from M2Mmodel.M2M import M2M_rna2atac #Luz from M2M import M2M_rna2atac
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, roc_auc_score

# DDP 
# import New_Accelerate as Accelerator
from M2Mmodel.utils import * #Luz from utils import *

# 调试工具：梯度异常会对对应位置报错
torch.autograd.set_detect_anomaly = True
# 设置多线程文件系统
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    
    parser = argparse.ArgumentParser(description='evaluating script of M2M') #Luz parser = argparse.ArgumentParser(description='pretrain script of M2M')
    
    # path
    parser.add_argument('-d', '--data_dir', type=str, default=None, help="dir of preprocessed paired data")
    parser.add_argument('-s', '--save', type=str, default="save", help="where tht model's state_dict should be saved at")
    parser.add_argument('-n', '--name', type=str, help="the name of this project")
    parser.add_argument('-c', '--config_file', type=str, default="rna2atac_config.yaml", help="config file for model parameters")
    parser.add_argument('-l', '--load', type=str, default=None, help="if set, load previous model, path to previous model state_dict")
    # parser.add_argument('--iconv_weight', type=str, default=None, help='Path ot pretrained iConv weight')
    
    args = parser.parse_args()
    
    # parameters
    data_dir = args.data_dir
    save_path = args.save
    save_name = args.name
    saved_model_path = args.load
    config_file = args.config_file
    # iconv_weight = args.iconv_weight
    
    # from config
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    #if os.path.isdir(data_dir):
    #    files = load_data_path(data_dir, greed=True)
    #else:
    #    files = [data_dir]
    
    batch_size = config["training"].get('batch_size')
    SEED = config["training"].get('SEED')
    lr = float(config["training"].get('lr'))
    gamma_step = config["training"].get('gamma_step')
    gamma = config["training"].get('gamma')
    num_workers = config["training"].get('num_workers')

    epoch = config["training"].get('epoch')
    log_every = config["training"].get('log_every')

    # init
    setup_seed(SEED)
    accelerator = New_Accelerator(step_scheduler_with_optimizer = False)
    
    if os.path.isdir(data_dir):
        files = os.listdir(data_dir)
    else:
        data_dir, files = os.path.split(data_dir)
        files = [files]
    
    # parameters for dataloader construction
    dataloader_kwargs = {'batch_size': batch_size, 'shuffle': False}#Luz True}
    cuda_kwargs = {
        # 'pin_memory': True,  # 将加载的数据张量复制到 CUDA 设备的固定内存中，提高效率但会消耗更多内存
        "num_workers": num_workers,
        'prefetch_factor': 2
    }  
    dataloader_kwargs.update(cuda_kwargs)
    
    with accelerator.main_process_first():
        
        # model loading
        model = M2M_rna2atac(
            dim = config["model"].get("dim"),
            enc_num_gene_tokens = config["model"].get("total_gene") + 1,
            enc_num_value_tokens = config["model"].get('max_express') + 1, # +special tokens
            enc_depth = config["model"].get("enc_depth"),
            enc_heads = config["model"].get("enc_heads"),
            enc_ff_mult = config["model"].get("enc_ff_mult"),
            enc_dim_head = config["model"].get("enc_dim_head"),
            enc_emb_dropout = config["model"].get("enc_emb_dropout"),
            enc_ff_dropout = config["model"].get("enc_ff_dropout"),
            enc_attn_dropout = config["model"].get("enc_attn_dropout"),
            
            dec_depth = config["model"].get("dec_depth"),
            dec_heads = config["model"].get("dec_heads"),
            dec_ff_mult = config["model"].get("dec_ff_mult"),
            dec_dim_head = config["model"].get("dec_dim_head"),
            dec_emb_dropout = config["model"].get("dec_emb_dropout"),
            dec_ff_dropout = config["model"].get("dec_ff_dropout"),
            dec_attn_dropout = config["model"].get("dec_attn_dropout")
        )
        # model = model.half() #将模型参数类型改为FP16，半精度浮点数
        
        # 加载模型
        if saved_model_path:
            model.load_state_dict(torch.load(saved_model_path), strict = False)
            accelerator.print("previous model loaded!")
        else:
            accelerator.print("warning: you haven't load previous model!")
        
        # if iconv_weight:
        #     iconv_state_dict = torch.load(iconv_weight)
        #     emb_dict = {"weight": iconv_state_dict["enc.weight"]}
        #     dec_dict = {"weight": iconv_state_dict["dec.weight"], "bias": iconv_state_dict["dec.bias"]}
        #     model.enc.iConv_enc.load_state_dict(emb_dict)
        #     model.dec.iConv_dec.load_state_dict(dec_dict)
        #     accelerator.print("iConv weights loaded!")
            
        #     # iConv freeze
        #     for name, param in model.named_parameters():
        #         if "enc.iConv_enc" in name and iconv_weight:
        #             param.requires_grad = False
        #             accelerator.print(f"{name} weights freezed!")
        #         elif "dec.iConv_dec" in name and iconv_weight:
        #             param.requires_grad = False
        #             accelerator.print(f"{name} weights freezed!")
            
    #---  Prepare Optimizer ---#
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    #---  Prepare Scheduler ---#
    scheduler = StepLR(optimizer, step_size=gamma_step, gamma=gamma) 
    device = accelerator.device
    model = accelerator.prepare(model)
    #Luz  model, optimizer, train_loader, scheduler = accelerator.prepare(
    #Luz      model, optimizer, train_loader, scheduler
    #Luz  )
    
    start_date = str(datetime.datetime.now().date())
    #Luz writer = SummaryWriter(os.path.join("logs", f"{start_date}_{save_name}_logs"))
    global_iter = 0
    #Luz min_loss = 1e6
    
    model.eval()
    y_hat_out = torch.Tensor() # Luz
    with torch.no_grad():
        ########### data loading 后面还需要修改 #############
        files_iter = tqdm.tqdm(files, desc=f"Predicting", ncols=80, total=len(files)) if accelerator.is_main_process else files
        for file in files_iter:
            test_data = torch.load(os.path.join(data_dir, file))
            test_dataset = PreDataset(test_data)
            test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_kwargs)
            test_loader = accelerator.prepare(test_loader)
            data_iter = enumerate(test_loader)
            
            for i, (rna_sequence, rna_value, atac_sequence, tgt, enc_pad_mask) in data_iter:
                # rna_value: b, n
                # rna_sequence: b, n, 6
                # rna_pad_mask: b, n
                # atac_sequence: b, n, 6
                # tgt: b, n (A 01 sequence)
                    
                #y_hat_out_tmp = torch.Tensor()
                #for j in [0]:#for i in range(atac_sequence.shape[1]//10000+1):
                #    if j != atac_sequence.shape[1]//10000:
                #        atac_sequence_tmp = atac_sequence[:, (j*10000):((j+1)*10000), :]
                #    else:
                #        atac_sequence_tmp = atac_sequence[:, (j*10000):, :]
                #    start = time.time()
                   
                # run model
                logist = model(rna_sequence, atac_sequence, rna_value, enc_mask = enc_pad_mask) # b, n
                logist_hat = torch.sigmoid(logist)
                y_hat_out = torch.cat((y_hat_out, accelerator.gather_for_metrics(logist_hat).to('cpu')), dim=0)
                accelerator.wait_for_everyone()
                global_iter += 1
            accelerator.free_dataloader()
    np.save('predict.npy', y_hat_out.numpy())  #Luz
                
            #Luz if accelerator.is_main_process:    
                #Luz writer.add_scalars("Loss", {"train": epoch_loss / val_num}, e+1)
                #Luz writer.add_scalars("AUROC", {"train": epoch_auroc / val_num}, e+1)
                #Luz writer.add_scalars("AUPRC", {"train": epoch_auprc / val_num}, e+1)
                
                # # 记录梯度
                # for name, param in model.named_parameters():
                #     if param.requires_grad and 'weight' in name:
                #         writer.add_histogram(f"{name}_grad", param.grad, e+1)

            #Luz scheduler.step()
                
            #Luz accelerator.save_model(model, os.path.join(save_path, f"{start_date}_{save_name}"))
            
                     
    #Luz writer.close()
    accelerator.print(
        "#############################",
        "#### Predicting success! ####",
        "#############################",
        sep = "\n"
        )

if __name__ == "__main__":
    main()
