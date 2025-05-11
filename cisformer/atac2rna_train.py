import warnings
warnings.filterwarnings("ignore")
# import sys
import tqdm
# import argparse
import os
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import numpy as np
import argparse
# sys.path.append("M2Mmodel")

# from sklearn.metrics import accuracy_score, precision_score, recall_score
import datetime
import time
from torch.utils.tensorboard import SummaryWriter

from cisformer.M2Mmodel.M2M import M2M_atac2rna
from cisformer.M2Mmodel.utils import *
# import accelerate.utils as au

# 调试工具：梯度异常会对对应位置报错
torch.autograd.set_detect_anomaly = True
# 设置多线程文件系统
torch.multiprocessing.set_sharing_strategy('file_system')

def CrossEntropy_weight_computor(data, num_classes, device):
    value, counts = np.unique(data[1].numpy(), return_counts=True)
    total = counts.sum()
    frequencies = counts / total
    weights = 1 / frequencies
    normalized_weights = weights / np.sum(weights)
    new_normalized_weights = [0]*int(num_classes)
    for each in range(len(value)):
        new_normalized_weights[value[each]] = normalized_weights[each]
    weights = torch.tensor(new_normalized_weights, dtype=torch.float32, device=device)
    weights[0] *= 2
    # weights[-1] *= 10
    # weights[-2] *= 10
    # weights[1] *= 2
    # weights = torch.ones_like(weights)
    return weights

def main():
    
    parser = argparse.ArgumentParser(description='pretrain script of M2M')
    
    parser.add_argument("-d", "--data_dir", required=True, help="Data directory")
    parser.add_argument("-n", "--name", required=True, help="Project name")
    parser.add_argument("-s", "--save", default="save", help="Save directory")
    parser.add_argument("-c", "--config_file", default="cisformer_config/atac2rna_config.yaml", help="Config file")
    parser.add_argument("-m", "--model_parameters", default=None, help="Load previous model")

    # # path
    # parser.add_argument('-d', '--data_dir', type=str, help="Preprocessed file end with '.pt'. You can specify the directory or filename of certain file")
    # parser.add_argument('-n', '--name', type=str, help="the name of this project")
    # parser.add_argument('-s', '--save', type=str, default="save", help="where tht model's state_dict should be saved at")
    # parser.add_argument('-c', '--config_file', type=str, default="config/atac2rna_config.yaml", help="config file for model parameters")
    # parser.add_argument('-l', '--load', type=str, default=None, help="if set, load previous model, path to previous model state_dict")
    # # parser.add_argument('--iconv_weight', type=str, default=None, help='Path ot pretrained iConv weight')
    
    args = parser.parse_args()
    
    # parameters
    data_dir = args.data_dir
    save_path = args.save
    save_name = args.name
    saved_model_path = args.model_parameters
    config_file = args.config_file
    # patience_counter = 0
    # iconv_weight = args.iconv_weight
    
    # from config
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    batch_size = config["training"].get('batch_size')
    SEED = config["training"].get('SEED')
    lr = float(config["training"].get('lr'))
    gamma_step = config["training"].get('gamma_step')
    gamma = config["training"].get('gamma')
    num_workers = config["training"].get('num_workers')

    epoch = config["training"].get('epoch')
    patience = config["training"].get('patience')
    # log_every = config["training"].get('log_every')
    
    total_gene = config["model"].get("total_gene") + 1 # +1 for <PAD>
    total_value = config["model"].get('max_express') + 1 # +1 for 0

    # init
    setup_seed(SEED)
    accelerator = New_Accelerator(step_scheduler_with_optimizer = False)
    stopper = Accelerate_early_stop_helper(accelerator, patience)
    
    if os.path.isdir(data_dir):
        files = os.listdir(data_dir)
    else:
        data_dir, files = os.path.split(data_dir)
        files = [files]
    files = [f for f in files if f.endswith(".pt")]
    assert len(files) > 0, "No file with format '.pt'"
    train_val_dict = {}
    
    # parameters for dataloader construction
    dataloader_kwargs = {'batch_size': batch_size, 'shuffle': True}
    cuda_kwargs = {
        # 'pin_memory': True,  # 将加载的数据张量复制到 CUDA 设备的固定内存中，提高效率但会消耗更多内存
        "num_workers": num_workers,
        'prefetch_factor': 2
    }  
    dataloader_kwargs.update(cuda_kwargs)
    
    with accelerator.main_process_first():    
        # model loading
        model = M2M_atac2rna(
            dim = config["model"].get("dim"),
            
            enc_depth = 0, # No encoder
            enc_heads = config["model"].get("dec_heads"),
            enc_ff_mult = config["model"].get("dec_ff_mult"),
            enc_dim_head = config["model"].get("dec_dim_head"),
            enc_emb_dropout = config["model"].get("dec_emb_dropout"),
            enc_ff_dropout = config["model"].get("dec_ff_dropout"),
            enc_attn_dropout = config["model"].get("dec_attn_dropout"),
            
            dec_num_gene_tokens = total_gene, # +1 for <PAD>
            dec_num_value_tokens = total_value, # +1 for 0
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
        if saved_model_path and saved_model_path != "None":
            model.load_state_dict(torch.load(saved_model_path), strict = True)
            accelerator.print("previous model loaded!")
                
    #---  Prepare Optimizer ---#
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    #---  Prepare Scheduler ---#
    scheduler = StepLR(optimizer, step_size=gamma_step, gamma=gamma) 
    device = accelerator.device
    
    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )
    # loss_fn = nn.CrossEntropyLoss()
    # weight = torch.ones(max_express + 3).to(device)
    # weight[3] = 0.1  # 下调值为1的权重
    
    start_date = str(datetime.datetime.now().date())
    writer = SummaryWriter(os.path.join("logs", f"{start_date}_{save_name}_logs"))
    # min_loss = 1e6
    
    max_val_epoch_accuracy = 0
    max_val_epoch_peason_corr = 0
    min_val_epoch_loss = 1e10
    total_box = tqdm.tqdm(range(epoch), ncols=80) if accelerator.is_main_process else range(epoch)
    for e in total_box:
        # accelerator.print("here", e)
        train_num = 1e-6
        val_num = 1e-6
        train_epoch_accuracy = 0
        train_epoch_peason_corr = 0
        train_epoch_loss = 0
        val_epoch_accuracy = 0
        val_epoch_peason_corr = 0
        val_epoch_loss = 0
          
        for file in files:  
            #---  Prepare Dataloader ---#
            accelerator.print(file)
            data = torch.load(os.path.join(data_dir, file))
            # dynamic criterion
            weights = CrossEntropy_weight_computor(data, total_value, device)
            # loss_fn1 = nn.CrossEntropyLoss(weight=weights)
            # loss_fn2 = nn.CrossEntropyLoss(ignore_index=0)
            loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=0)
            # loss_fn = nn.CrossEntropyLoss()
            # divide dataset
            if train_val_dict.get(file):
                train_index, _ = train_val_dict.get(file)
            else:
                cell_number = len(data[0])
                train_index = random.sample(list(range(cell_number)), k=round(cell_number * 0.9))
                val_index = [each for each in range(cell_number) if each not in train_index]
                train_val_dict[file] = [train_index, val_index]
            train_data = [each[train_index] for each in data]
            train_dataset = PreDataset(train_data)
            train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)
            train_loader = accelerator.prepare(train_loader)
            model.train()
            for i, (rna_sequence, rna_value, atac_sequence, atac_value, enc_pad_mask) in enumerate(train_loader):
                # rna_sequence: b, n
                # rna_value: b, n
                # atac_sequence: b, n, 6
                # enc_pad_mask: b, n
                
                start = time.time()
                # batch = enc_pad_mask.shape[0]
                # tgt_length = torch.sum(enc_pad_mask, dim=-1).to(device) # b
                
                # run model
                optimizer.zero_grad()
                logist = model(atac_sequence, rna_sequence, enc_mask = enc_pad_mask, dec_mask = (rna_value != 0)) # b, n, num_value_tokens
                loss = loss_fn(logist.permute(0, 2, 1), rna_value)
                # loss = loss_fn1(logist.permute(0, 2, 1), rna_value) + weights.mean().cpu().item() * loss_fn2(logist.permute(0, 2, 1), rna_value)
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                
                # accelerator.print("gpu allocate:", torch.cuda.max_memory_allocated() / 1024**3)
                # accelerator.print("loss:", loss.item())
                
                # log
                # if i % log_every == 0:
                train_num += 1
                
                # # accuracy:
                # logist_hat = logist.argmax(dim = -1)
                # accuracy = round(torch.eq(logist_hat, rna_value).sum().item() / rna_value.numel(), 4)
                
                # # peason correlation:
                # logist_hat = logist_hat.cpu().flatten().numpy()
                # rna_value = rna_value.cpu().flatten().numpy()
                # accelerator.print(np.isnan(logist_hat).any(), np.isnan(rna_value).any())
                # peason_corr = np.corrcoef(logist_hat, rna_value)[0,1]
                
                # train_epoch_accuracy += accuracy
                # train_epoch_peason_corr += peason_corr
                train_epoch_loss += loss.item()
                # loss变化小于阈值自动停止
                # if (min_loss - loss.item()) / loss.item() < (loss.item() / 100):
                #     accelerator.print("loss has no change, exit!")
                #     sys.exit()
                # if loss.item() < min_loss:
                #     min_loss = loss.item()
                
                end = time.time()
                accelerator.print(
                    f"Train",
                    f"epoch: {e+1}",
                    f"iter: {i+1}", 
                    f"time: {round(end -start, 2)}", 
                    f"loss: {round(loss.item(), 4)}", 
                    # f"Accuracy: {accuracy}",
                    # f"Peason correlation coefficient: {peason_corr}",
                    sep = "\t")

            accelerator.wait_for_everyone()
            accelerator.free_dataloader()
        scheduler.step()    
             
        for file in files:    
            accelerator.print(file)
            data = torch.load(os.path.join(data_dir, file))
            # dynamic criterion
            weights = CrossEntropy_weight_computor(data, total_value, device)
            # loss_fn1 = nn.CrossEntropyLoss(weight=weights)
            # loss_fn2 = nn.CrossEntropyLoss(ignore_index=0)
            loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=0)
            # loss_fn = nn.CrossEntropyLoss()
            # divide dataset
            _, val_index = train_val_dict.get(file)
            val_data = [each[val_index] for each in data]
            val_dataset = PreDataset(val_data)
            val_loader = torch.utils.data.DataLoader(val_dataset, **dataloader_kwargs)
            val_loader = accelerator.prepare(val_loader)
            model.eval()
            with torch.no_grad():
                for i, (rna_sequence, rna_value, atac_sequence, atac_value, enc_pad_mask) in enumerate(val_loader):
                    # rna_sequence: b, n
                    # rna_value: b, n
                    # atac_sequence: b, n, 6
                    # enc_pad_mask: b, n
                    
                    start = time.time()
                    # batch = enc_pad_mask.shape[0]
                    # tgt_length = torch.sum(enc_pad_mask, dim=-1).to(device) # b
                    # zero_tokens = rna_value == 0
                    # nonzero_tokens = rna_value != 0
                    
                    # run model
                    logist = model(atac_sequence, rna_sequence, enc_mask = enc_pad_mask, dec_mask = (rna_value != 0)) # b, n, num_value_tokens
                    loss = loss_fn(logist.permute(0, 2, 1), rna_value)
                    # loss = loss_fn1(logist.permute(0, 2, 1), rna_value) + weights.mean().cpu().item() * loss_fn2(logist.permute(0, 2, 1), rna_value)
                    
                    # log
                    # if i % log_every == 0:
                    val_num += 1
                    
                    # # accuracy:
                    # logist_hat = logist.argmax(dim = -1)
                    # accuracy = round(torch.eq(logist_hat, rna_value).sum().item() / rna_value.numel(), 4)
                    
                    # # peason correlation:
                    # logist_hat = logist_hat.cpu().flatten().numpy()
                    # rna_value = rna_value.cpu().flatten().numpy()
                    # accelerator.print(np.isnan(logist_hat).any(), np.isnan(rna_value).any())
                    # peason_corr = np.corrcoef(logist_hat, rna_value)[0,1]
                    
                    # val_epoch_accuracy += accuracy
                    # val_epoch_peason_corr += peason_corr
                    val_epoch_loss += loss.item()
                    # loss变化小于阈值自动停止
                    # if (min_loss - loss.item()) / loss.item() < (loss.item() / 100):
                    #     accelerator.print("loss has no change, exit!")
                    #     sys.exit()
                    # if loss.item() < min_loss:
                    #     min_loss = loss.item()
                    
                    end = time.time()
                    accelerator.print(
                        f"Val",
                        f"epoch: {e+1}",
                        f"iter: {i+1}", 
                        f"time: {round(end -start, 2)}", 
                        f"loss: {round(loss.item(), 4)}", 
                        # f"Accuracy: {accuracy}",
                        # f"Peason correlation coefficient: {peason_corr}",
                        sep = "\t")
        
            accelerator.wait_for_everyone()  
            accelerator.free_dataloader()
                
        if accelerator.is_main_process:    
            writer.add_scalars("Loss", {"train": train_epoch_loss / train_num}, e+1)
            # writer.add_scalars("Accuracy", {"train": train_epoch_accuracy / train_num}, e+1)
            # writer.add_scalars("Peason Correlation Coefficient", {"train": train_epoch_peason_corr / train_num}, e+1)
            writer.add_scalars("Loss", {"val": val_epoch_loss / val_num}, e+1)
            # writer.add_scalars("Accuracy", {"val": val_epoch_accuracy / val_num}, e+1)
            # writer.add_scalars("Peason Correlation Coefficient", {"val": val_epoch_peason_corr / val_num}, e+1)
            
            # # 记录梯度
            # for name, param in model.named_parameters():
            #     if param.requires_grad and 'weight' in name:
            #         writer.add_histogram(f"{name}_grad", param.grad, e+1)
            
        # accelerator.print("model saving start")
        if accelerator.is_main_process:
            os.makedirs(os.path.join(save_path, f"{start_date}_{save_name}"), exist_ok=True)
        
        # early stop:
        all_val_epoch_losses = accelerator.gather(torch.tensor(val_epoch_loss, dtype=float, device=accelerator.device)) # Only nested list/tuple/dicts of objects that are valid for `is_torch_tensor` should be passed.
        avg_val_epoch_loss = all_val_epoch_losses.mean()
        # print(avg_val_epoch_loss, accelerator.process_index)
        
        if e > 2 and not np.isnan(val_epoch_peason_corr): # Warm up for first three epoch
            if avg_val_epoch_loss < min_val_epoch_loss:
                min_val_epoch_loss = avg_val_epoch_loss
                accelerator.save_model(model, os.path.join(save_path, f"{start_date}_{save_name}", f"epoch{e+1}"))
                accelerator.print("model saved!")
            else:   
                accelerator.print("stop checked")
                stopper.set()
            if stopper.should_break():
                accelerator.print("stopped")
                break
                         
    writer.close()
    accelerator.print(
        "#############################",
        "##### Training success! #####",
        "#############################",
        sep = "\n"
        )

if __name__ == "__main__":
    main()