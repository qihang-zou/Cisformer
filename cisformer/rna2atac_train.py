import warnings
warnings.filterwarnings("ignore")
# import sys
import tqdm
import argparse
import os
import yaml

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

# from pytorchtools import EarlyStopping #Qihang

# sys.path.append("M2Mmodel")

from cisformer.M2Mmodel.M2M import M2M_rna2atac #Luz from M2M import M2M_rna2atac
import datetime
import time
# from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import binary_auprc, binary_auroc
#Luz from sklearn.metrics import precision_score, roc_auc_score
#Luz from sklearn.metrics import average_precision_score # Luz

# DDP 
# import New_Accelerate as Accelerator
from cisformer.M2Mmodel.utils import *  #Luz from utils import *

# 调试工具：梯度异常会对对应位置报错
torch.autograd.set_detect_anomaly = True
# 设置多线程文件系统
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_data_dir", required=True, help="Training data directory")
    parser.add_argument("-v", "--val_data_dir", required=True, help="Validation data directory")
    parser.add_argument("-n", "--name", required=True, help="Project name")
    parser.add_argument("-s", "--save", default="save", help="Save directory")
    parser.add_argument("-c", "--config_file", default="cisformer_config/rna2atac_config.yaml", help="Config file")
    parser.add_argument("-m", "--model_parameters", default=None, help="Load previous model")
    
    # # path
    # #Luz parser.add_argument('-d', '--data_dir', type=str, default=None, help="Preprocessed file end with '.pt'. You can specify the directory or filename of certain file")
    # #Qihang: 将训练数据集按照8:2分割成训练集和验证集，原验证集作为测试集
    # parser.add_argument('-t', '--train_data_dir', type=str, default=None, help="dir of preprocessed paired train data")
    # parser.add_argument('-v', '--val_data_dir', type=str, default=None, help="dir of preprocessed paired val data")
    # parser.add_argument('-s', '--save', type=str, default="save", help="where tht model's state_dict should be saved at")
    # parser.add_argument('-n', '--name', type=str, help="the name of this project")
    # parser.add_argument('-c', '--config_file', type=str, default="config/rna2atac_config.yaml", help="config file for model parameters")
    # parser.add_argument('-l', '--load', type=str, default=None, help="if set, load previous model, path to previous model state_dict")
    # # parser.add_argument('--iconv_weight', type=str, default=None, help='Path ot pretrained iConv weight')
    
    args = parser.parse_args()
    
    # parameters
    train_data_dir = args.train_data_dir #Luz data_dir = args.data_dir #Qihang data_dir = args.train_data_dir
    val_data_dir = args.val_data_dir # Luz
    save_path = args.save
    save_name = args.name
    saved_model_path = args.model_parameters
    config_file = args.config_file
    # iconv_weight = args.iconv_weight
    
    # from config
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    #Qihang
    #Qihang     files = load_data_path(data_dir, greed=True)
    #Qihang else:
    #Qihang     files = [data_dir]
    
    #Qihang
    #Qihang val_data_dir = args.val_data_dir #Luz
    #Qihang val_files = load_data_path(val_data_dir, greed=True) #Luz

    batch_size = config["training"].get('batch_size')
    SEED = config["training"].get('SEED')
    lr = float(config["training"].get('lr'))
    gamma_step = config["training"].get('gamma_step')
    gamma = config["training"].get('gamma')
    num_workers = config["training"].get('num_workers')

    epoch = config["training"].get('epoch')
    patience = config["training"].get('patience') #Qihang
    #Qihang log_every = config["training"].get('log_every')

    # init
    setup_seed(SEED)
    accelerator = New_Accelerator(step_scheduler_with_optimizer = False)
    stopper = Accelerate_early_stop_helper(accelerator, patience) #Qihang
    #Qihang
    #Luz if os.path.isdir(data_dir):
    #Luz     files = os.listdir(data_dir)
    #Luz else:
    #Luz     data_dir, files = os.path.split(data_dir)
    #Luz     files = [files]
    
    #Luz if os.path.isdir(train_data_dir): 
    #Luz     train_files = os.listdir(train_data_dir)
    #Luz else:
    #Luz     train_data_dir, train_files = os.path.split(train_data_dir)
    #Luz     train_files = [train_files]
    
    #Luz if os.path.isdir(val_data_dir):
    #Luz     val_files = os.listdir(val_data_dir)
    #Luz else:
    #Luz     val_data_dir, val_files = os.path.split(val_data_dir)
    #Luz     val_files = [val_files]
    
    train_files = []
    for root, dirs, files in os.walk(train_data_dir):
        for filename in files:
            if filename[-2:]=='pt':
                train_files.append(os.path.join(root, filename))

    random.seed(0)
    random.shuffle(train_files)

    val_files = []
    for root, dirs, files in os.walk(val_data_dir):
        for filename in files:
            if filename[-2:]=='pt':
                val_files.append(os.path.join(root, filename))

    #Luz train_val_dict = {}
    
    # parameters for dataloader construction
    dataloader_kwargs = {'batch_size': batch_size, 'shuffle': True} #Qihang train_kwargs = {'batch_size': batch_size, 'shuffle': True}
    val_dataloader_kwargs = {'batch_size': batch_size, 'shuffle': False} #Luz
    cuda_kwargs = {
        # 'pin_memory': True,  # 将加载的数据张量复制到 CUDA 设备的固定内存中，提高效率但会消耗更多内存
        "num_workers": num_workers,
        'prefetch_factor': 2
    }  
    dataloader_kwargs.update(cuda_kwargs) #Qihang
    
    #Qihang
    #Qihang train_kwargs.update(cuda_kwargs)
    
    #Qihang val_kwargs = {'batch_size': batch_size, 'shuffle': False} #Luz
    #Qihang val_kwargs.update(cuda_kwargs) #Luz
    
    #Qihang early_stopping = EarlyStopping(patience=5, verbose=True) #Luz

    with accelerator.main_process_first():
        
        # model loading
        model = M2M_rna2atac(
            dim = config["model"].get("dim"),
            
            enc_num_gene_tokens = config["model"].get("total_gene") + 1, # +1 for <PAD>
            enc_num_value_tokens = config["model"].get('max_express') + 1, # +1 for <PAD>
            enc_depth = 0,
            enc_heads = config["model"].get("dec_heads"),
            enc_ff_mult = config["model"].get("dec_ff_mult"),
            enc_dim_head = config["model"].get("dec_dim_head"),
            enc_emb_dropout = config["model"].get("dec_emb_dropout"),
            enc_ff_dropout = config["model"].get("dec_ff_dropout"),
            enc_attn_dropout = config["model"].get("dec_attn_dropout"),
            
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
            model.load_state_dict(torch.load(saved_model_path), strict = False)
            accelerator.print("previous model loaded!")
        
    #---  Prepare Optimizer ---#
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    #---  Prepare Scheduler ---#
    scheduler = StepLR(optimizer, step_size=gamma_step, gamma=gamma) 
    device = accelerator.device
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    
    start_date = str(datetime.datetime.now().date())
    #Luz writer = SummaryWriter(os.path.join("logs", f"{start_date}_{save_name}_logs"))
    #Qihang global_iter = 0
    #Luz min_loss = 1e6
    #Luz model.train()
    #Luz total_box = tqdm.tqdm(range(epoch)) if accelerator.is_main_process else range(epoch)
    
    max_auroc = 0 #Luz min_loss = 10000 #max_auprc = 0
    min_loss = 1e10

    for e in range(epoch): #Luz for e in total_box:
        val_num = 0
        epoch_auroc = 0
        epoch_auprc = 0
        epoch_loss = 0
        
        #Qihang 
        train_files_iter = tqdm.tqdm(train_files, desc=f"Training epoch {e+1}", ncols=80) if accelerator.is_main_process else train_files
        for file in train_files_iter: #Luz for file in files:  
            #---  Prepare Dataloader ---#
            train_data = torch.load(file) #train_data = torch.load(os.path.join(train_data_dir, file)) #Luz data = torch.load(os.path.join(data_dir, file))
            #Luz if train_val_dict.get(file):
            #Luz     train_index, _ = train_val_dict.get(file)
            #Luz else:
            #Luz     cell_number = len(data[0])
            #Luz     train_index = random.sample(list(range(cell_number)), k=round(cell_number * 0.9))
            #Luz     val_index = [each for each in range(cell_number) if each not in train_index]
            #Luz     train_val_dict[file] = [train_index, val_index]
            #Luz train_data = [each[train_index] for each in data]
            train_dataset = PreDataset(train_data)
            train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)
            train_loader = accelerator.prepare(train_loader)
            
            model.train()
            for i, (rna_sequence, rna_value, atac_sequence, tgt, enc_pad_mask) in enumerate(train_loader): #Qihang for i, (rna_value, rna_sequence, rna_pad_mask, atac_sequence, tgt) in data_iter:
                # rna_sequence: b, n
                # rna_value: b, n
                # atac_sequence: b, n, 6
                # enc_pad_mask: b, n
                # tgt: b, n (A 01 sequence)
                start = time.time()
                #accelerator.print(tgt.shape) #Luz 
                #sys.exit() #Luz
                # batch = atac_pad_mask.shape[0]
                # tgt_length = torch.sum(atac_pad_mask, dim=-1).to(device) # b
                
                # run model
                optimizer.zero_grad()
                logist = model(rna_sequence, atac_sequence, rna_value, enc_mask = enc_pad_mask) #Qihang logist = model(rna_sequence, atac_sequence, rna_value, enc_mask = rna_pad_mask) # b, n
                logist_hat = torch.sigmoid(logist)
                loss = F.binary_cross_entropy(logist_hat, tgt.float())
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                
                # accelerator.print("gpu allocate:", torch.cuda.max_memory_allocated() / 1024**3)
                #Luz accelerator.print("loss:", loss.item())
                
                # 这里只是做完整index的accuracy的计算，真正做预测需要用conditional masked language model
                #Qihang if i % log_every == 0 and accelerator.is_main_process:
                val_num += 1
                #Luz logist_hat = logist_hat.flatten().detach().cpu().numpy()
                #Luz tgt = tgt.flatten().cpu().numpy()
                #Luz auroc = roc_auc_score(tgt.astype(float), logist_hat)
                #Luz auprc= average_precision_score(tgt.astype(float), logist_hat) #Luz auprc = precision_score(tgt.astype(float), logist_hat)
                auroc = binary_auroc(logist_hat.cpu(), tgt.cpu(), num_tasks=logist_hat.shape[0]).mean().item()
                auprc = binary_auprc(logist_hat.cpu(), tgt.cpu(), num_tasks=logist_hat.shape[0]).mean().item()

                epoch_auroc += auroc
                epoch_auprc += auprc
                epoch_loss += loss.item()
                # loss变化小于阈值自动停止
                # if (min_loss - loss.item()) / loss.item() < (loss.item() / 100):
                #     accelerator.print("loss has no change, exit!")
                #     sys.exit()
                #Luz if loss.item() < min_loss:
                #Luz     min_loss = loss.item()
                
                end = time.time()
                
                #Luz accelerator.print(
                #Luz      f"epoch: {e+1}",
                #Luz      f"iter: {i+1}",  
                #Luz      f"time: {round(end -start, 2)}", 
                #Luz      f"loss: {round(loss.item(), 4)}", 
                #Luz      f"AUROC: {auroc}",
                #Luz      f"AUPRC: {auprc}",
                #Luz      sep = "\t")

            accelerator.wait_for_everyone()
            accelerator.free_dataloader()
            #Qihang global_iter += 1
            #Qihang if accelerator.is_main_process:
        accelerator.print(f"Training epoch {e+1}:", f"    loss: {round(epoch_loss/val_num, 4)}", f"    auroc: {round(epoch_auroc/val_num, 4)}", f"    auprc: {round(epoch_auprc/val_num, 4)}") #Luz
        scheduler.step()  
        
        ## validation
        #Qihang
        epoch_auroc_val = 0
        epoch_auprc_val = 0
        epoch_loss_val = 0
        val_num = 0
        val_files_iter = tqdm.tqdm(val_files, desc=f"Validating epoch {e+1}", ncols=80) if accelerator.is_main_process else val_files
        for file in val_files_iter: #Luz for file in files:    
            val_data = torch.load(file) #val_data = torch.load(os.path.join(val_data_dir, file)) #Luz data = torch.load(os.path.join(data_dir, file))
            #Luz _, val_index = train_val_dict.get(file)
            #Luz val_data = [each[val_index] for each in data]
            val_dataset = PreDataset(val_data)
            val_loader = torch.utils.data.DataLoader(val_dataset, **val_dataloader_kwargs) #Luz val_loader = torch.utils.data.DataLoader(val_dataset, **dataloader_kwargs)
            val_loader = accelerator.prepare(val_loader)
            model.eval()
            
            #Luz val_num = 0
            #Luz epoch_auroc_val = 0
            #Luz epoch_auprc_val = 0
            #Luz epoch_loss_val = 0

            #Qihang files_iter = tqdm.tqdm(val_files, desc=f"Validating epoch {e+1}", ncols=80, total = len(val_files)) if accelerator.is_main_process else val_files
            #Qihang for file in files_iter:
            #Qihang     val_data = torch.load(file)
            #Qihang     val_dataset = PreDataset(val_data)
            #Qihang     val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs)
            #Qihang     val_loader = accelerator.prepare(val_loader)
                #Luz accelerator.print("file:", file)
                
                #Luz if accelerator.is_main_process:
                #Luz     data_iter = tqdm.tqdm(enumerate(val_loader),
                #Luz                         mininterval=0.1,
                #Luz                         desc=f"Validating epoch {e+1}",
                #Luz                         total=len(val_loader),
                #Luz                         bar_format="{l_bar}{r_bar}")
                #Luz else:
                #Luz     data_iter = enumerate(val_loader)
                
                #Qihang data_iter = enumerate(val_loader)
                
            for i, (rna_sequence, rna_value, atac_sequence, tgt, enc_pad_mask) in enumerate(val_loader): #Qihang for i, (rna_value, rna_sequence, rna_pad_mask, atac_sequence, tgt) in data_iter:
                # rna_sequence: b, n
                # rna_value: b, n
                # atac_sequence: b, n, 6
                # enc_pad_mask: b, n
                # tgt: b, n (A 01 sequence)
                val_num += 1
                start = time.time()
                with torch.no_grad():
                    logist = model(rna_sequence, atac_sequence, rna_value, enc_mask = enc_pad_mask) # b, n
                    logist_hat = torch.sigmoid(logist)
                    loss = F.binary_cross_entropy(logist_hat, tgt.float())

                    #Luz logist_hat = logist_hat.flatten().detach().cpu().numpy()
                    #Luz tgt = tgt.flatten().cpu().numpy()
                    #Luz auroc = roc_auc_score(tgt.astype(float), logist_hat)
                    #Luz auprc= average_precision_score(tgt.astype(float), logist_hat) #Luz auprc = precision_score(tgt.astype(float), logist_hat)
                    auroc = binary_auroc(logist_hat.cpu(), tgt.cpu(), num_tasks=logist_hat.shape[0]).mean().item()
                    auprc = binary_auprc(logist_hat.cpu(), tgt.cpu(), num_tasks=logist_hat.shape[0]).mean().item()

                    epoch_auroc_val += auroc
                    epoch_auprc_val += auprc
                    epoch_loss_val += loss.item()
                end = time.time()
            
            accelerator.wait_for_everyone()
            accelerator.free_dataloader()
                    
        #Qihang if accelerator.is_main_process: #Luz 
        epoch_loss_val = epoch_loss_val / val_num
        accelerator.print(f"Validating epoch {e+1}:", f"    val_loss: {round(epoch_loss_val, 4)}", f"    auroc: {round(epoch_auroc_val/val_num, 4)}", f"    auprc: {round(epoch_auprc_val/val_num, 4)}") #Luz
        accelerator.print('#'*80)

        #Qihang scheduler.step()
        if accelerator.is_main_process:
            os.makedirs(os.path.join(save_path, f"{start_date}_{save_name}"), exist_ok=True)
        # accelerator.save_model(model, os.path.join(save_path, f"{start_date}_{save_name}"))
        
        #Qihang
        #Luz stop_loss = False

        # epoch_auroc_val_avg = accelerator.gather_for_metrics(torch.tensor([round(epoch_auroc_val/val_num, 4)])).mean().item()
        if epoch_loss_val>=min_loss: #Luz if (epoch_auroc_val_avg-max_auroc)<=0.0001:
            # stop_auroc = True
            accelerator.print("stop checked")
            stopper.set()
        else:
            min_loss = epoch_loss_val
            accelerator.save_model(model, os.path.join(save_path, f"{start_date}_{save_name}", f"epoch{e+1}"))
            accelerator.print("model saved!")
            # stop_auroc = False

        #Luz print(f"epoch_auroc_val_avg:{epoch_auroc_val_avg}", f"        max_auroc:{max_auroc}", f"     patience:{stopper.patience}")

        #Luz if epoch_loss_val >= min_loss:
        #Luz     stop_loss = True
        #Luz else:
        #Luz     min_loss = epoch_loss_val
        #Luz     stop_loss = False
        
        # if stop_auroc: #Luz if stop_loss: #Luz if [condition for early stop]:
        #     #accelerator.print("stop checked")
        #     #accelerator.print()
        #     stopper.set()

        if stopper.should_break():
            accelerator.print("stopped")
            break

        #Luz accelerator.print(patience); accelerator.print(stopper.patience)
        
        #Qihang if accelerator.is_main_process:
        #Qihang     early_stopping(round(epoch_loss_val/val_num, 4), model)
        #Qihang     if early_stopping.counter == 0:
        #Qihang         #accelerator.print(f"save model (epoch={e+1})")
        #Qihang         accelerator.save_model(model, os.path.join(save_path, f"{save_name}"))

        #Qihang if early_stopping.early_stop and accelerator.is_main_process:
        #Qihang     accelerator.print('Early stopping')
        #Qihang     # accelerator.save_model(model, os.path.join(save_path, f"{start_date}_{save_name}_{e+1}"))
        #Qihang     accelerator.set_trigger() #Luz break
        
        #Qihang if accelerator.check_trigger(): #Luz
        #Qihang     break #Luz
        

    #Luz writer.close()
    accelerator.print(
        "#############################",
        "##### Training success! #####",
        "#############################",
        sep = "\n"
        )

if __name__ == "__main__":
    main()
