import torch
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as tf
from tqdm import tqdm
from predict import *
from HIST2ST import *
# from dataset import ViT_HER2ST, ViT_SKIN
from scipy.stats import pearsonr,spearmanr
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
# from copy import deepcopy as dcp
# from collections import defaultdict as dfd
# from sklearn.metrics import adjusted_rand_score as ari_score
# from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For debugging CUDA errors

def train(tag='5-7-2-8-4-16-32', lr = 1e-5):
    k,p,d1,d2,d3,h,c=map(lambda x:int(x),tag.split('-'))
    seed = 42
    max_epochs = 350
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    fold=5
    neighbours = 5
    genelist = "3CA"
    # data='her2st'
    data = 'hest1k'
    if data == 'her2st':
        prune = 'Grid'
        genes = 785
    elif data == 'cscc':
        prune = 'NA'
        genes = 171
    elif data == 'hest1k':
        prune = 'NA'# IDK??
        if genelist == "3CA":
            genes = 2977
        else: genes = 0
        
    trainset = pk_load('train',dataset =data, flatten=False, adj=True, ori=True, prune=prune, neighs=neighbours, genelist=genelist)
    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)

    testset = pk_load('test', dataset =data, flatten=False, adj=True, ori=True, prune=prune, neighs=neighbours, genelist=genelist)
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
    

    log_name = 'hist2st-1_hestpreprocessed'
    logger = TensorBoardLogger(
        'logs', 
        name=log_name
    )

    model=Hist2ST(
        depth1=d1, depth2=d2,depth3=d3,n_genes=genes,
        learning_rate=lr, kernel_size=k, patch_size=p,
        heads=h, channel=c, dropout=0.2,
        zinb=0.25, nb=False,
        bake=5, lamb=0.5
    )
    
    from datetime import date
    today = date.today()
    print(f"Today's date: {today}")
    logger=None
    trainer = pl.Trainer( accelerator = 'gpu', 
                        max_epochs=max_epochs,
                        logger=logger,
                        check_val_every_n_epoch=2,
    )
    trainer.fit(model, train_loader, test_loader)

    import os
    from datetime import date
    if not os.path.isdir("./model/"):
        os.mkdir("./model/")

    torch.save(model.state_dict(),f"./model/Hist2ST_{today}.ckpt")
        
if __name__ == "__main__":
    train()