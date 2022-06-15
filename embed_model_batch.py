import argparse
import json
import logging
import os
import sys
import pandas as pd
import numpy as np

#import sagemaker_containers
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
import time

from s3fs.core import S3FileSystem
import io
s3 = S3FileSystem()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(skipgram, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)   
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True) 
        self.embedding_dim = embedding_dim
        self.init_emb()
    
    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)
    
    def forward(self, u_pos, v_pos, v_neg_city, v_neg_country):

        embed_u = self.u_embeddings(u_pos)
        embed_v = self.v_embeddings(v_pos)
        
        embed_u = embed_u.unsqueeze(0)

        score  = torch.mul(embed_u, embed_v)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score).squeeze()

       
        neg_embed_v_city = self.v_embeddings(v_neg_city)
        neg_embed_v_country = self.v_embeddings(v_neg_country)

        neg_score_city = torch.mul(neg_embed_v_city, embed_u)
        neg_score_city = torch.sum(neg_score_city, dim=1)
        sum_log_neg_score_city = F.logsigmoid(-1*neg_score_city).squeeze()
        
        
        neg_score_country = torch.mul(neg_embed_v_country, embed_u)
        neg_score_country = torch.sum(neg_score_country, dim=1)
        sum_log_neg_score_country = F.logsigmoid(-1*neg_score_country).squeeze()
        

        loss = log_target.sum()+ sum_log_neg_score_city.sum() + sum_log_neg_score_country.sum()

        return -1*loss
    
    def input_embeddings(self):
        return self.u_embeddings.weight.data.cpu().numpy()

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "embedding_v1.pth")
    torch.save(model.cpu().state_dict(), path)
    
def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size
def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        with torch.no_grad():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= size
def train(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")


    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.backend, dist.get_world_size()
            )
            + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(), args.num_gpus)
        )

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    pos_sample = np.load(s3.open('s3://prod-search-ranking-ml/data/dnw/Embedding_data_preprocessed/aug/pos_sample_aug.npy'),allow_pickle=True)
    neg_sample_city = np.load(s3.open('s3://prod-search-ranking-ml/data/dnw/Embedding_data_preprocessed/aug/neg_sample_city_aug.npy'),allow_pickle=True)
    neg_sample_country = np.load(s3.open('s3://prod-search-ranking-ml/data/dnw/Embedding_data_preprocessed/aug/neg_sample_country_aug.npy'),allow_pickle=True)
    idx2hotel=np.load(s3.open('s3://prod-search-ranking-ml/data/dnw/Embedding_data_preprocessed/aug/idx2word_aug.npy'),allow_pickle=True)

    neg_sample_city=neg_sample_city.tolist()
    neg_sample_country=neg_sample_country.tolist()
    vocab_size=len(idx2hotel)
    model = skipgram(vocab_size, 32).to(device)
   
    batch_size=args.batch_size
    
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    print("TRAINING STARTED.....")
    for epoch in range(1, args.epoch + 1):
        model.train()
        epoch_loss = 0
        tic=time.time()
        for i in range(len(pos_sample)//batch_size):
            loss=0
            for j in range (batch_size):
                k=i*batch_size+j
                if (k>=len(pos_sample)):
                    break
                center_id = torch.tensor(pos_sample[k][0]).to(device)
                positive = torch.tensor(pos_sample[k][1]).to(device)
                neg_within_city = torch.tensor(neg_sample_city[k]).to(device)
                neg_within_country = torch.tensor(neg_sample_country[k]).to(device)
                optimizer.zero_grad()
                loss = loss + model(center_id, positive, neg_within_city, neg_within_country)
            loss=loss/batch_size
            loss.backward()
            epoch_loss += loss.item()
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()
            toc=time.time()
        print("Epoch: {}  Loss:{} Time {}" .format(epoch,epoch_loss/(len(pos_sample)//batch_size),str((toc-tic)/60)))
    save_model(model, args.model_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logger.info("started")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="number of epoch"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="batch size"
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    #parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())
