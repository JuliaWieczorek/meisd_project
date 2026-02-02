"""Since explicit speaker identifiers are not available, we construct pseudo-user personas using
(a) linguistic style clustering and
(b) dialogue structure grouping. We further introduce a hybrid persona representation combining both signals,
enabling human-centered personalization and memory modeling."""


"""
- multitask
- soft-sharing BERT
- personalization
- user-history memory
- calibration layer
- explainable heads
- temporal validation
"""

""" bazed on  Kazienko et. al. (2023)
human-centered	        text + user modeling
personalized	        user embeddings
personal calibration	scale+bias layer
multi-task              sent + emo + int
soft-sharing            3 encodery + L2
history memory	        rolling embedding memory
reasoning-compatible	separate encoders
explainable	            token attentions
temporal validation	    past/present/future
ablation-ready	        config switches
paper metrics	        macro/micro/user
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, deque

# ======================
# CONFIG
# ======================

CONFIG = {
    "model": "bert-base-uncased",
    "max_len": 128,
    "batch": 16,
    "epochs": 5,
    "lr": 2e-5,

    "use_personalization": True,
    "use_history": True,
    "use_calibration": True,
    "use_soft_sharing": True,

    "history_size": 5
}

# ======================
# TIME SPLIT
# ======================

def time_split(df, col):
    df = df.sort_values(col)
    n = len(df)
    return df[:int(.7*n)], df[int(.7*n):int(.85*n)], df[int(.85*n):]

# ======================
# PSEUDO PERSONA BUILDER
# ======================

class PseudoUserBuilder:

    def __init__(self, mode="hybrid", n_clusters=40, text_col="Utterances"):
        self.mode = mode
        self.n_clusters = n_clusters
        self.text_col = text_col

    def style_personas(self, texts):
        enc = SentenceTransformer("all-MiniLM-L6-v2")
        X = enc.encode(texts.tolist(), normalize_embeddings=True)
        return KMeans(self.n_clusters, n_init=10, random_state=42).fit_predict(X)

    def dialog_personas(self, df):
        key = df[["TV Series","seasons","episodes","dialog_ids"]].astype(str).agg("_".join, axis=1)
        return LabelEncoder().fit_transform(key)

    def add(self, df):
        df = df.copy()

        if self.mode == "style":
            df["pseudo_user"] = self.style_personas(df[self.text_col])

        elif self.mode == "dialog":
            df["pseudo_user"] = self.dialog_personas(df)

        else:
            s = self.style_personas(df[self.text_col])
            d = self.dialog_personas(df)
            combo = pd.Series(s.astype(str)) + "_" + pd.Series(d.astype(str))
            df["pseudo_user"] = LabelEncoder().fit_transform(combo)

        print("Pseudo users:", df["pseudo_user"].nunique())
        return df

# ======================
# HISTORY MEMORY
# ======================

class UserHistoryMemory:

    def __init__(self, size):
        self.mem = defaultdict(lambda: deque(maxlen=size))

    def push(self, user, vec):
        self.mem[user].append(vec.detach().cpu())

    def get(self, user, dim, device):
        if len(self.mem[user]) == 0:
            return torch.zeros(dim, device=device)
        return torch.stack(list(self.mem[user])).mean(0).to(device)

# ======================
# DATASET
# ======================

class HCDataset(Dataset):

    def __init__(self, df, tok, max_len, emo_cols, int_cols, user_map):
        self.df=df
        self.tok=tok
        self.max_len=max_len
        self.emo_cols=emo_cols
        self.int_cols=int_cols
        self.user_map=user_map

    def __len__(self): return len(self.df)

    def __getitem__(self,i):

        r=self.df.iloc[i]

        enc=self.tok(
            r["Utterances"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "ids": enc["input_ids"].squeeze(0),
            "mask": enc["attention_mask"].squeeze(0),
            "sent": torch.tensor(r["sentiment"], dtype=torch.long),
            "emo": torch.tensor(r[self.emo_cols].values, dtype=torch.float),
            "int": torch.tensor(r[self.int_cols].values, dtype=torch.long),
            "user": torch.tensor(self.user_map[r["pseudo_user"]], dtype=torch.long)
        }

# ======================
# CALIBRATION
# ======================

class PersonalCalibration(nn.Module):

    def __init__(self,n_users,dim):
        super().__init__()
        self.scale=nn.Embedding(n_users,dim)
        self.bias=nn.Embedding(n_users,dim)
        nn.init.ones_(self.scale.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self,x,u):
        return x*self.scale(u)+self.bias(u)

# ======================
# EXPLANATION HEAD
# ======================

class ExplanationHead(nn.Module):

    def __init__(self,hid):
        super().__init__()
        self.att=nn.Linear(hid,1)

    def forward(self,states):
        w=torch.softmax(self.att(states).squeeze(-1),dim=1)
        pooled=(states*w.unsqueeze(-1)).sum(1)
        return pooled,w

# ======================
# MODEL
# ======================

class HCModel(nn.Module):

    def __init__(self,cfg,n_users,n_emo):
        super().__init__()
        self.cfg=cfg

        self.enc_s=AutoModel.from_pretrained(cfg["model"])
        self.enc_e=AutoModel.from_pretrained(cfg["model"])
        self.enc_i=AutoModel.from_pretrained(cfg["model"])

        hid=self.enc_s.config.hidden_size

        self.expl=ExplanationHead(hid)

        if cfg["use_personalization"]:
            self.user_emb=nn.Embedding(n_users,32)

        if cfg["use_calibration"]:
            self.cal=PersonalCalibration(n_users,hid)

        fusion=hid+(32 if cfg["use_personalization"] else 0)
        self.fc=nn.Linear(fusion,hid//2)

        self.h_sent=nn.Linear(hid//2,3)
        self.h_emo=nn.Linear(hid//2,n_emo)
        self.h_int=nn.Linear(hid//2,n_emo*3)

    def encode(self,enc,ids,mask,user,mem):

        out=enc(ids,mask).last_hidden_state
        pooled,att=self.expl(out)

        if self.cfg["use_history"]:
            hist = torch.stack([
                mem.get(u.item(), pooled.size(1), pooled.device)
                for u in user
            ])
            pooled = pooled + hist

        if self.cfg["use_calibration"]:
            pooled=self.cal(pooled,user)

        return pooled,att

    def fuse(self,x,user):
        if self.cfg["use_personalization"]:
            x=torch.cat([x,self.user_emb(user)],1)
        return torch.relu(self.fc(x))

    def forward(self,ids,mask,user,mem):

        s,att_s=self.encode(self.enc_s,ids,mask,user,mem)
        e,att_e=self.encode(self.enc_e,ids,mask,user,mem)
        i,att_i=self.encode(self.enc_i,ids,mask,user,mem)

        s=self.fuse(s,user)
        e=self.fuse(e,user)
        i=self.fuse(i,user)

        return {
            "sentiment": self.h_sent(s),
            "emotion": self.h_emo(e),
            "intensity": self.h_int(i),
            "att": {"sent":att_s,"emo":att_e,"int":att_i},
            "repr": s.detach()
        }

    def soft_loss(self):
        if not self.cfg["use_soft_sharing"]:
            return 0
        loss=0
        encs=[self.enc_s,self.enc_e,self.enc_i]
        for a in range(3):
            for b in range(a+1,3):
                for p1,p2 in zip(encs[a].parameters(),encs[b].parameters()):
                    if p1.shape==p2.shape:
                        loss+=(p1-p2).pow(2).sum()
        return 1e-4*loss

# ======================
# TRAIN
# ======================

def train_epoch(model,loader,opt,mem,device,n_emo):

    ce=nn.CrossEntropyLoss()
    bce=nn.BCEWithLogitsLoss()

    model.train()

    for b in loader:

        ids=b["ids"].to(device)
        mask=b["mask"].to(device)
        user=b["user"].to(device)

        out=model(ids,mask,user,mem)

        loss=ce(out["sentiment"], b["sent"].to(device))
        loss+=bce(out["emotion"], b["emo"].to(device))

        B=ids.size(0)
        il=out["intensity"].view(B,n_emo,3)

        for j in range(n_emo):
            loss+=ce(il[:,j,:], b["int"][:,j].to(device))

        loss+=model.soft_loss()

        loss.backward()
        opt.step()
        opt.zero_grad()

        for u,v in zip(user.tolist(), out["repr"]):
            mem.push(u,v)

# ======================
# RUN
# ======================

def run(csv):

    csv = "C:/Users/Julixus/DataspellProjects/meisd_project/data/MEISD_text.csv"

    df=pd.read_csv(csv)
    df["timestamp"]=pd.to_datetime(df["start_times"])

    builder=PseudoUserBuilder("hybrid",40,"Utterances")
    df=builder.add(df)

    emo_cols=[c for c in df if c.startswith("emotion")]
    int_cols=[c.replace("emotion","intensity") for c in emo_cols]

    users=df.pseudo_user.unique()
    user_map={u:i for i,u in enumerate(users)}

    past,present,future=time_split(df,"timestamp")

    tok=AutoTokenizer.from_pretrained(CONFIG["model"])

    train_ds=HCDataset(past,tok,CONFIG["max_len"],emo_cols,int_cols,user_map)
    train_loader=DataLoader(train_ds,batch_size=CONFIG["batch"],shuffle=True)

    device="cuda" if torch.cuda.is_available() else "cpu"

    model=HCModel(CONFIG,len(users),len(emo_cols)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=CONFIG["lr"])
    mem=UserHistoryMemory(CONFIG["history_size"])

    for ep in range(CONFIG["epochs"]):
        train_epoch(model,train_loader,opt,mem,device,len(emo_cols))
        print("epoch",ep,"done")
