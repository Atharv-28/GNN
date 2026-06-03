"""HGCM-Net Training — RTX 3050 4GB VRAM
Full-batch GNN forward pass on GPU.
Contrastive loss uses 512-sample mini-batch to avoid OOM.
"""
import os, time, logging
from pathlib import Path
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from config import HGCMConfig
from model import HGCMNet
from losses import HGCMCombinedLoss

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log", encoding="utf-8")]
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_metrics(logits, labels):
    preds = (logits.sigmoid() > 0.5).long()
    tp = ((preds==1)&(labels==1)).sum().float()
    fp = ((preds==1)&(labels==0)).sum().float()
    fn = ((preds==0)&(labels==1)).sum().float()
    tn = ((preds==0)&(labels==0)).sum().float()
    acc  = (preds==labels).float().mean().item()
    prec = (tp/(tp+fp+1e-8)).item()
    rec  = (tp/(tp+fn+1e-8)).item()
    f1   = 2*prec*rec/(prec+rec+1e-8)
    return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,
            "tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn)}


def train_epoch(model, x_dict, edge_dict, labels, train_mask,
                criterion, optimizer, device, contrastive_bs=512):
    model.train()
    optimizer.zero_grad()

    logits, user_emb, content_emb = model(x_dict, edge_dict)

    # BCE on all training nodes
    logits_m = logits[train_mask]
    labels_m = labels[train_mask]
    loss_bce = criterion.bce(logits_m, labels_m.float())

    # Contrastive on 512-sample mini-batch to avoid OOM
    train_idx = train_mask.nonzero(as_tuple=False).squeeze(1)
    perm = torch.randperm(train_idx.size(0), device=device)[:contrastive_bs]
    c_batch = content_emb[train_idx[perm]]
    u_batch = user_emb.mean(0, keepdim=True).expand(c_batch.size(0), -1)
    loss_con = criterion.contrastive(u_batch, c_batch)

    loss = loss_bce + criterion.lam * loss_con
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    m = compute_metrics(logits_m.detach(), labels_m.detach())
    m.update({"loss": loss.item(), "loss_bce": loss_bce.item(), "loss_con": loss_con.item()})
    return m


@torch.no_grad()
def eval_split(model, x_dict, edge_dict, labels, mask, criterion):
    model.eval()
    logits, _, _ = model(x_dict, edge_dict)
    logits_m = logits[mask]; labels_m = labels[mask]
    loss = criterion.bce(logits_m, labels_m.float())
    m = compute_metrics(logits_m, labels_m)
    m["loss"] = loss.item()
    return m


def create_splits(n, val=0.15, test=0.15, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    nt, nv = int(n*test), int(n*val)
    tm = torch.zeros(n,dtype=torch.bool); tm[perm[nt+nv:]]=True
    vm = torch.zeros(n,dtype=torch.bool); vm[perm[nt:nt+nv]]=True
    sm = torch.zeros(n,dtype=torch.bool); sm[perm[:nt]]=True
    return tm, vm, sm


def main():
    config = HGCMConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s | VRAM: %.1f GB", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory/1e9)

    from data_pipeline import run_pipeline
    data = run_pipeline(config=config, n_users=config.n_users)
    n = data["content"].num_nodes
    logger.info("Content: %d | Users: %d", n, data["user"].num_nodes)
    logger.info("Labels: %s", dict(zip(
        *[v.tolist() for v in data["content"].y.unique(return_counts=True)])))

    # Move entire graph to GPU once
    x_dict    = {k: data[k].x.to(device)          for k in data.node_types}
    edge_dict = {k: data[k].edge_index.to(device)  for k in data.edge_types}
    labels    = data["content"].y.to(device)

    train_mask, val_mask, test_mask = create_splits(n, config.val_ratio, config.test_ratio)
    train_mask, val_mask, test_mask = [m.to(device) for m in (train_mask, val_mask, test_mask)]
    logger.info("Train: %d | Val: %d | Test: %d",
                train_mask.sum(), val_mask.sum(), test_mask.sum())

    model = HGCMNet(config).to(device)
    with torch.no_grad(): model(x_dict, edge_dict)  # lazy init
    logger.info("Params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler_warmup, T_mult=2)
    criterion = HGCMCombinedLoss(config.contrastive_lambda, config.temperature).to(device)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_f1, patience_ctr = 0.0, 0
    logger.info("="*65)
    logger.info("Training started | epochs=%d lr=%.1e patience=%d",
                config.epochs, config.lr, config.patience)
    logger.info("="*65)

    for epoch in range(1, config.epochs+1):
        t0 = time.time()
        tm = train_epoch(model, x_dict, edge_dict, labels, train_mask,
                         criterion, optimizer, device)
        vm = eval_split(model, x_dict, edge_dict, labels, val_mask, criterion)
        scheduler.step()

        logger.info("Ep %3d | loss=%.4f bce=%.4f con=%.4f | "
                    "tr acc=%.3f f1=%.3f | val acc=%.3f f1=%.3f | %.1fs",
                    epoch, tm["loss"], tm["loss_bce"], tm["loss_con"],
                    tm["accuracy"], tm["f1"], vm["accuracy"], vm["f1"],
                    time.time()-t0)

        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]; patience_ctr = 0
            torch.save({"epoch":epoch,"model_state":model.state_dict(),
                        "val_f1":best_f1,"val_acc":vm["accuracy"]},
                       Path(config.checkpoint_dir)/"best_model.pt")
            logger.info("  >> Saved (val_f1=%.4f acc=%.4f)", best_f1, vm["accuracy"])
        else:
            patience_ctr += 1
            if patience_ctr >= config.patience:
                logger.info("Early stop at epoch %d", epoch); break

    ckpt = torch.load(Path(config.checkpoint_dir)/"best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    sm = eval_split(model, x_dict, edge_dict, labels, test_mask, criterion)
    logger.info("="*65)
    logger.info("TEST | Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f",
                sm["accuracy"], sm["precision"], sm["recall"], sm["f1"])
    logger.info("="*65)

if __name__ == "__main__":
    main()
