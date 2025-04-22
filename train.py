import torch
import torch.nn as nn
from dataset import FibrosisCTDataset, EmphysemaPatchDataset, custom_collate, Unlabeled
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from models.minimal_cswin import CSWinMinimal
from models.MedViT.MedViT import MedViT_small
import types

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fib_dir   = 'path to fibrosis dataset'
pat_dir   = 'path to emphysema patches'
slices_dir= 'path to emphysema slices'
csv_path  = 'path to patch labels'
slice_csv_path = 'path to slice labels'

fib_ds = FibrosisCTDataset(fib_dir)
pat_ds = EmphysemaPatchDataset(pat_dir, csv_path)
full_ds = ConcatDataset([fib_ds, pat_ds])
n = len(full_ds)
train_n, val_n = int(0.7*n), int(0.15*n)
test_n = n - train_n - val_n
train_ds, val_ds, test_ds = random_split(full_ds, [train_n,val_n,test_n],
                                         generator=torch.Generator().manual_seed(42))

batch=4
train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, collate_fn=custom_collate)
test_loader = DataLoader(Unlabeled(test_ds), batch_size=batch, shuffle=False, collate_fn=custom_collate)

class FusionMedViT(nn.Module):
    def __init__(self, cswin, medvit, num_classes=5):
        super().__init__()
        self.cswin  = cswin          # ROI encoder
        self.medvit = medvit         # full‑image encoder

        D = cswin.stages[-1][0].attn_h.dim   # 256
        M = medvit.proj_head[0].in_features  # 1024

        # confidence head → scalar in (0,1)
        self.conf_head = nn.Sequential(
            nn.Linear(D, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        # final classifier on weighted concat
        self.classifier = nn.Linear(D+M, num_classes)

    def forward(self, rois, full_img):
        B,K,C,H,W = rois.shape
        flat = rois.view(B*K, C, H, W)

        # 1) CSWin features
        f_cs = self.cswin.forward_features(flat)        # [B*K, D]
        f_cs = f_cs.view(B, K, -1).mean(1)              # [B, D]

        # 2) MedViT features
        mv_maps = self.medvit.forward_features(full_img)
        f_mv = mv_maps[-1].mean(dim=[2,3])              # [B, M]

        # 3) confidence weight  w in (0,1)
        w = self.conf_head(f_cs).clamp(0.05, 0.95)      # [B,1]

        # 4) weighted fusion
        fused = torch.cat([w * f_cs, (1.0 - w) * f_mv], dim=1)  # [B, D+M]

        return self.classifier(fused), w.squeeze(1)     # (logits, confidence)
    

#Model Initiat
cswin  = CSWinMinimal()
medvit = MedViT_small(num_classes=5).to(device)

model  = FusionMedViT(cswin, medvit, num_classes=5).to(device)
for p in model.cswin.parameters():
    p.requires_grad = False
trainable = filter(lambda p: p.requires_grad, model.parameters())

#Training Loop
optimizer  = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
criterion  = nn.CrossEntropyLoss(reduction='none')  # keep per‑sample loss

USE_WEIGHTED_LOSS = True

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss, tot_correct, tot_samples, tot_conf = 0,0,0,0
    with torch.set_grad_enabled(train):
        for rois, full, labels in loader:
            rois, full, labels = rois.to(device), full.to(device), labels.to(device)

            logits, conf = model(rois, full)
            loss_vec = criterion(logits, labels)

            if USE_WEIGHTED_LOSS:                   # confidence re‑weighting
                loss = (conf.detach() * loss_vec).mean()
            else:
                loss = loss_vec.mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tot_loss    += loss.item()
            tot_correct += (logits.argmax(1)==labels).sum().item()
            tot_samples += labels.size(0)
            tot_conf    += conf.mean().item()*labels.size(0)

    return (tot_loss/len(loader),
            100.*tot_correct/tot_samples,
            tot_conf/tot_samples)

num_epochs, best_val = 20, 0
for ep in range(1, num_epochs+1):
    tr_l, tr_a, tr_c = run_epoch(train_loader, train=True)
    vl_l, vl_a, vl_c = run_epoch(val_loader,   train=False)
    scheduler.step()

    print(f"Ep {ep:2d} | "
          f"tr {tr_l:.3f}/{tr_a:5.1f}%  w={tr_c:.2f} | "
          f"val {vl_l:.3f}/{vl_a:5.1f}%  w={vl_c:.2f}")

    if vl_a > best_val:
        best_val = vl_a
        torch.save(model.state_dict(), "best_fusion_rew.pth")
        print("saved best")