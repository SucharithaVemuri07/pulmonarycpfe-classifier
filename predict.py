import torch
import numpy as np
from train import test_loader, val_loader, model
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score,
                             roc_curve, auc, matthews_corrcoef)
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Validation
label_names = ['NORMAL','FIBROSIS','NT','CLE','PSE']
n_classes   = len(label_names)

model.load_state_dict(torch.load("path to best model"))
model.eval()

y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for rois, full, labels in val_loader:
        logits, _ = model(rois.to(device), full.to(device))

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_prob.extend(F.softmax(logits, dim=1).cpu().numpy())   # probs

y_true  = np.array(y_true)
y_pred  = np.array(y_pred)
y_prob  = np.vstack(y_prob)          # shape [N, n_classes]

# -----------  classification report & confusion matrix  -------------
print("\nValidation classification report:")
print(classification_report(y_true, y_pred, target_names=label_names))

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['N','Fib','NT','CLE','PSE']
                      ).plot(cmap='Blues')
plt.xticks(rotation=45); plt.show()

# ---------------------  Matthews‑corr‑coef  --------------------------
mcc = matthews_corrcoef(y_true, y_pred)
print(f"Matthews correlation coefficient (MCC): {mcc:.4f}")

# ---------------------  ROC‑AUC  (OvR)  ------------------------------
y_true_1hot = np.zeros_like(y_prob)
y_true_1hot[np.arange(len(y_true)), y_true] = 1
per_class_auc = dict()
for i in range(n_classes):
    try:
        per_class_auc[label_names[i]] = roc_auc_score(
            y_true_1hot[:, i], y_prob[:, i])
    except ValueError:     
        per_class_auc[label_names[i]] = np.nan

macro_auc = np.nanmean(list(per_class_auc.values()))
print("\nPer‑class AUC:")
for k,v in per_class_auc.items():
    print(f"  {k:<8}: {v:.4f}")
print(f"Macro‑average AUC: {macro_auc:.4f}")

# ---------------------  plot macro ROC curve -------------------------
all_fpr = np.unique(
    np.concatenate([ roc_curve(y_true_1hot[:, i], y_prob[:, i])[0]
                     for i in range(n_classes) if not np.isnan(per_class_auc[label_names[i]]) ]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    if np.isnan(per_class_auc[label_names[i]]):
        continue
    fpr, tpr, _ = roc_curve(y_true_1hot[:, i], y_prob[:, i])
    mean_tpr += np.interp(all_fpr, fpr, tpr)

mean_tpr /= np.sum(~np.isnan(list(per_class_auc.values())))

plt.figure(figsize=(5,5))
plt.plot(all_fpr, mean_tpr, label=f"macro ROC (AUC={macro_auc:.3f})")
plt.plot([0,1],[0,1],'k--',alpha=.4)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Macro‑average ROC curve – validation")
plt.legend(); plt.grid(alpha=.2); plt.show()


# ---------- Testing ----------
pred_classes, pred_conf = [], []
model.eval()
with torch.no_grad():
    for rois, full_img, _ in test_loader:           # test loader has no labels, ignore the label
        logits,_ = model(rois.to(device), full_img.to(device))
        probs = torch.softmax(logits, dim=1)
        conf, cls = probs.max(1)                 # confidence & class index
        pred_classes.extend(cls.cpu().tolist())
        pred_conf.extend(conf.cpu().tolist())

n_show = 20
fig = plt.figure(figsize=(15, 8))
for i in range(n_show):
    rois, full_img = test_loader.dataset[i]      
    img_np = (full_img.permute(1,2,0).cpu().numpy()*0.5+0.5).clip(0,1)

    ax = fig.add_subplot(4, 5, i+1)
    ax.imshow(img_np)
    lbl = label_names[pred_classes[i]]
    ax.set_title(f"Pred: {lbl}\nConf: {pred_conf[i]:.2f}", fontsize=8)
    ax.axis('off')
plt.tight_layout(); plt.show()
