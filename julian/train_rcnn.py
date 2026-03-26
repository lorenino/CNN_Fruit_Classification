import glob, os, time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from xml.etree import ElementTree as et

TRAIN_ROOT = '/Users/julian/Documents/ESGI/DeepLearning/pres/archive (1)/train_zip/train/'
VAL_ROOT   = '/Users/julian/Documents/ESGI/DeepLearning/pres/archive (1)/test_zip/test/'

LABELS        = ['background', 'orange', 'apple', 'banana']
LABEL2TARGET  = {l: t for t, l in enumerate(LABELS)}
NUM_CLASSES   = len(LABELS)
DEVICE        = 'cpu'
N_EPOCHS      = 5

class FruitsDataset(Dataset):
    def __init__(self, root):
        self.img_paths = sorted(glob.glob(os.path.join(root, '*.jpg')))
        self.xml_paths = sorted(glob.glob(os.path.join(root, '*.xml')))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        W_OUT = H_OUT = 224
        img = Image.open(self.img_paths[idx]).convert('RGB')
        W, H = img.size
        img = np.array(img.resize((W_OUT, H_OUT), resample=Image.Resampling.BILINEAR)) / 255.

        labels_list, boxes = [], []
        for obj in et.parse(self.xml_paths[idx]).findall('object'):
            label = obj.find('name').text
            if label not in LABEL2TARGET:
                continue
            labels_list.append(LABEL2TARGET[label])
            xmin = float(obj.find('bndbox/xmin').text)
            ymin = float(obj.find('bndbox/ymin').text)
            xmax = float(obj.find('bndbox/xmax').text)
            ymax = float(obj.find('bndbox/ymax').text)
            boxes.append([int(xmin/W*W_OUT), int(ymin/H*H_OUT),
                          int(xmax/W*W_OUT), int(ymax/H*H_OUT)])

        return (
            torch.tensor(img).permute(2, 0, 1).float(),
            {'labels': torch.tensor(labels_list, dtype=torch.long),
             'boxes':  torch.tensor(boxes, dtype=torch.float32)}
        )

    def collate_fn(self, batch):
        return tuple(zip(*batch))


def get_model():
    weights = models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model   = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


tr_ds  = FruitsDataset(TRAIN_ROOT)
val_ds = FruitsDataset(VAL_ROOT)
tr_dl  = DataLoader(tr_ds, batch_size=4, shuffle=True,  collate_fn=tr_ds.collate_fn)
val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=val_ds.collate_fn)
print(f'Train: {len(tr_ds)} imgs | Val: {len(val_ds)} imgs')

model     = get_model().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

print(f'\nEntraînement sur {DEVICE} — {N_EPOCHS} epochs\n')

for epoch in range(N_EPOCHS):
    t0 = time.time()

    model.train()
    trn_losses = []
    for i, (imgs, targets) in enumerate(tr_dl):
        imgs    = [img.to(DEVICE) for img in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        losses = model(imgs, targets)
        loss   = sum(losses.values())
        loss.backward()
        optimizer.step()
        trn_losses.append(loss.item())
        print(f'  Epoch {epoch+1} batch {i+1}/{len(tr_dl)} loss={loss.item():.4f}', end='\r')

    val_losses = []
    for imgs, targets in val_dl:
        imgs    = [img.to(DEVICE) for img in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            model.train()
            losses = model(imgs, targets)
            val_losses.append(sum(losses.values()).item())

    elapsed = time.time() - t0
    print(f'Epoch {epoch+1:02d}/{N_EPOCHS} — '
          f'train: {sum(trn_losses)/len(trn_losses):.4f} | '
          f'val: {sum(val_losses)/len(val_losses):.4f} | '
          f'{elapsed:.0f}s        ')

torch.save(model.state_dict(), 'faster_rcnn_fruits.pth')
print('\nModèle sauvegardé : faster_rcnn_fruits.pth')
