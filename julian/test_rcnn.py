import sys, glob, os, random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

LABELS       = ['background', 'orange', 'apple', 'banana']
LABEL2TARGET = {l: t for t, l in enumerate(LABELS)}
TARGET2LABEL = {t: l for l, t in LABEL2TARGET.items()}
NUM_CLASSES  = len(LABELS)
COLORS       = {'apple': 'red', 'banana': 'gold', 'orange': 'darkorange'}
DEVICE       = 'cpu'
MODEL_PATH   = 'faster_rcnn_fruits.pth'
VAL_ROOT     = '/Users/julian/Documents/ESGI/DeepLearning/pres/archive (1)/test_zip/test/'

def load_model():
    weights = models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model   = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def predict(model, img_path, conf=0.4):
    img = Image.open(img_path).convert('RGB')
    W, H = img.size
    img_resized = np.array(img.resize((224, 224), resample=Image.Resampling.BILINEAR)) / 255.
    tensor = torch.tensor(img_resized).permute(2, 0, 1).float().unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)[0]

    boxes  = output['boxes']
    scores = output['scores']
    labels = output['labels']

    keep = scores >= conf
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if len(boxes) == 0:
        return [], [], []

    idxs   = nms(boxes, scores, iou_threshold=0.3)
    boxes  = boxes[idxs].numpy().astype(int).tolist()
    scores = scores[idxs].numpy().tolist()
    labels = [TARGET2LABEL[l.item()] for l in labels[idxs]]
    return boxes, scores, labels

def show(img_path, boxes, scores, labels):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    if not boxes:
        ax.set_title('Aucune détection')
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        color = COLORS.get(label, 'white')
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor='none'
        ))
        ax.text(x1, max(y1-5, 0), f'{label} {score:.0%}',
                color='white', fontsize=11, fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.8, pad=2))
    ax.axis('off')
    ax.set_title(os.path.basename(img_path))
    plt.tight_layout()
    plt.show()

model = load_model()
print(f'Modèle chargé depuis {MODEL_PATH}\n')

# Si une image est passée en argument : python test_rcnn.py mon_image.jpg
if len(sys.argv) > 1:
    img_paths = sys.argv[1:]
else:
    # Sinon on prend 6 images aléatoires du set de test
    all_imgs  = glob.glob(os.path.join(VAL_ROOT, '*.jpg'))
    img_paths = random.sample(all_imgs, min(20, len(all_imgs)))

for path in img_paths:
    boxes, scores, labels = predict(model, path)
    if labels:
        print(f'{os.path.basename(path)} → {list(zip(labels, [f"{s:.0%}" for s in scores]))}')
    else:
        print(f'{os.path.basename(path)} → aucune détection')
    show(path, boxes, scores, labels)
