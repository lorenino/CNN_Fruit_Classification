"""
Test du CNN entraîné par cnn_experiment.py
Usage :
  python test_cnn_experiment.py                  # 12 images aléatoires du test set
  python test_cnn_experiment.py mon_image.jpg    # image perso
"""

import sys, os, glob, random
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, datasets

# ── doit correspondre à ce que tu avais dans cnn_experiment.py ──
TRAIN_DIR  = '/Users/julian/Documents/ESGI/DeepLearning/pres/archive/fruits-360_100x100/fruits-360/Training'
TEST_DIR   = '/Users/julian/Documents/ESGI/DeepLearning/pres/archive/fruits-360_100x100/fruits-360/Test'
MODEL_PATH = 'cnn_experiment.pth'

CLASSES      = [
    # Almonds / Nuts
    'Almonds 1', 'Chestnut 1', 'Hazelnut 1', 'Nut 1', 'Nut 2', 'Nut 3',
    'Nut 4', 'Nut 5', 'Nut Forest 1', 'Nut Pecan 1', 'Pistachio 1',
    'Peanut shell 1x 1',
    # Apples
    'Apple Braeburn 1', 'Apple Crimson Snow 1', 'Apple Golden 1',
    'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith 1',
    'Apple Pink Lady 1', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3',
    'Apple Red Delicious 1', 'Apple Red Yellow 1', 'Apple Red Yellow 2',
    # Avocado
    'Apricot 1', 'Avocado 1', 'Avocado 2', 'Avocado Black 1',
    'Avocado Black 2', 'Avocado Green 1',
    # Banana
    'Banana 1', 'Banana 3', 'Banana 4', 'Banana Lady Finger 1', 'Banana Red 1',
    # Berries
    'Blackberry 1', 'Blueberry 1', 'Gooseberry 1', 'Huckleberry 1',
    'Mulberry 1', 'Raspberry 1', 'Raspberry 2', 'Raspberry 3',
    'Raspberry 4', 'Raspberry 5', 'Raspberry 6', 'Redcurrant 1',
    'Strawberry 1', 'Strawberry 2', 'Strawberry 3', 'Strawberry Wedge 1',
    # Cactus / Tropical
    'Cactus fruit 1', 'Cactus fruit green 1', 'Cactus fruit red 1',
    'Caju seed 1', 'Granadilla 1', 'Guava 1', 'Lychee 1',
    'Maracuja 1', 'Passion Fruit 1', 'Pitahaya Red 1', 'Rambutan 1',
    'Salak 1', 'Tamarillo 1',
    # Cantaloupe / Melon
    'Cantaloupe 1', 'Cantaloupe 2', 'Cantaloupe 3',
    'Melon Piel de Sapo 1', 'Watermelon 1',
    # Citrus
    'Clementine 1', 'Grapefruit Pink 1', 'Grapefruit White 1',
    'Kumquats 1', 'Lemon 1', 'Lemon Meyer 1', 'Limes 1',
    'Mandarine 1', 'Orange 1', 'Orange 2', 'Orange 3',
    'Pomelo Sweetie 1', 'Tangelo 1',
    # Cherry
    'Cherry 1', 'Cherry 2', 'Cherry 3', 'Cherry 4', 'Cherry 5',
    'Cherry Rainier 1', 'Cherry Rainier 2', 'Cherry Rainier 3',
    'Cherry Sour 1', 'Cherry Wax 1', 'Cherry Wax 2', 'Cherry Wax Black 1',
    'Cherry Wax Red 1', 'Cherry Wax Red 2', 'Cherry Wax Red 3',
    'Cherry Wax Yellow 1',
    # Vegetables
    'Bean pod 1', 'Beetroot 1', 'Cabbage red 1', 'Cabbage white 1',
    'Carrot 1', 'Cauliflower 1', 'Cocos 1', 'Corn 1', 'Corn Husk 1',
    'Cucumber 1', 'Eggplant 1', 'Eggplant long 1', 'Ginger 2',
    'Ginger Root 1', 'Kaki 1', 'Kohlrabi 1', 'Onion 2', 'Onion Red 1',
    'Onion Red 2', 'Onion Red 3', 'Onion White 1', 'Onion White 2',
    'Pepper 1', 'Pepper 2', 'Pepper Green 1', 'Pepper Orange 1',
    'Pepper Orange 2', 'Pepper Red 1', 'Pepper Red 2', 'Pepper Red 3',
    'Pepper Red 4', 'Pepper Red 5', 'Pepper Yellow 1',
    'Potato Red 1', 'Potato Red 2', 'Potato Sweet 1', 'Potato White 1',
    'Zucchini 1', 'Zucchini Green 1', 'Zucchini dark 1',
    # Others
    'Carambula 1', 'Cherimoya 1', 'Dates 1', 'Dates 2', 'Fig 1',
    'Grape 1', 'Grape Blue 1', 'Grape Pink 1', 'Grape White 1',
    'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grape pink 2',
    'Kiwi 1', 'Mango 1', 'Mango Red 1', 'Mangostan 1',
    'Nectarine 1', 'Nectarine Flat 1', 'Nectarine Flat 2',
    'Papaya 1', 'Papaya 2', 'Peach 1',
    'Pear 1', 'Pear Abate 1', 'Pear Forelle 1', 'Pear Kaiser 1',
    'Pear Monster 1', 'Pear Red 1', 'Pear Stone 1', 'Pear Williams 1',
    'Pepino 1', 'Physalis 1', 'Physalis with Husk 1',
    'Pineapple 1', 'Pineapple Mini 1', 'Plum 1',
    'Pomegranate 1', 'Quince 1', 'Quince 2',
    'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato 5',
    'Tomato 7', 'Tomato 8', 'Tomato 9', 'Tomato 10', 'Tomato 11',
    'Tomato Cherry Maroon 1', 'Tomato Cherry Orange 1',
    'Tomato Cherry Red 1', 'Tomato Cherry Red 2', 'Tomato Cherry Yellow 1',
    'Tomato Heart 1', 'Tomato Maroon 1', 'Tomato Maroon 2', 'Tomato Yellow 1',
    'Walnut 1',
]
IMG_SIZE     = 64
CONV_LAYERS  = [(32, 3), (64, 3), (128, 3), (256, 3)]
USE_BATCHNORM = True
USE_DROPOUT   = True
FC_HIDDEN     = 1024
ACTIVATION    = 'relu'
# ────────────────────────────────────────────────────────────────

DEVICE    = 'mps' if torch.backends.mps.is_available() else 'cpu'
N_CLASSES = len(CLASSES)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_ds   = datasets.ImageFolder(TRAIN_DIR)
class2idx = full_ds.class_to_idx
keep_idxs = [class2idx[c] for c in CLASSES if c in class2idx]
label_map = {old: new for new, old in enumerate(keep_idxs)}
idx2name  = {new: c for new, c in enumerate([c for c in CLASSES if c in class2idx])}


def get_activation():
    return {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}[ACTIVATION]


class FruitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch, k in CONV_LAYERS:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=0))
            if USE_BATCHNORM:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(get_activation())
            layers.append(nn.MaxPool2d(2))
            if USE_DROPOUT:
                layers.append(nn.Dropout2d(0.25))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        flat  = self.features(dummy).view(1, -1).shape[1]
        fc = []
        if FC_HIDDEN > 0:
            fc.append(nn.Linear(flat, FC_HIDDEN))
            if USE_BATCHNORM:
                fc.append(nn.BatchNorm1d(FC_HIDDEN))
            fc.append(get_activation())
            if USE_DROPOUT:
                fc.append(nn.Dropout(0.5))
            fc.append(nn.Linear(FC_HIDDEN, N_CLASSES))
        else:
            fc.append(nn.Linear(flat, N_CLASSES))
        self.classifier = nn.Sequential(*fc)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_model():
    model = FruitCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


def predict(model, img_path):
    img    = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = model(tensor).softmax(1)[0]
        pred  = probs.argmax().item()
    return idx2name[pred], probs[pred].item(), probs.cpu().numpy()


model = load_model()
print(f'Modèle chargé : {MODEL_PATH} | Device : {DEVICE} | {len(CLASSES)} classes\n')

if len(sys.argv) > 1:
    for path in sys.argv[1:]:
        label, conf, probs = predict(model, path)
        print(f'{os.path.basename(path)} → {label} ({conf:.0%})')
        top5 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:5]
        for idx, p in top5:
            bar = '█' * int(p * 30)
            print(f'  {idx2name[idx]:30s} {bar} {p:.1%}')
        print()
else:
    img_paths = []
    for cls in CLASSES:
        cls_dir = os.path.join(TEST_DIR, cls)
        if os.path.isdir(cls_dir):
            imgs = glob.glob(os.path.join(cls_dir, '*.jpg'))
            img_paths += random.sample(imgs, min(2, len(imgs)))

    correct = total = 0
    errors  = []
    for path in img_paths:
        label, conf, _ = predict(model, path)
        folder = os.path.basename(os.path.dirname(path))
        ok = label == folder
        correct += ok
        total   += 1
        if not ok:
            errors.append((folder, label, conf))
        print(f'{"✓" if ok else "✗"} {folder:30s} → {label:30s} ({conf:.0%})')

    print(f'\n{"─"*70}')
    print(f'Accuracy : {correct}/{total} = {correct/total:.1%}')
    if errors:
        print(f'\nErreurs ({len(errors)}) :')
        for true, pred, conf in errors:
            print(f'  {true:30s} → {pred} ({conf:.0%})')
