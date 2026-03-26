"""
CNN EXPÉRIMENTAL — Fruits-360
================================
Inspiré des cours Bonneton (CNN-MNIST + CNN-CAT-DOG) en PyTorch.
Lance-le : python cnn_experiment.py
Modifie les zones marquées  ▶ TWEAK  ◀  et relance pour voir l'effet.
"""

import os, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
#  ▶ TWEAK 1 — Données
# ─────────────────────────────────────────────────────────────
TRAIN_DIR = '/Users/julian/Documents/ESGI/DeepLearning/pres/archive/fruits-360_100x100/fruits-360/Training'
TEST_DIR  = '/Users/julian/Documents/ESGI/DeepLearning/pres/archive/fruits-360_100x100/fruits-360/Test'

CLASSES = [
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

IMG_SIZE = 64   # 64 = rapide, 100 = fidèle aux images d'origine

# ─────────────────────────────────────────────────────────────
#  ▶ TWEAK 2 — Entraînement
# ─────────────────────────────────────────────────────────────
N_EPOCHS   = 30
BATCH_SIZE = 64
LR         = 1e-3       # essaie 1e-2, 5e-4, 1e-4

OPTIMIZER  = 'adam'     # 'adam' | 'sgd' | 'rmsprop'
# SGD classique comme dans le cours :
# OPTIMIZER = 'sgd'

# ─────────────────────────────────────────────────────────────
#  ▶ TWEAK 3 — Architecture CNN
# ─────────────────────────────────────────────────────────────
ACTIVATION   = 'relu'   # 'relu' | 'sigmoid' | 'tanh'
USE_BATCHNORM = True    # True = plus stable (comme CNN-CAT-DOG)
USE_DROPOUT   = True    # True = moins d'overfitting
DROPOUT_RATE  = 0.25    # entre 0.1 et 0.5

# Couches convolutives : liste de (nb_filtres, taille_noyau)
# Modèle minimal du cours MNIST :
# CONV_LAYERS = [(8, 3), (16, 3)]
# Modèle Cat/Dog du cours :
CONV_LAYERS = [(32, 3), (64, 3), (128, 3), (256, 3)]
# Essaie d'enlever une couche ou de changer les filtres

FC_HIDDEN = 1024  # neurones dans la couche Dense cachée (0 = pas de couche cachée)

# ─────────────────────────────────────────────────────────────
#  Fin des tweaks — code automatique en dessous
# ─────────────────────────────────────────────────────────────

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
N_CLASSES = len(CLASSES)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

full_train = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
full_test  = datasets.ImageFolder(TEST_DIR,  transform=test_tf)
class2idx  = full_train.class_to_idx
keep_idxs  = [class2idx[c] for c in CLASSES if c in class2idx]
label_map  = {old: new for new, old in enumerate(keep_idxs)}

train_ds = Subset(full_train, [i for i, (_, l) in enumerate(full_train.samples) if l in keep_idxs])
test_ds  = Subset(full_test,  [i for i, (_, l) in enumerate(full_test.samples)  if l in keep_idxs])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f'Device : {DEVICE}')
print(f'Classes : {CLASSES}')
print(f'Train : {len(train_ds)} imgs | Test : {len(test_ds)} imgs\n')


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
                layers.append(nn.Dropout2d(DROPOUT_RATE))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)

        # calcule la taille de sortie automatiquement
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        flat  = self.features(dummy).view(1, -1).shape[1]

        fc_layers = []
        if FC_HIDDEN > 0:
            fc_layers.append(nn.Linear(flat, FC_HIDDEN))
            if USE_BATCHNORM:
                fc_layers.append(nn.BatchNorm1d(FC_HIDDEN))
            fc_layers.append(get_activation())
            if USE_DROPOUT:
                fc_layers.append(nn.Dropout(0.5))
            fc_layers.append(nn.Linear(FC_HIDDEN, N_CLASSES))
        else:
            fc_layers.append(nn.Linear(flat, N_CLASSES))

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


model = FruitCNN().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Paramètres entraînables : {total_params:,}\n')
print(model)
print()

criterion = nn.CrossEntropyLoss()

if OPTIMIZER == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
elif OPTIMIZER == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
else:
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

for epoch in range(N_EPOCHS):
    t0 = time.time()
    model.train()
    trn_loss = trn_correct = trn_total = 0

    for imgs, labels in train_dl:
        labels = torch.tensor([label_map[l.item()] for l in labels])
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        trn_loss    += loss.item() * imgs.size(0)
        trn_correct += (out.argmax(1) == labels).sum().item()
        trn_total   += imgs.size(0)

    model.eval()
    val_loss = val_correct = val_total = 0
    with torch.no_grad():
        for imgs, labels in test_dl:
            labels = torch.tensor([label_map[l.item()] for l in labels])
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            val_loss    += loss.item() * imgs.size(0)
            val_correct += (out.argmax(1) == labels).sum().item()
            val_total   += imgs.size(0)

    scheduler.step()

    ta = trn_correct / trn_total
    va = val_correct / val_total
    tl = trn_loss / trn_total
    vl = val_loss / val_total
    history['train_acc'].append(ta)
    history['val_acc'].append(va)
    history['train_loss'].append(tl)
    history['val_loss'].append(vl)

    elapsed = time.time() - t0
    print(f'Epoch {epoch+1:02d}/{N_EPOCHS}  '
          f'loss {tl:.4f}/{vl:.4f}  '
          f'acc {ta:.2%}/{va:.2%}  '
          f'({elapsed:.1f}s)')

torch.save(model.state_dict(), 'cnn_experiment.pth')
print(f'\nModèle sauvegardé : cnn_experiment.pth')
print(f'Accuracy finale : {history["val_acc"][-1]:.2%}')

# ─────────────────────────────────────────────────────────────
#  Courbes d'apprentissage
# ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, N_EPOCHS + 1)

ax1.plot(epochs_range, history['train_loss'], label='Train')
ax1.plot(epochs_range, history['val_loss'],   label='Val')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(epochs_range, history['train_acc'], label='Train')
ax2.plot(epochs_range, history['val_acc'],   label='Val')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylim(0, 1)
ax2.legend()

cfg = f'Conv:{CONV_LAYERS} | Act:{ACTIVATION} | BN:{USE_BATCHNORM} | Drop:{USE_DROPOUT} | Opt:{OPTIMIZER} lr={LR}'
fig.suptitle(cfg, fontsize=9)
plt.tight_layout()
plt.savefig('cnn_experiment_curves.png', dpi=120)
plt.show()
print('Courbes sauvegardées : cnn_experiment_curves.png')
