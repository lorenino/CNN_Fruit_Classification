# Documentation — Faster R-CNN Détection de Fruits

## C'est quoi un R-CNN ?

R-CNN signifie **Region-based Convolutional Neural Network**. C'est une famille d'architectures inventée par Ross Girshick (Microsoft Research / Facebook AI) à partir de 2014, spécialement conçue pour la **détection d'objets** : trouver **où** se trouve un objet dans une image ET **ce que c'est**.

Ce n'est pas la même chose qu'un classifieur comme ResNet :

| Classifieur (ResNet) | Détecteur (R-CNN) |
|---|---|
| Entrée : image | Entrée : image |
| Sortie : une étiquette | Sortie : N boîtes + N étiquettes + N scores |
| "C'est une banane" | "Il y a une banane ici (x1,y1,x2,y2) à 94%" |

---

## L'évolution de la famille R-CNN

### R-CNN (2014) — l'original
Fonctionnement naïf :
1. Algorithme classique (Selective Search) propose ~2000 zones candidates dans l'image
2. Chaque zone est redimensionnée et passée **séparément** dans un CNN
3. Un SVM classifie chaque zone

Problème : **extrêmement lent** (47 secondes par image), chaque zone est traitée indépendamment.

### Fast R-CNN (2015)
Amélioration clé : passer l'image entière dans le CNN **une seule fois**, puis extraire les features des zones candidates depuis la feature map résultante. Beaucoup plus rapide, mais Selective Search reste lent.

### Faster R-CNN (2015) — ce qu'on utilise
**L'innovation majeure** : remplacer Selective Search par un **RPN (Region Proposal Network)**, un réseau de neurones qui apprend lui-même à proposer des zones. Le modèle est maintenant entièrement neuronal, end-to-end.

C'est le modèle qu'on a utilisé, et c'est le standard de référence pour la détection d'objets depuis 2015.

---

## Comment fonctionne Faster R-CNN de bout en bout

```
Image d'entrée (224×224×3)
         │
         ▼
┌─────────────────────┐
│      BACKBONE       │  ← réseau pré-entraîné qui extrait les features
│  (MobileNetV3 ici)  │    produit une feature map 2D condensée
└─────────────────────┘
         │
         ▼  feature map (ex: 7×7×256)
┌─────────────────────┐
│        FPN          │  ← Feature Pyramid Network
│  (multi-échelles)   │    combine les features à différentes résolutions
└─────────────────────┘    pour détecter les petits ET grands objets
         │
         ▼
┌─────────────────────┐
│        RPN          │  ← Region Proposal Network
│  (proposeur zones)  │    parcourt la feature map avec des "anchors"
└─────────────────────┘    propose ~300 zones candidates avec score d'objectness
         │
         ▼  ~300 boîtes candidates
┌─────────────────────┐
│     RoI Pooling     │  ← Region of Interest Pooling
│  (normalisation)    │    redimensionne chaque zone candidate à taille fixe
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Box Predictor     │  ← tête de classification (qu'on a remplacée)
│  (FastRCNNPredictor)│    prédit la classe ET affine les coordonnées de la boîte
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│        NMS          │  ← Non-Maximum Suppression (post-traitement)
│  (filtre doublons)  │    élimine les boîtes qui se chevauchent trop
└─────────────────────┘
         │
         ▼
  Boîtes finales + classes + scores
```

### Détail du RPN — le cœur de Faster R-CNN

Le RPN est ce qui distingue Faster R-CNN des versions précédentes. Il glisse sur la feature map et, à chaque position, teste plusieurs **anchors** (boîtes de référence de tailles et ratios différents : petite/carrée, grande/allongée, etc.).

Pour chaque anchor il prédit :
- Est-ce qu'il y a un objet là ? (score 0-1)
- Si oui, comment ajuster la boîte pour qu'elle colle mieux à l'objet ? (4 deltas)

C'est lui qui remplace les 2000 propositions manuelles de Selective Search par un mécanisme appris.

---

## Ce qu'on a fait concrètement

### 1. On n'a pas créé le modèle from scratch

Faster R-CNN est un modèle complexe avec des millions de paramètres. Le créer et l'entraîner de zéro nécessiterait des semaines sur GPU. On a fait du **Transfer Learning** :

```python
weights = models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
model   = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
```

On charge un modèle **pré-entraîné sur COCO** (dataset de 80 classes : chats, voitures, personnes...). Tout le backbone + RPN + FPN arrivent déjà "intelligents".

### 2. On a remplacé uniquement la tête de classification

```python
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
```

Le modèle COCO prédit 80 classes. On remplace juste la dernière couche par une tête qui prédit nos **4 classes** (background, orange, apple, banana). Toute la partie "comprendre les images" reste intacte.

C'est l'équivalent de changer le dernier étage d'un immeuble déjà construit.

### 3. Backbone : pourquoi MobileNetV3 et pas ResNet50

Le notebook Kaggle original utilisait ResNet50. On l'a changé pour **MobileNetV3** pour une raison simple : MPS (Apple Silicon) ne supporte pas toutes les opérations de Faster R-CNN avec ResNet50, ce qui forçait un fallback silencieux sur CPU tout en occupant la mémoire GPU. Résultat : 528 secondes par epoch.

MobileNetV3 est un backbone conçu pour être léger (mobile), il tourne bien sur CPU et donne des résultats comparables pour ce type de tâche simple (3 classes de fruits).

### 4. Dataset — Format Pascal VOC

Le dataset de détection (dans `archive (1)`) n'est pas le même que Fruits-360. Chaque image a un fichier XML associé au format Pascal VOC :

```xml
<object>
  <name>apple</name>
  <bndbox>
    <xmin>45</xmin>
    <ymin>30</ymin>
    <xmax>180</xmax>
    <ymax>160</ymax>
  </bndbox>
</object>
```

Le `FruitsDataset` lit ces XMLs, remet les coordonnées à l'échelle 224×224, et les passe au modèle.

---

## Les choix d'entraînement

### Optimiseur : SGD avec momentum

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
```

C'est la configuration standard pour Faster R-CNN issue du papier original. Adam fonctionne moins bien sur les modèles de détection car les gradients du RPN et de la tête de classification ont des magnitudes très différentes. SGD avec momentum 0.9 est plus stable.

- `lr=0.005` : learning rate faible, on fine-tune un modèle pré-entraîné, on ne repart pas de zéro
- `momentum=0.9` : accumule les gradients passés pour éviter les oscillations
- `weight_decay=5e-4` : régularisation L2, limite l'overfitting

### Device : CPU forcé

```python
DEVICE = 'cpu'
```

MPS (Metal Performance Shaders) d'Apple ne supporte pas toutes les opérations de Faster R-CNN (notamment certaines ops du RPN). Plutôt qu'un fallback silencieux qui consomme mémoire sans accélérer, on force CPU directement.

### Loss : 4 composantes

Faster R-CNN ne renvoie pas une seule loss mais un dictionnaire :

```
loss_classifier   : erreur de classification des boîtes (cross-entropy)
loss_box_reg      : erreur de localisation (smooth L1)
loss_objectness   : erreur du RPN sur "est-ce un objet ?"
loss_rpn_box_reg  : erreur du RPN sur l'ajustement des anchors
```

On les somme toutes :
```python
loss = sum(losses.values())
```

C'est le résultat qu'on a obtenu au premier test : `{'loss_classifier': 1.6037, 'loss_box_reg': 0.3714, 'loss_objectness': 0.0032, 'loss_rpn_box_reg': 0.0095}`.

La `loss_classifier` élevée à 1.6 au départ est normale : la nouvelle tête de classification part aléatoirement. En 5 epochs elle converge.

### Évaluation en mode train()

```python
with torch.no_grad():
    model.train()   # intentionnel
    losses = model(imgs, targets)
```

Faster R-CNN ne renvoie les losses **que** quand il est en mode `train()`. En mode `eval()`, il renvoie les prédictions. Pour calculer la val loss sans modifier les poids, on combine `model.train()` + `torch.no_grad()`.

### 5 epochs — suffisant ?

Pour du fine-tuning sur 3 classes avec un backbone déjà pré-entraîné : oui. Le modèle n'apprend pas à "voir", il apprend juste à reconnaître pomme/orange/banane dans les features qu'il extrait déjà bien. Plus d'epochs amèneraient peu de gain et risqueraient l'overfitting.

---

## Ce que fait `test_rcnn.py`

```
Image → resize 224×224 → tensor → modèle (eval) → boxes + scores + labels
     → filtre conf < 0.4 → NMS (iou 0.3) → affichage matplotlib
```

Le seuil de confiance à 0.4 élimine les détections incertaines. Le NMS (Non-Maximum Suppression) à IoU 0.3 élimine les boîtes qui se chevauchent trop : si deux boîtes détectent "apple" au même endroit avec un chevauchement > 30%, on ne garde que la meilleure.

---

## Résumé

On a pris un modèle Faster R-CNN existant (entraîné sur COCO par Facebook AI), on a remplacé uniquement sa tête de classification pour l'adapter à nos 3 fruits, et on l'a fine-tuné 5 epochs sur notre dataset Pascal VOC. Le backbone MobileNetV3 a été choisi pour sa compatibilité CPU/MPS sur Mac. L'architecture complète (RPN, FPN, RoI Pooling) est héritée du papier original de Ren et al. (2015) et implémentée dans TorchVision.
