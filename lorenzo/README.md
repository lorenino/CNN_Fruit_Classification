# Lorenzo Faloci — CNN Fruit Classification

## Approche : CNN From Scratch + Transfer Learning (EfficientNetB0)
**Dataset** : Fruits-360 — 255 classes, 90 000+ images, 100×100px, fond blanc
**Framework** : TensorFlow / Keras | **Runtime** : Google Colab (GPU T4 → CPU)

---

## Résultats

| Modèle | Val Accuracy | Val Loss | Epochs | Temps | Params |
|--------|:-----------:|:--------:|:------:|:-----:|:------:|
| CNN Scratch | **99.53%** | 0.0160 | 20 | 150 min | 4.87M |
| TL EfficientNetB0 *(Phase 1)* | 97.24% | 0.0740 | 7/10 | 53 min | 5.33M |

> **Note TL** : Phase 2 (fine-tuning) non effectuée — quota GPU épuisé. Avec fine-tuning complet, attendu **> 99%**.

---

## Visualisations

| Plot | Description |
|------|-------------|
| `plots/training_comparison.png` | Courbes loss & accuracy des 2 modèles (toutes epochs) |
| `plots/model_comparison_bar.png` | Comparatif accuracy / temps / paramètres |
| `plots/confusion_matrices.png` | Matrices de confusion normalisées — 30 classes |
| `plots/gradcam_examples.png` | GradCAM — zones d'attention du CNN sur les fruits |
| `plots/predictions_grid.png` | Prédictions CNN Scratch vs TL côte à côte (12 exemples) |

---

## Architecture CNN Scratch

```
Input(100×100×3)
  → Conv2D(32) + BatchNorm + ReLU + MaxPool       # détection de bords
  → Conv2D(64) + BatchNorm + ReLU + MaxPool       # textures
  → Conv2D(128) + BatchNorm + ReLU + MaxPool      # formes complexes
  → Flatten → Dense(256) + ReLU → Dropout(0.5)
  → Dense(255, softmax)
```

**Pourquoi ces choix ?**
- **3 blocs conv** : suffisant pour 100×100 (au-delà, feature map trop petit)
- **32→64→128** : progression classique, plus de filtres = features plus abstraites
- **BatchNorm** : stabilise l'entraînement, permet LR plus élevé
- **Dropout(0.5)** : prévient l'overfitting sur le fully-connected

---

## Architecture Transfer Learning

```
Input(100×100×3)
  → Rescaling(255.0)          ← FIX CRUCIAL
  → EfficientNetB0 (gelé)     ← features ImageNet
  → GlobalAveragePooling2D
  → Dropout(0.3)
  → Dense(255, softmax)
```

**Pourquoi ces choix ?**
- **EfficientNetB0** : plus léger de la famille (5.3M params), bon ratio perf/vitesse
- **Rescaling(255.0)** : EfficientNet attend [0,255] mais ImageDataGenerator normalise en [0,1] — sans ce fix, accuracy bloquée à ~0.7% (aléatoire)
- **GlobalAveragePooling** : réduit le risque d'overfitting vs Flatten sur des features pré-entraînées
- **Phase 1 (backbone gelé)** : entraîne d'abord la tête — évite de "casser" les features ImageNet

---

## Décisions clés & leçons apprises

### 1. Pourquoi CNN Scratch ≈ Transfer Learning ici ?
Fruits-360 est un dataset **"facile"** :
- Fond blanc uniforme → le CNN n'a pas besoin de features complexes pour séparer les classes
- Images studio standardisées → peu de variabilité intra-classe
- Sur des vraies photos de fruits (supermarché, jardin), le TL serait **nettement supérieur**

### 2. Le bug EfficientNet
EfficientNet (et d'autres modèles Keras comme ResNet, InceptionV3) intègre sa propre normalisation **interne** et attend des pixels en `[0, 255]`. Si on passe des images déjà normalisées en `[0, 1]`, le modèle reçoit des valeurs 255× trop petites → toutes les activations sont proches de zéro → accuracy ~0.7% (équivalent à prédire aléatoirement sur 255 classes). **Fix** : ajouter `tf.keras.layers.Rescaling(255.0)` comme première couche.

### 3. Vitesse de convergence
Le TL atteint **97%+ dès l'epoch 1** grâce aux features ImageNet pré-apprises. Le CNN Scratch met 5-6 epochs pour dépasser 95%. En production où le GPU coûte cher, l'avantage du TL est décisif.

### 4. Callbacks utilisés
- **EarlyStopping(patience=5)** : arrête si val_loss ne s'améliore plus → évite l'overfitting, restaure les meilleurs poids
- **ReduceLROnPlateau(factor=0.5, patience=3)** : réduit le LR quand le val_loss stagne → affinage fin de la convergence

---

## Fichiers

```
lorenzo/
├── README.md                        ← ce fichier
├── results.json                     ← toutes les métriques + explications des choix
├── fruit_classification_cnn.ipynb   ← notebook Colab complet
└── plots/
    ├── training_comparison.png
    ├── model_comparison_bar.png
    ├── confusion_matrices.png
    ├── gradcam_examples.png
    └── predictions_grid.png
```
