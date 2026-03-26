# CNN Fruit Classification — Benchmark collectif

Comparaison de différentes approches CNN/Deep Learning sur la classification d'images.
Chaque membre crée un dossier à son prénom avec son travail.

## Structure

```
CNN_Fruit_Classification/
├── lorenzo/        ← CNN Scratch + Transfer Learning (EfficientNetB0) sur Fruits-360
├── prenom2/        ← à compléter
└── prenom3/        ← à compléter
```

## Comment contribuer

1. Crée un dossier **à ton prénom** à la racine du repo
2. Mets-y :
   - `README.md` — description de ton approche + décisions + résultats
   - `results.json` — métriques standardisées (voir format ci-dessous)
   - `notebook.ipynb` — ton notebook Colab
   - `plots/` — tes visualisations (courbes, confusion matrix, etc.)
3. `git add prenom/ && git commit -m "Add prenom results" && git push`

## Format results.json minimal

```json
{
  "author": "Ton Prénom",
  "dataset": "Nom du dataset",
  "num_classes": 255,
  "models": {
    "nom_modele": {
      "val_accuracy": 0.99,
      "val_loss": 0.02,
      "epochs": 20,
      "train_time_min": 60,
      "params_millions": 5.0
    }
  },
  "key_observations": ["leçon 1", "leçon 2"]
}
```
