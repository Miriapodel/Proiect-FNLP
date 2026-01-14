Preprocessing data commands:
python src/preprocess/preprocess_liar.py
python preprocessing.py

Running baselines for each dataset:
python -m src.models.baseline_bow --task liar
python -m src.models.baseline_bow --task rumour
