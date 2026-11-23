# Spotify Hit/Flop Classification

An educational machine learning project by Abderrahmane Belghiti, originally developed in 2025. A TensorFlow/Keras neural network classifies music tracks as **Hit** or **Flop** using numerical audio features.

## Dataset

The supplied `dataset-of-10s.csv` contains **6,398 tracks**, equally divided between 3,199 hits and 3,199 flops, with 15 numerical input features. Track names, artists and Spotify URIs are excluded from training.

According to the accompanying project notes, a hit appeared in the Billboard Hot 100 during the relevant decade. Flops were selected using additional artist, genre and US-market criteria. These labels describe the dataset's selection rules, not musical quality. The original dataset author, source URL and redistribution license were not included in the supplied project files; attribution remains incomplete.

## Method

- Explore class balance, danceability versus energy, and feature correlations.
- Shuffle the data and split it into 80% training, 10% validation and 10% test data.
- Standardize features using statistics fitted on the training set only.
- Train a multilayer perceptron with three 50-unit ReLU hidden layers and a two-unit softmax output.
- Use Adam, sparse categorical cross-entropy, batches of 300 and up to 100 epochs. Early stopping monitors validation loss with patience 3 and restores the best weights.
- Evaluate test accuracy and inspect individual test tracks through notebook widgets.

## Run

Use Python 3.11 in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab spotify_hit_flop.ipynb
```

Run every cell from top to bottom. Keep the CSV in the same directory as the notebook. No Spotify API credentials are required.

## Results and limitations

The original saved notebook reported **80.50% test accuracy**. That historical run standardized the entire dataset before splitting, which leaked test-set statistics. The published notebook fixes this and also corrects the alignment of track titles in the prediction widget. Its outputs have been cleared; the historical accuracy is **not a verified result for the corrected version**, which must be rerun.

This is a retrospective classification exercise, not a reliable forecast of future chart success or a recommendation system. The random split does not isolate artists or release dates, so generalization to unseen artists and future releases is not established. Softmax scores are not calibrated probabilities. NumPy and TensorFlow seeds are set, but results can vary across software and hardware environments.
