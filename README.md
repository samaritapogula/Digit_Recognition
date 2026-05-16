Spoken Digit Recognition from Noisy Audio

A high-performing deep learning solution for recognizing spoken digits (0-9) from short audio clips.

Kaggle Private Score: 0.99509

⸻

Overview

This project was developed for a Kaggle competition where the objective is to classify spoken digits from 1-second .wav audio recordings.

The challenge is designed to test generalization to:

* Unseen speakers
* Background noise
* Augmented and distorted recordings
* Variations in recording conditions

To achieve strong performance, this solution combines:

* Mel-spectrogram feature extraction
* Convolutional Neural Networks (CNNs)
* Multiple spectrogram resolutions
* Test-Time Augmentation (TTA)
* Weighted ensemble averaging

⸻

Final Result

Metric	Score
Private Leaderboard Score	0.99509
Number of Classes	10
Audio Duration	1 second
Sample Rate	16 kHz

⸻

Dataset Structure

train_audio/{id}.wav
train.csv
test_audio/{id}.wav
test.csv

train.csv

Column	Description
id	Audio file identifier
label	Digit class (0-9)

test.csv

Column	Description
id	Audio file identifier

⸻

Project Structure

.
├── notebook.ipynb
├── train.csv
├── test.csv
├── train_audio/
├── test_audio/
├── best_model_64.h5
├── best_model_80.h5
├── best_model_128.h5
├── submission.csv
└── README.md

⸻

Approach Summary

1. Audio Preprocessing

Each audio clip is:

* Loaded using librosa
* Resampled to 16,000 Hz
* Truncated or zero-padded to exactly 1 second
* Converted into Mel-spectrograms
* Transformed to decibel scale

⸻

Feature Extraction

Three Mel-spectrogram resolutions are used:

* 64 Mel bins
* 80 Mel bins
* 128 Mel bins

Each resolution captures complementary information:

* Lower resolutions provide compact and robust representations
* Higher resolutions capture finer spectral details

⸻

Model Architecture

A custom CNN is trained separately for each Mel resolution.

Architecture

Input Mel-Spectrogram
    ↓
Conv2D(32) × 2 + BatchNorm + MaxPool
    ↓
Conv2D(64) × 2 + BatchNorm + MaxPool
    ↓
Conv2D(128) + BatchNorm + MaxPool
    ↓
Conv2D(256) + BatchNorm + MaxPool
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(10, Softmax)

Key Design Choices

* Batch Normalization improves convergence and stability
* Global Average Pooling reduces overfitting
* Dropout (0.5) increases generalization
* Adam optimizer with learning rate scheduling

⸻

Training Strategy

Each of the three models is trained independently.

Training Parameters

Parameter	Value
Epochs	40
Batch Size	32
Validation Split	20%
Optimizer	Adam
Loss	Sparse Categorical Crossentropy
Metric	Accuracy

Callbacks

* ModelCheckpoint — saves the best model by validation accuracy
* ReduceLROnPlateau — lowers learning rate when validation loss stalls

⸻

Test-Time Augmentation (TTA)

During inference, each test spectrogram is perturbed multiple times by adding small Gaussian noise.

noise ~ N(0, 0.01)

Settings

* Number of augmentations: 7
* Predictions are averaged across all augmented versions

Benefit

TTA improves robustness against noise and slight distortions in the test data.

⸻

Ensemble Strategy

Predictions from the three models are combined using weighted averaging.

Model	Weight
64 Mel bins	0.30
80 Mel bins	0.40
128 Mel bins	0.30

The final prediction is:

Final = 0.3 × Pred64 + 0.4 × Pred80 + 0.3 × Pred128

The digit with the highest probability is selected.

⸻

Why This Works So Well

This solution performs strongly because it combines several complementary techniques:

1. Multi-resolution features capture both coarse and fine acoustic patterns.
2. CNNs effectively learn local time-frequency structures.
3. TTA reduces prediction variance.
4. Weighted ensembling leverages strengths of each model.
5. Learning rate scheduling improves convergence.

⸻

Installation

Clone Repository

git clone https://github.com/yourusername/spoken-digit-recognition.git
cd spoken-digit-recognition

Install Dependencies

pip install -r requirements.txt

Recommended requirements.txt

numpy
pandas
librosa
tqdm
tensorflow
scikit-learn

⸻

Usage

Train Models

Open and run the notebook:

jupyter notebook notebook.ipynb

Generate Submission

Running the full notebook will produce:

submission.csv

⸻

Submission Format

id,label
00001,7
00002,3
00003,0

Note: Ensure the submission column is named label to match competition requirements.

⸻

Performance Tips

If you want to improve the score further, consider:

* K-fold cross-validation
* Mixup or SpecAugment
* 1D audio models
* Pretrained models (e.g., YAMNet, PANNs)
* Snapshot ensembling

⸻

Memory Requirements

Since the dataset is large (~3.3 GB), loading all spectrograms into memory can require significant RAM.

Recommended:

* 16 GB RAM or more for local training
* Kaggle GPU notebooks for faster execution

⸻

Reproducibility

For fully reproducible results, set random seeds for:

* NumPy
* TensorFlow
* Python

⸻

Future Improvements

Potential enhancements include:

* Speaker normalization
* SpecAugment
* Attention-based architectures
* EfficientNet over spectrograms
* Probability calibration

⸻

Key Learnings

This project demonstrates that a relatively simple CNN can achieve near-perfect accuracy when combined with:

* Strong feature engineering
* Multi-scale representations
* Test-time augmentation
* Weighted ensembling

⸻

Acknowledgements

* Competition organizers for the dataset
* urlLibrosahttps://librosa.org for audio processing
* urlTensorFlow/Kerashttps://www.tensorflow.org for deep learning
* urlKagglehttps://www.kaggle.com for the competition platform

⸻

Author

Your Name

If you found this repository helpful, consider starring the project.

⸻

License

This project is released under the MIT License.
