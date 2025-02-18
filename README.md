# Comment-Toxicity

## Overview
Comment-Toxicity is a machine learning project that detects and scores the toxicity of text comments. It uses a deep learning model with an embedding layer and a bidirectional LSTM network to classify comments as toxic or non-toxic. The model is deployed using Gradio to provide an interactive user interface for real-time predictions.

## Features
- **Deep Learning Model**: Uses an Embedding layer with Bidirectional LSTM.
- **Multi-Layer Neural Network**: Includes dense layers for feature extraction and classification.
- **Gradio Interface**: Provides an easy-to-use web-based UI for users to input comments and get toxicity scores.
- **Kaggle Dataset**: Uses a preprocessed dataset of comments labeled for toxicity.

## Technologies Used
- Python
- TensorFlow/Keras
- NumPy & Pandas
- Gradio (for UI)
- Kaggle API (for dataset retrieval)

## Installation
### 1. Clone the repository:
```sh
git clone https://github.com/yourusername/comment-Toxicity.git
cd comment-Toxicity
```
### 2. Install dependencies:
```sh
pip install -r requirements.txt
```
### 3. Download the dataset from Kaggle:
```sh
pip install kaggle
kaggle datasets download -d <dataset-name>
unzip dataset.zip
```

## Model Summary
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

model = Sequential()
model.add(Embedding(MAX_FEATURES+1, 32))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='sigmoid'))

model.summary()
```

## Running the Application
```sh
python app.py
```
This will launch the Gradio web interface where users can input comments to check toxicity levels.

## Usage Example
```python
import gradio as gr

def score_comment(text):
    # Preprocess text and make predictions
    return model.predict([text])

interface = gr.Interface(fn=score_comment, 
                         inputs=gr.Textbox(lines=2, placeholder='Enter a comment...'),
                         outputs='text')

interface.launch()
```

## Contributing
Feel free to fork this repository, make improvements, and submit pull requests. Contributions are always welcome!

## License
This project is licensed under the MIT License.

## Author
-(https://github.com/Manpreet-kour)

---
Happy coding! 🚀
