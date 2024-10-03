# Plant Disease Classifier

## Overview

The Plant Disease Classifier is a web application powered by FastAPI and TensorFlow, designed to assist in the detection of plant diseases through image analysis. Users can upload images of their plants, and the application utilizes a trained Convolutional Neural Network (CNN) model to predict the health status of the plant, providing predictions with confidence levels.


## Table of Contents

- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Features

- User-friendly interface for uploading images.
- Real-time predictions for plant diseases.
- Confidence scores for each prediction.
- Built using the FastAPI framework for performance and scalability.
- TensorFlow for deep learning model implementation.

## Technologies

- **FastAPI**: For building the web application and handling API requests.
- **TensorFlow**: For creating and deploying the Convolutional Neural Network.
- **Pillow**: For image processing tasks.
- **Bootstrap**: For responsive web design.
- **JavaScript**: For asynchronous communication with the backend.

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd plant_disease_classifier

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt

## Usage


2. **Run the application:**
   ```bash
   uvicorn app:app --reload
3. Access the application:
Open your web browser and navigate to http://127.0.0.1:8000.

4. Upload an image:
Use the provided interface to upload an image of a plant to receive a prediction about its health.

## Model Training
To train the CNN model on your dataset, you can execute the model_training.py script. The trained model will be saved in the specified directory (models/potatoes.h5 by default).

2. **Run the training script:**
   ```bash
   python model_training.py
Make sure to adjust the training parameters in config.py as per your requirements.

Contributing
Contributions are welcome! To contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/YourFeature).
3. Make your changes and commit them (git commit -m 'Add your feature').
4. Push your branch (git push origin feature/YourFeature).
5. Open a pull request to merge your changes. 

Please ensure that your code follows the project’s style guide and is well-documented.


Author: [Ali jahani]

Email: [alijahani1919@gmail.com]

