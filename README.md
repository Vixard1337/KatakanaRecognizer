# KatakanaRecognizer

Engineering thesis project focused on handwritten Japanese Katakana character recognition using Deep Learning and NoSQL data modeling.

The application allows drawing characters, preprocessing images, training a neural network model and performing predictions accelerated using GPU.

## Project Description

The goal of the project is to design and implement a system capable of recognizing handwritten Katakana characters using a Convolutional Neural Network (CNN).



The system includes:

- image preprocessing pipeline,

- neural network training,

- prediction and evaluation,

- graphical interface for drawing characters,

##  Development Environment

Due to discontinued TensorFlow GPU support on native Windows, the project was developed using WSL2.



### Operating System

WSL2 – Kali Linux

Linux 5.15.153.1-microsoft-standard-WSL2 x86\_64 GNU/Linux



### Python Environment

Python 3.9.20

Conda 24.7.1 (Miniconda)

CUDA 11.2 

GPU acceleration was enabled using CUDA and cuDNN.



## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- WSL2 (Kali Linux)
- VS Code
- Git \& GitHub



## Project Structure

src/ - model implementation and utilities

tests/ - unit tests

notebooks/ - data exploration

scripts/ - automation scripts

gui.py - drawing interface



## Installation

Clone repository:

bash

git clone https://github.com/Vixard1337/KatakanaRecognizer.git

cd KatakanaRecognizer

Install dependencies:

pip install -r requirements.txt


Running the Project

Train model

python src/train.py

Evaluate model

python src/evaluate.py

Run GUI

python gui.py

Running Tests

python3 -m unittest tests/test\\\_data\\\_preprocessing.py

or

pytest tests/

Working with WSL2

Project files can be accessed from Windows Explorer using:

\\\\\\\\wsl$\\\\kali-linux\\\\home\\\\admin

Academic Purpose

This repository was created as part of an engineering thesis focused on machine learning systems and NoSQL database design.


License

Educational use only.





