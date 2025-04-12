# Emotion-And-Gesture-Based-Music-Recomendation-System
Recomendation System using Facial expression or Hand Gesture


## Overview
This project presents a hybrid music recommendation system that leverages both **hand gestures** and **facial emotions** to personalize song suggestions. By combining gesture recognition and emotion analysis, the system enhances user interaction and provides a tailored auditory experience.

## Key Features
- **Gesture Recognition** using **MediaPipe** and **TensorFlow**.<br>
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcREjN3y5sR8dtGkoOwaks7KShZA-aUvSM3xIQ&s" width="95">&nbsp;<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQRt0OU-18slF5XnJ0no2b9gA8twntZG6UngQ&s" width="100"/>
- **Emotion Detection** using **Facial Expression Recognizer (FER)**.<br>
  <img src="https://miro.medium.com/v2/resize:fit:753/1*slyZ64ftG12VU4VTEmSfBQ.png" width="100"><br>
- Songs are recommended based on detected gestures (higher priority) and emotions.<br>
   <img src="https://miro.medium.com/v2/resize:fit:600/format:webp/1*e_7bN4nfREd0KGai-eQzGQ.gif" width="100"><br>
- Utilizes the concept of **Navarasa** (nine emotions) mapped to **Melakarta ragas**.
- Built with Python, OpenCV, Tkinter (for GUI), and Pygame (for music playback).

## Datasets Used
- **FER2013**: For facial emotion classification into seven emotions.
- **Hand Gesture Recognition Dataset (Kaggle)**: 24,000 images across 20 gestures.

## Model Architecture
- Hand gesture classification using DenseNet201.
- Emotion detection via CNN trained on FER2013.
- Combined accuracy of 88%.

## Use Cases
- Music recommendation tailored to real-time emotional and gestural inputs.
- Interface designed for ease of use, especially helpful for hands-free scenarios.

## Conclusion
The project achieved its goal of integrating gestures and emotions for dynamic music recommendation. With a modular and scalable design, it opens avenues for applications in multimedia interaction and assistive technologies.

## Future Scope
- Improved real-time adaptability and feedback.
- Context-aware music selection.
- Integration with other domains like video content and virtual assistants.

## Contributors
Each member contributed to various aspects including UI design, dataset handling, model implementation, and documentation.

