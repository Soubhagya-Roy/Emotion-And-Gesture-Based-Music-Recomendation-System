
<div align="center">
 
   <img src="https://media.istockphoto.com/id/1162732645/vector/hand-drawn-music-instruments-orcestra-horizontal-banner-or-cover-for-social-media-ink-style.jpg?s=612x612&w=0&k=20&c=lGSh2yGEyVtxoGFMq5joeGo8tnBwqrHDoPUFM_tEoIg=" width="900">
 
</div>
 

 
# Emotion And Gesture Based Music Recomendation System
 

 
## Overview
 
This project presents a hybrid music recommendation system that leverages both **hand gestures** and **facial emotions** to personalize song suggestions. By combining gesture recognition and emotion analysis, the system enhances user interaction and provides a tailored auditory experience.
 

 
## Key Features
 
- **Gesture Recognition** using **MediaPipe** and **TensorFlow**.<br>
 
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcREjN3y5sR8dtGkoOwaks7KShZA-aUvSM3xIQ&s" width="95">&nbsp;<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQRt0OU-18slF5XnJ0no2b9gA8twntZG6UngQ&s" width="100"/>
 
- **Emotion Detection** using **Facial Expression Recognizer (FER)**.<br>
 
  <img src="https://miro.medium.com/v2/resize:fit:753/1*slyZ64ftG12VU4VTEmSfBQ.png" width="100">&nbsp;<img src="https://miro.medium.com/v2/resize:fit:600/format:webp/1*e_7bN4nfREd0KGai-eQzGQ.gif" width="100"><br>
 
- Songs are recommended based on detected gestures (higher priority) and emotions.<br>
 
- Utilizes the concept of **Navarasa** (nine emotions) mapped to **Melakarta ragas**.<br>
 

 
- Built with Python, OpenCV, Tkinter (for GUI), and Pygame (for music playback).<br>
 
   <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQGvGShLAJbL5g1fezQUTHYX7zWX7XRXmNv8A&s" width="100">&nbsp;<img src="https://3.bp.blogspot.com/-yvrV6MUueGg/ToICp0YIDPI/AAAAAAAAADg/SYKg4dWpyC43AAfrDwBTR0VYmYT0QshEgCPcBGAYYCw/s1600/OpenCV_Logo.png" width="114">&nbsp;<img src="https://pbs.twimg.com/profile_images/1137034734872203266/BMH5Eplh_400x400.png" width="101">&nbsp;<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRgoqby6z-8P6NRa3cPjz8He-GrRNDo0OotEA&s" width="101">
 

 
## Datasets Used
 
- **FER2013**: For facial emotion classification into seven emotions.<br>
 
  <img src="https://datasets.activeloop.ai/wp-content/uploads/2022/09/FER2013-1024x613.png" width="200"><br>
 
- **Hand Gesture Recognition Dataset (Kaggle)**: 24,000 images across 20 gestures.<br>
 
  <img src="https://miro.medium.com/v2/resize:fit:1400/0*UXTIVr1ghJ30DOya.png" width="400"><br>
 

 
## Model Architecture
 
- Hand gesture classification using DenseNet201.
 
- Emotion detection via CNN trained on FER2013.
 
- Combined accuracy of 88%.
 

 
## Use Cases
 
- Music recommendation tailored to real-time emotional and gestural inputs.
 
- Interface designed for ease of use, especially helpful for hands-free scenarios.


## Note :-

This project is developed for **academic and research purposes** music recommendation system that leverages both **hand gestures** and **facial emotions** to personalize song suggestions. Any implementation in real-world scenarios would require thorough testing, ethical review, and compliance with relevant legal frameworks.. For any external or commercial use, please contact the authors for permissions.
