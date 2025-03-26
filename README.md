# 3D Crack Detection & Measurement in Metashape
Automated crack detection and 3D measurement system for photogrammetric models using deep learning and computer vision techniques.
This is the repository of the journal paper "Automated Crack Width Measurement in 3D Models: A Photogrammetric Approach with Image Selection". 

## Features
- 🕵️‍♂️ Crack detection using U-Net deep learning models
- 📏 Hybrid crack width measurement algorithm
- 🌐 3D marker projection onto Metashape models
- 📊 Automated scale bar creation and measurement logging
- 📸 Camera selection based on surface orientation analysis

## Installation
1. Clone repository or download scripts
2. Download required libraries following the website:
https://agisoft.freshdesk.com/support/solutions/articles/31000136860-how-to-install-external-python-module-to-photoscan-professional-pacakge
3. Download the model under the "out_model" folder from:
https://drive.google.com/drive/folders/1-hCmQVZA_kQqLYHwt4e_k8Z5-44nDGQl?usp=share_link
4. Execute the scripts with "run script" command

### Prerequisites
- Agisoft Metashape Professional (1.8.0+)
- Python 3.8+ with pip
- List of the required packages will be add

## Notes
- Crack detection model is developed using the repository:
https://github.com/khanhha/crack_segmentation
- Crack edge selection algorithm modified version of the repository:
https://github.com/JeremyOng96/A-Hybrid-Method-for-Pavement-Crack-Width-Measurement
