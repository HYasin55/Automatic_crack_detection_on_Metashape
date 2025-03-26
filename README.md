# Metashape Surface Analysis Toolkit

A collection of Python scripts for Agisoft Metashape that analyze camera-surface relationships or generate masked imagery.

## Features

### 1. Surface Orientation Analysis (`calculate_distance_and_orientation.py`)
- Calculates the distance between the camera center and the surface point projected from image center
- Calculates the angle between camera optical axis and the estimated surface.
- It projects 4 quadrant points of a circle centered on the image center. The radius of the circle can be modified.
- Two different angle values are estimated, one for vertical and other for horizontal.
- Work in progress: These two angles can be combined to measure the angle between surface normal and optical axis

### 2. Mask-Image Combiner (`combine_mask_and_images.py`)
- Combines camera images with their corresponding masks. It converts masked region to black.
- Saves masked images to new directory

### 3. Projecting Marker (`marker_add_sc`)
- Projects the selected point on the image to the selected surface.
- It is script version of "Add Marker" command.

## Notes
- Tested on Agisoft Metashape version 2.1

## Installation
1. Clone repository or download scripts
2. Download required libraries following the website:
https://agisoft.freshdesk.com/support/solutions/articles/31000136860-how-to-install-external-python-module-to-photoscan-professional-pacakge
3. Execute the scripts with "run script" command