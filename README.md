# EEC 174A/B Senior Design Project: Multimodal / Multispectral Imaging for Agriculture

## Problem
Multispectral imagery can offer key targeted information to plant breeders and farmers to make informed decisions on crop stress before obvious visual cues appear, but has a high cost of deployment.

Additionally, trait extraction in the low-altitude imaging case with plot and plant-level resolution typically requires large labeled datasets. Manual labeling presents an expensive and error-prone process, especially in agricultural cases where labeling standards have yet to be rigorously defined.

## Proposed Solution
To solve the aforementioned problems, we propose a four-band Raspberry Pi 5-based imaging setup that is flexible for both low and high altitude deployment based on its small form factor. 

To allow ratios such as NDVI to be extracted at a plant and organ-level, we use image feature alignment techniques found in `/lightGlue` to apply homography maps to the each band image to minimize the effects of parallax. Additionally, we use this alignment pipeline to apply super-resolution methods.

Of the four selected bands for deployment, the 685nm band isolates the flowers for garbanzo and cowpea data extremely well. By using this band together with OpenCV blob detection using a threshold for flower brightness relative to the darker canopy, we are able to extract traits at a high speed, though accuracy is sacrificed. Building on this, we investigate the use of a trained YOLOv26n model and benchmark its performance in a deployment setting.

## Repository Components
**Feature Alignment**: `/lightGlue`
**Edge Deployment**: `UPLOAD SOON`
**Super-resolution**: `/super-res`, `/Restormer`
**Auto-labeling**: `/utils/prelabel_flowers.py`
**Ratio Extraction**: `UPLOAD SOON`

## Parts List
- Raspberry Pi 5
- RTC battery for Raspberry Pi 5
- Portable charger / power bank
- Arducam 1MP*4 Quadrascopic OV9281 Global Shutter Monochrome Camera Bundle Kit
- 4x MidOpt filters: 685, 725, 750, and 1000nm
