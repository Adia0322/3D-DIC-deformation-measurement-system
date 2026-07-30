
# 3D DIC deformation measurement system 
A stereo vision-based **3D displacement measurement system** using  
**Digital Image Correlation (DIC)** for high-precision surface deformation analysis. 

## Overview
This project implements a optimized **Stereo Digital Image Correlation (Stereo-DIC)** method  
to reconstruct and measure **3D surface displacement** using two webcams.

It is designed for:
- High-precision deformation measurement
- Fast computation pipeline
- Low-cost stereo vision setup

## Features
* High-Precision 3D Displacement Measurement
* High Efficiency
* Low-Cost Camera-Based Setup
* User-Friendly Interface & Configuration

## Example:
Below shows displacement measurement of a **deformed rubber surface with speckle pattern**:
## How to run:
* Create new venv
```shell
python -m venv venv
.\venv\Scripts\Activate.ps1 # Activating
```
if it show error msg while Activating, run below command first:
```shell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
then try again.

* Install Modules in venv:
```shell
pip install opencv-python
pip install matplotlib
```
* Clone with submodules
```shell
git clone --recurse-submodules <URL>
```

### STEP1: capture checkerboard images (10 more)
* Objection:   
correct the two images to parallel to each others, make the result more precise  
* How to:  
capture severals checkerboards image, and run stereo camera calibration to obtain the intrinsic and extrinsic params. Use params to rectify images.  

Ternminal:
  
```
python -m stereo_vision.apps.STEP1_GenerateCBImage
```
### left camera:　　

![camera1 image2](https://github.com/Adia0322/Stereo-digital-image-correlation/assets/89566671/f4016d4d-7cb0-4e49-b08b-fdda729ab5be)　　


### right camera:　　

![camera2 image2](https://github.com/Adia0322/Stereo-digital-image-correlation/assets/89566671/9a6a3aaa-752d-4090-aebc-e46d18fc7bf4)　　

  

### STEP2: camera calibration
Run stereo camera calibration to obtain the intrinsic and extrinsic params, and use it to rectify images.

```
python -m stereo_vision.apps.STEP2_StereoCalibration
```

![step2](https://github.com/Adia0322/Stereo-digital-image-correlation/assets/89566671/b11a0ef2-1789-4eb8-ad4d-472e5f6ff5a9)


### STEP3: check image correaltion results (optional)
You can test if the calibration works in this step
```
python -m stereo_vision.apps.STEP2_StereoCalibration
```

### STEP4: measure the displacement of the surface on rubber
Run the following command from the project root directory (3D-DIC_measurement_system/):  
* Build:
```shell
mingw32-make all
```
* Run 3D measurement
```shell
python -m stereo_vision.apps.STEP4_DisplacementField
```
And you can obtain the average displacemnt of rubber in surface  
> image will be uploaded later

> image will be uploaded later

> image will be uploaded later

## Example Result
