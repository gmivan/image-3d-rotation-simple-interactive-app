# Medical Image 3D Rotation

This Space provides an interactive 3D visualization tool for medical images and their segmentation masks. You can upload your own medical images and masks, then rotate them in 3D space to better understand the spatial relationships.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/anonimous777/sennet-image-3d-rotation)

## Features
- Upload medical images and their corresponding masks
- Interactive 3D rotation with sliders for X, Y, and Z axes
- Real-time visualization of the rotation process
- Overlay of segmentation masks on the rotated images
- Support for both 8-bit and 16-bit medical images

## How to Use
1. Upload your medical image (supports common medical image formats)
2. Upload the corresponding mask (should be a binary mask)
3. Adjust the rotation angles using the sliders:
   - X-axis rotation: -180° to 180°
   - Y-axis rotation: -180° to 180°
   - Z-axis rotation: -180° to 180°
4. Click "Rotate" to see the 3D visualization

The visualization shows:
- Original slice in 3D
- Rotated coordinates in 3D
- Final rotated slice with mask overlay

## Examples

### Example 1: Kidney Vessel Segmentation
Try uploading one of the sample images from the blood vessel segmentation dataset:

Image: [data/train/kidney_3_sparse/images/0496.tif](data/train/kidney_3_sparse/images/0496.tif)
Mask: [data/train/kidney_3_dense/labels/0496.tif](data/train/kidney_3_dense/labels/0496.tif)

### Example 2: Different Rotation Angles
Try these rotation combinations:
- X: 45°, Y: 0°, Z: 0° - Tilt forward
- X: 0°, Y: 45°, Z: 0° - Tilt right
- X: 0°, Y: 0°, Z: 45° - Rotate clockwise
- X: 45°, Y: 45°, Z: 45° - Complex rotation

## Data Directory Structure
```
data/
├── train/
│   ├── kidney_3_sparse/
│   │   ├── images/          # Original CT images
│   │   └── images_enhanced/ # Enhanced versions
│   └── kidney_3_dense/
│       └── labels/          # Vessel segmentation masks
```

## Technical Details
The app uses:
- Gradio for the web interface
- Matplotlib for 3D visualization
- SciPy for 3D rotation and interpolation
- OpenCV for image processing

## Live Demo
Try the interactive demo on Hugging Face Spaces:
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/anonimous777/sennet-image-3d-rotation)

## Local Development
1. Clone the repository:
```bash
git clone https://github.com/gmivan/image-3d-rotation-simple-interactive-app.git
cd image-3d-rotation-simple-interactive-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
python app.py
```