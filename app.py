 import gradio as gr
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import cv2

def create_3d_visualization(image, mask, angle_x, angle_y, angle_z):
    """Create 3D visualization of rotated image and mask"""
    # Convert to numpy arrays if they aren't already
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    if isinstance(mask, str):
        mask = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
    
    # Convert RGB to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Normalize image to 0-1 range
    image = image.astype(np.float32) / 65535.0 if image.dtype == np.uint16 else image.astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 65535.0 if mask.dtype == np.uint16 else mask.astype(np.float32) / 255.0
    
    # Create rotation matrix
    r = Rotation.from_euler('xyz', [angle_x, angle_y, angle_z], degrees=True)
    rotation_matrix = r.as_matrix()
    
    # Get coordinates
    h, w = image.shape
    y, x = np.mgrid[0:h, 0:w]
    
    # Center the coordinates
    x = x - w/2
    y = y - h/2
    
    # Create 3D coordinates
    coords = np.stack([x, y, np.zeros_like(x)], axis=-1)
    
    # Apply rotation
    rotated_coords = np.dot(coords, rotation_matrix.T)
    
    # Move coordinates back
    rotated_coords[..., 0] += w/2
    rotated_coords[..., 1] += h/2
    
    # Create interpolators
    y_coords = np.arange(h)
    x_coords = np.arange(w)
    image_interp = RegularGridInterpolator((y_coords, x_coords), image, method='linear', bounds_error=False, fill_value=0)
    mask_interp = RegularGridInterpolator((y_coords, x_coords), mask, method='linear', bounds_error=False, fill_value=0)
    
    # Interpolate
    points = np.stack([rotated_coords[..., 1], rotated_coords[..., 0]], axis=-1)
    rotated_image = image_interp(points)
    rotated_mask = mask_interp(points)
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Original slice
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(x, y, np.zeros_like(x), facecolors=plt.cm.gray(image))
    ax1.set_title('Original Slice')
    
    # Rotated coordinates
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(rotated_coords[..., 0], rotated_coords[..., 1], rotated_coords[..., 2], 
                    facecolors=plt.cm.gray(image))
    ax2.set_title('Rotated Coordinates')
    
    # Final rotated slice with mask overlay
    ax3 = fig.add_subplot(133)
    ax3.imshow(rotated_image, cmap='gray')
    ax3.imshow(rotated_mask > 0.5, cmap='Reds', alpha=0.3)
    ax3.set_title('Final Rotated Slice')
    
    plt.tight_layout()
    return fig

# Create custom colormap
nalphas = 256
color_array = plt.get_cmap('Reds')(range(nalphas))
color_array[:, -1] = np.linspace(0, 1, nalphas)
Reds_alpha_objects = LinearSegmentedColormap.from_list(name='Reds_alpha', colors=color_array)
mpl.colormaps.register(Reds_alpha_objects)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Medical Image 3D Rotation")
    gr.Markdown("Upload a medical image and its mask, then adjust the rotation angles to see the 3D visualization.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Medical Image", type="filepath")
            mask_input = gr.Image(label="Mask", type="filepath")
            
            with gr.Row():
                angle_x = gr.Slider(minimum=-180, maximum=180, value=0, label="X Rotation (degrees)")
                angle_y = gr.Slider(minimum=-180, maximum=180, value=0, label="Y Rotation (degrees)")
                angle_z = gr.Slider(minimum=-180, maximum=180, value=0, label="Z Rotation (degrees)")
            
            rotate_btn = gr.Button("Rotate")
        
        with gr.Column():
            output = gr.Plot()
    
    rotate_btn.click(
        fn=create_3d_visualization,
        inputs=[image_input, mask_input, angle_x, angle_y, angle_z],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch() 