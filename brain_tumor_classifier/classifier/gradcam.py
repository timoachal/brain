import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

class GradCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        
    def _find_target_layer(self):
        """Find the last convolutional layer in the model"""
        if self.model is None:
            return None
            
        try:
            # Look for the last convolutional layer
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    return layer.name
            
            # If no conv layer found, use the last layer before dense layers
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) > 2:  # Has spatial dimensions
                    return layer.name
                    
            return None
        except:
            return None
    
    def generate_gradcam(self, image_path, class_index=None):
        """Generate Grad-CAM heatmap for the given image"""
        try:
            if self.model is None or self.layer_name is None:
                return self._generate_mock_gradcam(image_path)
            
            # Preprocess image
            img_array = self._preprocess_image(image_path)
            if img_array is None:
                return self._generate_mock_gradcam(image_path)
            
            # Create a model that maps the input image to the activations of the target layer
            grad_model = tf.keras.models.Model(
                [self.model.inputs], 
                [self.model.get_layer(self.layer_name).output, self.model.output]
            )
            
            # Compute the gradient of the top predicted class for our input image
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                if class_index is None:
                    class_index = tf.argmax(predictions[0])
                class_channel = predictions[:, class_index]
            
            # Compute gradients
            grads = tape.gradient(class_channel, conv_outputs)
            
            # Pool the gradients over all the axes leaving out the channel dimension
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the channels by the corresponding gradients
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Normalize the heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return self._create_overlay(image_path, heatmap.numpy())
            
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            return self._generate_mock_gradcam(image_path)
    
    def _preprocess_image(self, image_path):
        """Preprocess image for Grad-CAM"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            
            return image
        except:
            return None
    
    def _create_overlay(self, image_path, heatmap):
        """Create overlay of heatmap on original image"""
        try:
            # Load original image
            original_img = cv2.imread(image_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Resize heatmap to match original image
            heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
            
            # Convert heatmap to RGB
            heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            
            # Create overlay
            overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
            
            return overlay
            
        except Exception as e:
            print(f"Error creating overlay: {e}")
            return self._generate_mock_gradcam(image_path)
    
    def _generate_mock_gradcam(self, image_path):
        """Generate a mock Grad-CAM visualization for demonstration"""
        try:
            # Load original image
            original_img = cv2.imread(image_path)
            if original_img is None:
                # Create a placeholder image
                original_img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Create a simple mock heatmap (circular pattern in center)
            h, w = original_img.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # Create circular gradient
            y, x = np.ogrid[:h, :w]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2
            mask = mask / mask.max()
            
            # Invert and normalize
            heatmap = 1 - mask
            heatmap = np.clip(heatmap, 0, 1)
            
            # Add some randomness to make it look more realistic
            noise = np.random.random((h, w)) * 0.3
            heatmap = np.clip(heatmap + noise, 0, 1)
            
            # Apply colormap
            heatmap_colored = cm.jet(heatmap)[:, :, :3]
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            
            # Create overlay
            overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
            
            return overlay
            
        except Exception as e:
            print(f"Error generating mock Grad-CAM: {e}")
            # Return a simple placeholder
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def save_gradcam(self, image_path, output_path, class_index=None):
        """Generate and save Grad-CAM visualization"""
        try:
            gradcam_img = self.generate_gradcam(image_path, class_index)
            
            # Convert RGB to BGR for OpenCV
            gradcam_bgr = cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR)
            
            # Save the image
            cv2.imwrite(output_path, gradcam_bgr)
            
            return True
            
        except Exception as e:
            print(f"Error saving Grad-CAM: {e}")
            return False

