import cv2
import numpy as np
import matplotlib.pyplot as plt


## !! This Script was Created by Zidelmal Mohamed Sherif !! ##
## !! if You use it you must give Credit @ to the owner of the script !!##
# Made in Algeria , Skikda !

def load_image(image_path):
    """Load the image from the specified path and convert it to RGB format."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_image(image, target_size=(1000, 1000)):
    """Resize and enhance the image for better vegetation detection."""
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return image_resized

def get_vegetation_percentage(image, lower_threshold=(30, 50, 50), upper_threshold=(90, 255, 255)):
    """Calculate the percentage of vegetation using HSV color space."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Create a mask for vegetation based on HSV thresholds
    vegetation_mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)
    
    # Exclude black pixels (background)
    black_mask = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) == 0).astype(np.uint8) * 255
    vegetation_mask = cv2.bitwise_and(vegetation_mask, vegetation_mask, mask=~black_mask)
    
    # Calculate the percentage of vegetation pixels
    total_non_black_area = np.sum(black_mask == 0)  # Total area excluding black pixels
    vegetation_area = np.sum(vegetation_mask > 0)  # Vegetation area within the non-black region
    
    vegetation_percentage = (vegetation_area / total_non_black_area) * 100 if total_non_black_area > 0 else 0
    
    return vegetation_percentage, vegetation_mask

def add_text_to_image(image, text, position, color, font_scale=1, thickness=2):
    """Add text overlay to the image."""
    image_with_text = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_with_text, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image_with_text

def save_vegetation_mask(vegetation_mask, output_path):
    """Save the vegetation mask as an image."""
    cv2.imwrite(output_path, vegetation_mask)
    print(f"Vegetation mask saved to {output_path}")

def display_images(original_image, image_with_text, vegetation_mask):
    """Display the original image, image with text, and vegetation mask."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image_with_text)
    plt.title('Image with Vegetation Percentage')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(vegetation_mask, cmap='gray')
    plt.title('Vegetation Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main(image_path):
    """Main function to process the image and display results."""
    # Load and preprocess the image
    image = load_image(image_path)
    image_preprocessed = preprocess_image(image)
    
    # Calculate vegetation area percentage and get vegetation mask
    vegetation_percentage, vegetation_mask = get_vegetation_percentage(image_preprocessed)
    
    # Add vegetation percentage text to the image
    text = f"Vegetation: {vegetation_percentage:.2f}%"
    position = (10, image_preprocessed.shape[0] - 10)  # Bottom-left corner
    color = (0, 255, 0)  # Green color in RGB
    image_with_text = add_text_to_image(image_preprocessed, text, position, color)
    
    # Save the vegetation mask
    save_vegetation_mask(vegetation_mask, 'vegetation_mask.png')
    
    # Display results
    display_images(image_preprocessed, image_with_text, vegetation_mask)

if __name__ == "__main__":
    # Replace the path with the path to your vegetation map PNG
    main(r'C:\Users\USER\Downloads\s21.png')

