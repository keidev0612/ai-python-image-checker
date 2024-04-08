import cv2
import numpy as np
import os

def extract_cards(input_image_path, output_folder):
    # Read the input image
    image = cv2.imread(input_image_path)
    if image is None:
        print("Error: Unable to read input image.")
        return
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to obtain a binary image
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours with hierarchy
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Extract and save each card
    card_count = 0
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)

        # Filter contours based on a more adaptive area threshold
        if area > image.shape[0] * image.shape[1] * 0.0075:  # Threshold of 1% of the image size
            rect = cv2.minAreaRect(contour)
            
            # Get corner points and sort them
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Width and Height of the detected rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])

            src_pts = box.astype("float32")

            # Coordinate of the points in box points after the rectangle has been straightened
            dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")
            
            # The perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # Directly warp the rotated rectangle to get the straightened rectangle
            warped = cv2.warpPerspective(image, M, (width, height))

            # If the card is in landscape mode, rotate it to be in portrait mode
            if width > height:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

            # Save the card as a separate image
            card_output_path = os.path.join(output_folder, f"card_{card_count}.jpg")
            cv2.imwrite(card_output_path, warped)
            print(f"Card {card_count} saved as {card_output_path}")
            
            card_count += 1

# Example usage
input_image_path = "2.jpg"
output_folder = "extracted_cards"
extract_cards(input_image_path, output_folder)
