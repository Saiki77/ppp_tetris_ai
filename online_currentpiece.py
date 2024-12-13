import cv2
import numpy as np
import pyautogui

import os
from datetime import datetime

# Predefined region for current piece detection (adjust as needed)
CURRENT_PIECE_REGION = {
    "x": 1785,  # Left X coordinate of the piece region
    "y": 510,   # Top Y coordinate of the piece region
    "width": 250,  # Width of the region
    "height": 170, # Height of the region
}

# Directory to save ROI images
TEMPLATES_DIR = 'tetris_templates'

# Subdirectories for each Tetris piece and an 'Unknown' category
PIECES = ['I', 'O', 'T', 'S', 'Z', 'J', 'L', 'Unknown']

def setup_directories():
    """
    Create directories for each Tetris piece if they don't exist.
    """
    if not os.path.exists(TEMPLATES_DIR):
        os.makedirs(TEMPLATES_DIR)
        print(f"Created main templates directory: {TEMPLATES_DIR}")
    
    for piece in PIECES:
        piece_dir = os.path.join(TEMPLATES_DIR, piece)
        if not os.path.exists(piece_dir):
            os.makedirs(piece_dir)
            print(f"Created directory for piece '{piece}': {piece_dir}")

def take_screenshot():
    """
    Take a screenshot of the entire screen.
    """
    try:
        screenshot = pyautogui.screenshot()
        screenshot_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        return screenshot_bgr
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None

def highlight_region(image, region):
    """
    Highlight a region on the image.
    """
    x, y, w, h = region["x"], region["y"], region["width"], region["height"]
    highlighted_image = image.copy()
    cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
    return highlighted_image

def preprocess_image(image):
    """
    Convert image to grayscale and apply preprocessing.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def load_templates():
    """
    Load all templates and compute their descriptors.
    Returns a dictionary with piece names as keys and lists of descriptors as values.
    """
    orb = cv2.ORB_create()
    templates = {}
    for piece in PIECES[:-1]:  # Exclude 'Unknown'
        piece_dir = os.path.join(TEMPLATES_DIR, piece)
        templates[piece] = []
        for template_file in os.listdir(piece_dir):
            template_path = os.path.join(piece_dir, template_file)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"Warning: Failed to read template image {template_path}. Skipping.")
                continue
            kp, des = orb.detectAndCompute(template, None)
            if des is not None:
                templates[piece].append((kp, des))
            else:
                print(f"Warning: No descriptors found in template {template_path}. Skipping.")
    return templates

def save_roi_image(roi, piece_name):
    """
    Save the ROI image to the corresponding piece directory.
    If piece_name is None or 'Unknown', save it to the 'Unknown' directory.
    """
    if piece_name is None or piece_name not in PIECES[:-1]:
        piece_name = 'Unknown'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"{piece_name}_{timestamp}.png"
    filepath = os.path.join(TEMPLATES_DIR, piece_name, filename)
    
    # Verify that ROI is not empty
    if roi is None or roi.size == 0:
        print(f"Warning: ROI is empty. Cannot save image for piece '{piece_name}'.")
        return
    
    # Attempt to save the image
    success = cv2.imwrite(filepath, roi)
    if success:
        print(f"Successfully saved ROI image to {filepath}")
    else:
        print(f"Error: Failed to save ROI image to {filepath}")

def compare_with_templates(current_roi, templates, threshold=10):
    """
    Compare the current ROI with precomputed templates using ORB feature matching.
    Returns the best matching piece name and the number of good matches.
    """
    best_match = None
    max_good_matches = 0
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Find keypoints and descriptors in the current ROI
    kp1, des1 = orb.detectAndCompute(current_roi, None)
    
    if des1 is None:
        print("Warning: No descriptors found in current ROI.")
        return best_match, max_good_matches
    
    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    for piece, descriptors_list in templates.items():
        for kp2, des2 in descriptors_list:
            # Match descriptors
            matches = bf.match(des1, des2)
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            # Define a good match as distance less than a threshold
            good_matches = [m for m in matches if m.distance < 50]
            num_good_matches = len(good_matches)
            #print(f"Comparing with template of piece '{piece}': {num_good_matches} good matches.")
            # Update best match if current has more good matches
            if num_good_matches > max_good_matches and num_good_matches > threshold:
                max_good_matches = num_good_matches
                best_match = piece
    
    return best_match, max_good_matches

def main():
    print("Capturing screenshot and analyzing the current Tetris piece...")
    
    # Load precomputed templates
    print("Loading templates...")
    templates = load_templates()
    print("Templates loaded.")
    
    # Take a screenshot of the entire screen
    screenshot = take_screenshot()
    if screenshot is None:
        print("Error: Screenshot capture failed.")
        return
    
    # Highlight the current piece region (optional)
    highlighted_image = highlight_region(screenshot, CURRENT_PIECE_REGION)
    
    # Extract the region of interest (ROI)
    x, y, w, h = CURRENT_PIECE_REGION["x"], CURRENT_PIECE_REGION["y"], CURRENT_PIECE_REGION["width"], CURRENT_PIECE_REGION["height"]
    roi = screenshot[y:y+h, x:x+w]
    
    # Verify ROI extraction
    if roi.size == 0:
        print("Error: ROI extraction resulted in an empty image.")
        # Save the empty ROI as 'Unknown' for analysis
        save_roi_image(roi, 'Unknown')
        return
    else:
        print(f"ROI extracted: {roi.shape[1]}x{roi.shape[0]} pixels")
    
 
    
    # Preprocess the ROI
    preprocessed_roi = preprocess_image(roi)
    
 
    
    # Compare the ROI with templates
    print("Comparing ROI with templates...")
    piece, similarity = compare_with_templates(preprocessed_roi, templates, threshold=10)
    
    if piece:
        #print(f"Identified Tetris Piece: {piece} with {similarity} good matches.")
        print(piece)
        return piece
    else:
        print("Could not identify the Tetris piece based on templates.")
        piece = 'Unknown'
    return piece
    

if __name__ == "__main__":
    setup_directories()  # Ensure directories are set up before running
    #time.sleep(3)  # Delay to allow user to prepare the screen
    main()

