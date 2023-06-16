import numpy as np
import cv2
import streamlit as st
from PIL import Image

def show(img, wait=0):
    if img.dtype != np.uint8:
        img = normalize(img)
    cv2.imshow("img", img)
    cv2.waitKey(wait)

def save(path, img):
    if img.dtype != np.uint8:
        img = normalize(img)
    cv2.imwrite(path, img)


def normalize(img):
    if img.dtype == bool:
        return np.uint8(255 * img)
    img = img - img.min()
    img = img / img.max()
    return np.uint8(255 * img)

def grayscale(img):
    if len(img.shape) == 3:
        b = img[:,:,0]
        g = img[:,:,1]
        r = img[:,:,2]
        return np.uint8(.2*r + .7*g + .1*b)
    return img

def drawSeam(img, seam):
    h, w = img.shape[:2]
    for y, x in enumerate(seam):
        img[y,x] = [0,0,255]
        if x > 0: img[y,x-1] = [0,0,255]
        if x < w-1: img[y,x+1] = [0,0,255]

def removeSeam(img, seam):
    out = img * 1
    # Remove a column from the image by copying everything to the right onto the current pixel
    for y,x in enumerate(seam):
        out[y, x:-1] = img[y,x+1:]
    
    # Return an image with one less column
    return out[:, :-1]

def addSeam(img, seam):
    out = img * 1
    # Add a column to the image
    out = cv2.resize(out, (out.shape[1]+1, out.shape[0]))
    
    # Shift the pixels to the right of the seam and duplicate the seam pixel
    for y,x in enumerate(seam):
        out[y, x+1:] = img[y,x:]
        out[y, x] = img[y,x]
    return out

def rotate(img):
    if img is None:
        return None
    return img.swapaxes(0,1)

Edges = np.float32([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])

def findSeam(img, modImg=None, modFactor=10.0):
    h, w = img.shape[:2]

    # Calculate the energy map using an edge finding kernel
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    iX = cv2.filter2D(gray*1.0, -1, Edges)
    iY = cv2.filter2D(gray*1.0, -1, Edges.T)
    I = (iX**2 + iY**2)**.5
    
    # If a modified image is specified, add the energy of the modified image to the energy map
    # In the modImg, dark pixels are less important (repulsors), and white pixels add priority (attractors)
    if modImg is not None:
        I -= modImg * modFactor

    energy_map = I * 1.0
    # show(energy_map)

    # Calculate the cumulative energy map (each pixel is the sum of the energy of itself and all pixels above it)
    for row in range(h-1):
        energy_map[row + 1] += cv2.erode(energy_map[row], np.uint8([1,1,1])).ravel()
    # show(energy_map)

    # Find the seam by starting at the bottom and moving up, choosing the pixel with the lowest energy and avoiding the edge of the image
    x = np.argmin(energy_map[-1])
    seam = [x]
    for row in reversed(range(h-1)):
        x = seam[0]
        if x > 0 and energy_map[row, x-1] < energy_map[row, x] and (x == w-1 or energy_map[row, x-1] < energy_map[row, x+1]):
            seam.insert(0, x - 1)
        elif x == w-1 or energy_map[row, x] < energy_map[row, x+1]:
            seam.insert(0, x)
        else:
            seam.insert(0, x + 1)

    return seam

def findAndRemoveSeam(img, modImg=None, modFactor=10.0):
    seam = findSeam(img, modImg, modFactor)
    img = removeSeam(img, seam)
    if modImg is not None:
        modImg = removeSeam(modImg, seam)

    return img, modImg

def findAndAddSeams(src, px, srcModImg=None, modFactor=10.0, showProgress=False):
    img = src * 1
    modImg = srcModImg * 1 if srcModImg is not None else None
    seams = []
    for i in range(px):
        seams.append(findSeam(img, modImg, modFactor))
        img = removeSeam(img, seams[-1])
        if showProgress: show(img, 10)
        if modImg is not None: modImg = removeSeam(modImg, seams[-1])

    for seam in reversed(seams):
        src = addSeam(src, seam)
        if srcModImg is not None: srcModImg = addSeam(srcModImg, seam)
        if showProgress: show(src, 10)

    return src, srcModImg

def retarget(src, px=0, py=0, target=None, modImg=None, modFactor=10.0, showProgress=False):
    img = src * 1

    # If no target is specified, use the current dimensions plus the specified padding
    if target is None:
        target = (img.shape[0] + int(py), img.shape[1] + int(px))
    else: target = [int(x) for x in target]

    # Loop while dimensions are over target
    while img.shape[0] > target[0] or img.shape[1] > target[1]:
        if img.shape[1] > target[1]:
            img, modImg = findAndRemoveSeam(img, modImg, modFactor=modFactor)

        # If we're removing a row, we need to rotate the image first, then rotate it back after removing the seam
        if img.shape[0] > target[0]:
            img = rotate(img)
            modImg = rotate(modImg)
            img, modImg = findAndRemoveSeam(img, modImg, modFactor=modFactor)
            img = rotate(img)
            modImg = rotate(modImg)
        if showProgress: show(img, 10)

    # Loop while dimensions are under target
    if img.shape[1] < target[1]:
        img, modImg = findAndAddSeams(img, target[1] - img.shape[1], srcModImg=modImg, modFactor=modFactor, showProgress=showProgress)

    # If we're adding a row, we need to rotate the image first, then rotate it back after adding the seam
    if img.shape[0] < target[0]:
        img = rotate(img)
        modImg = rotate(modImg)

        img, modImg = findAndAddSeams(img, target[0] - img.shape[1], srcModImg=modImg, modFactor=modFactor, showProgress=showProgress)

        img = rotate(img)
        modImg = rotate(modImg)
    
    return img
    
def removeContent(src, modImg, modFactor=10.0, showProgress=False):
    srcDimesions = src.shape
    img = src * 1

    # Loop while modImg still has white pixels
    while np.where(modImg == 255)[0].size > 0:
        img, modImg = findAndRemoveSeam(img, modImg, modFactor)
        if showProgress: show(np.hstack((img, cv2.cvtColor(modImg, cv2.COLOR_GRAY2BGR))), 10)
        
        img = rotate(img)
        modImg = rotate(modImg)
        img, modImg = findAndRemoveSeam(img, modImg, modFactor)
        img = rotate(img)
        modImg = rotate(modImg)
    
    img = retarget(img, target=srcDimesions[:2], showProgress=showProgress)
    return img

# INTERFACE CODE

st.title('Image Seam Carver')
importResizeFactor = 0.33
image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if image_file is not None:
    input_image = Image.open(image_file)
    input_image = np.array(input_image)
    image1 = cv2.resize(input_image, (0,0), fx=importResizeFactor, fy=importResizeFactor)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    height, width, _ = image1.shape

    input_resize_X = st.slider('Resize X', 0.1, 2.0, 1.0)
    input_resize_Y = st.slider('Resize Y', 0.1, 2.0, 1.0)

    out = retarget(image1, target=(input_resize_Y * height, input_resize_X * width))
    st.write("Output Image:")
    st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width="auto")