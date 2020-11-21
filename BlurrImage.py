import numpy as np
import time 
from PIL import Image

# Function to reduce and blurr an image by averaging filter of specified stride (square stride)
def blurr(img,stride=2):

    # Transpose so we can get at the channels easily
    img_np = np.array(img).T

    # Set up matrices used for matrix multiplication, which implements the mean filter pass
    mult1 = np.zeros((img_np.shape[1]-stride+1,img_np.shape[1]))
    mult2 = np.zeros((img_np.shape[2]-stride+1,img_np.shape[2]))
    
    # Update matrix 1 to be correct
    s = 0
    for row in range(mult1.shape[0]):
        mult1[row,s:s+stride] = 1
        s += 1

    # Update matrix 2 to be correct
    s = 0
    for row in range(mult2.shape[0]):
        mult2[row,s:s+stride] = 1
        s += 1

    # Go through all channels, run filter (matrix mult), transpose and divide by elements of filter
    img_np = np.array([np.dot(c2,mult2.T) for c2 in
                      [np.dot(mult1,c1) for c1 in img_np]]).T*1/stride**2
    return Image.fromarray(img_np.astype(np.uint8))
