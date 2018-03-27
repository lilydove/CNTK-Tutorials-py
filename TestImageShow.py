# Test of showing images

from IPython.display import Image
from PIL import Image

import matplotlib.pyplot as plt # plt show images
import matplotlib.image as mpimg # mpimg read images
import numpy as np

#from skimage import io

#image = io.imread("http://cntk.ai/jup/MNIST-image.jpg")
#io.imshow(image)
#io.show()

# Figure 1
# Image(url="http://cntk.ai/jup/MNIST-image.jpg", width=300, height=300)

# lena = mpimg.imread("http://cntk.ai/jup/MNIST-image.jpg") # 
# lena.shape #(512, 512, 3)
 
# plt.imshow(lena) # 
# plt.axis('off') # 
# plt.show()

#import requests as req
#from io import BytesIO
#response = req.get("http://cntk.ai/jup/MNIST-image.jpg")
#image = Image.open(BytesIO(response.content))
#image.show()

 
#img = plt.imread(r'http://cntk.ai/jup/MNIST-image.jpg')
#plt.imshow(img)
#plt.show()

#------ Download images of jpg format ------
try:
	from urllib.request import urlretrieve
except ImportError:
	from urllib import urlretrieve

src = r'http://cntk.ai/jup/MNIST-image.jpg'
print ('Downloading ' + src)
fname, h = urlretrieve(src, 'E:\\ProgramLib\\Python\\CNTK\\testdata\\testdata20180310.jpg')
print ("File name : " , fname)
print ("File head info :", h)

img = Image.open(fname)

plt.figure("Image") # 
plt.imshow(img)
plt.axis('on') # 
plt.title('image') # 
plt.show()
