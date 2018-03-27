try:
	from urllib.request import urlretrieve
except ImportError:
	from urllib import urlretrieve

src = "http://img14.360buyimg.com/popWaterMark/jfs/t601/170/958056941/148174/98f25d6d/549bf34dN87f6e9a2.jpg"
src1 = 'http://cntk.ai/jup/MNIST-image.jpg'
src2 = "http://cntk.ai/jup/SimpleAEfig.jpg"
src3 = "http://cntk.ai/jup/DeepAEfig.jpg"
src4 = "http://www.cntk.ai/jup/sinewave.jpg"
src5 = "https://cntk.ai/jup/201/cifar-10.png"
scr6 = "https://cntk.ai/jup/201/Conv2D.png"
scr7 = "https://cntk.ai/jup/201/Conv2DFeatures.png"
scr8 = "https://cntk.ai/jup/201/MaxPooling.png"


print ('Downloading ' + scr8)
fname, h = urlretrieve(scr8, 'E:\\ProgramLib\\Python\\CNTK\\testdata\\MaxPooling.png')
print (fname)
print (h)
print ('Downloaded successfully')

