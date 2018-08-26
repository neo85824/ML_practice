from PIL import Image
import sys

path = str(sys.argv[1])
im = Image.open(path)
print(im.size)
pix = im.load()
width = im.size[0]
height = im.size[1]

for i in range(width):
	for j in range(height):
		r, g, b = pix[i,j]
		pix[i, j] = (r/2, g/2, b/2)
		

im.save("Q2.jpg")
