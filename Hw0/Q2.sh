from PIL import Image
im = Image.open("westbrook.jpg")
print(im.mode)
pix = im.load()
width = im.size[0]
height = im.size[1]

for i in range(width):
	for j in range(height):
		r, g, b = pix[i,j]
		pix[i, j] = (r/2, g/2, b/2)
		
		
im.save("Q2.jpg")
