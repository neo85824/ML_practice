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
		
		
		
resize_img = im.resize((300, 200), Image.BILINEAR)
resize_img.save("resize.jpg")


rotate_img = im.rotate(45, Image.BILINEAR)
rotate_img.save("rotate.jpg")

im.paste(resize_img, (200,300))
im.save("paste.jpg")

grey_img = im.convert("L")
grey_img.save("grey.jpg")



im.save("Q2.jpg")
