from PIL import Image
img = Image.open("MNIST/image/5.png")
img = img.resize((14,14))
img.save("MNIST/image/14x5.png")