# import the necessary libraries
import numpy as np
import torch
import torchvision
import cv2 as cv
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# a function for demonstrating structure of model's modules in a table
# use it if you want to know the name of convolution layers

def demonstrate_modules(model):
    table = PrettyTable(["Name", "Modules"])
    for name, mod in model.named_modules():
        table.add_row([name, mod])
    print(table)
    return

tr = torchvision.transforms.ToPILImage()
tr1 = torchvision.transforms.Resize([224, 224])
tr2 = torchvision.transforms.ToTensor()
tr3 = torchvision.transforms.Resize([60, 60])
d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining model/ if you want to know names of convolution layers please active the second line
model = torchvision.models.resnet50(pretrained=True)
demonstrate_modules(model)
model.eval()
model = model.to(d)

# defining image+ preprocessing
img = cv.imread("C:/Users/rozhi/Desktop/cat.jpeg")

img = tr2(img)
img = np.expand_dims(img, axis=0)
img = torch.from_numpy(img)
img = tr1(img)
img = img.to(d)

# choose number of layer/ bottleneck and convolution of output
number_layer = 4
number_bottleneck = 1
number_channel = 1
name_conv = 'layer' + str(number_layer) + '.' + str(number_bottleneck) + '.' + 'conv' + str(number_channel)

# list of forbidden modules to implement on image
list1 = ['','layer1','layer1.0','layer1.1','layer1.2','layer2','layer2.0','layer2.1','layer2.2','layer2.3','layer3', 'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4', 'layer4.0', 'layer4.1', 'layer4.2', 'layer1.0.downsample', 'layer1.0.downsample.0', 'layer1.0.downsample.1', 'layer2.0.downsample', 'layer2.0.downsample.0', 'layer2.0.downsample.1','layer3.0.downsample', 'layer3.0.downsample.0', 'layer3.0.downsample1', 'layer4.0.downsample', 'layer4.0.downsample.0', 'layer4.0.downsample.1']

# implementing the chosen convolution layer on image to show the changes that this layer makes
# if you want implement the first convolution just active the first line and deactivate for instruction

# img2 = model.conv1(img)
for name, layer in model.named_modules():
    a = name in list1
    if not a:
        if name_conv in name:
            img2 = layer(img)
            break
        else:
            img = layer(img)


a, b, c, d = img2.shape
print(img2.shape)

# show the output of the chosen convolution layer for each channel
# please choose a1 and b1 smaller than a and b
a1 = 1
b1 = 10
fig = plt.figure(figsize=(20, 20))
i = 0
f = 1
while i < b1:
    j = 0
    while j < a1:

        img3 = img2[j, i, :, :]
        img3 = tr(img3)
        img3 = tr3(img3)
        fig.add_subplot(a1, b1, f)
        plt.imshow(img3, cmap='gray')
        plt.axis('off')
        plt.title(f)

        f += 1
        j += 1
    i += 1
plt.show()

