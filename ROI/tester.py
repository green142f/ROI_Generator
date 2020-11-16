from PIL import Image, ImageOps
import os
import os.path
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import random


zeroes = np.zeros((5,5))+255
# plt.imshow(Image.fromarray(zeroes))
# plt.show()
print(zeroes)

print("hello")


mod_path = Path(__file__).parent
file_path = (mod_path / "imageOne.pickle").resolve()
data = []
targets = []
orderedList = []
covidList = []

# input_dir = (mod_path / "groundTruth" / "testing.tif").resolve()
# input_dir = Image.open(str(input_dir))
# input_dir = ImageOps.fit(input_dir,(100,90),Image.ANTIALIAS)

input_dir = os.listdir((mod_path / "groundTruth").resolve())
size = (50,50)
len_images = len(input_dir)


with open(file_path, 'rb') as f:
    entry = pickle.load(f, encoding='latin1')
    data = (entry['image'])
    if 'status' in entry:
        targets = (entry['status'])



for image in input_dir[0:len_images]:
    if image == '.DS_Store':
	    continue
    mod_path = Path(__file__).parent
    input_dir = (mod_path / "groundTruth" / image).resolve() #Path of the covid image
    
    im = Image.open(str(input_dir))
    
    # im  = ImageOps.fit(im,(100,90),Image.ANTIALIAS) #This is done for testing purposes
    
    

    img = np.mean(im, axis=2) #Convert to grayscale
    plt.imshow(Image.fromarray(img),cmap = "gray",vmin=0, vmax=255)
    print("wasgood")
    # plt.show()
    imgs = np.mean(im.crop((0,0,800,800)),axis= 2)
    plt.imshow(imgs,cmap = "gray",vmin=0, vmax=255)
    plt.title("testing")
    # plt.show()
    print(img.shape)
    print("YOLO")


    

    xBias = 0
    yBias = 0
    
    while(xBias + size[0]<=img.shape[0] and yBias + size[1] <= img.shape[1]):
        crop_rectangle = (xBias,yBias,xBias + size[0],yBias + size[1])
        orderedList.append(np.mean(im.crop(crop_rectangle),axis = 2))

        tempImage = np.mean(im.crop(crop_rectangle),axis = 2)

        if(tempImage[int(size[1]/2)][int(size[0]/2)]>240):
            # if(count == 5):
            #     print(tempImage[int(size[1]/2)][int(size[0]/2)])
            covidList.append(True)
        else:
            covidList.append(False)
        # plt.imshow(im.crop(crop_rectangle))
        # plt.show()
        # plt.imshow(orderedList[len(orderedList)-1],cmap = "gray")
        # plt.show()
        # if(np.mean(im.crop(crop_rectangle))[int(size[0]/2)][int(size[1]/2)]>240):
        #     covidList.append(True)
        # else:
        #     covidList.append(False) 
        if(xBias + size[0] == img.shape[0]):
            yBias +=1
            xBias = 0
        else:
            xBias+=1
for i in range(0,30):
    randInt = random.randint(0,(len(orderedList)-1))
    tempImage = orderedList[randInt]
    print(tempImage)
    plt.imshow(tempImage, cmap = 'gray',vmin=0, vmax=255)
    plt.title(covidList[randInt])
    plt.show()


for j in range(0,5):
    for i in range(1,30,3):
        randInt = random.randint(0,(len(orderedList)-1))
        tempImage = orderedList[randInt]
        sp = plt.subplot(10, 3, i)

        plt.imshow(Image.fromarray(zeroes),vmin=0, vmax=255)
        sp.set_title(str(targets[randInt]) + "  " + str(covidList[randInt]))
        sp = plt.subplot(10, 3, i+1)

        plt.imshow(tempImage, cmap = 'gray',vmin=0, vmax=255)
        sp = plt.subplot(10,3, i +2)
        
        plt.imshow(Image.fromarray(np.uint8(data[randInt]) , 'L'), cmap = "gray",vmin=0, vmax=255)
    plt.show()
# for j in range(0,10):
#     for i in range(1,30,3):
#         print(str(i) + " I")
#         randInt = random.randint(0,(len(data)))
#         xBias = randInt % int(len(np.array(input_dir)[0])-50)
#         yBias = int(randInt/len(np.array(input_dir)[0])-50)
#         crop_rectangle = (xBias,yBias,xBias + 50,yBias + 50)
#         tempImage = np.mean(input_dir.crop(crop_rectangle),axis = 2)
#         print(len(data))
#         print(randInt)
#         sp = plt.subplot(10, 3, i)
#         sp.axis("Off")
#         plt.imshow(Image.fromarray(zeroes))
#         sp.set_title(str(targets[randInt]) + " " + str(xBias) + " " + str(yBias))
#         sp = plt.subplot(10, 3, i+1)
#         sp.axis("Off")
#         plt.imshow(tempImage, cmap = 'gray')
#         print(xBias)
#         print(yBias)

#         sp = plt.subplot(10,3, i +2)
#         sp.axis("Off")
#         plt.imshow(Image.fromarray(np.uint8(data[randInt]) , 'L'), cmap = "gray")

#         plt.tight_layout()
#     plt.subplots_adjust(hspace=.01)
#     plt.show()