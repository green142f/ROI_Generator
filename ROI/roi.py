from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import time
import os

mod_path = Path(__file__).parent

input_dir = os.listdir((mod_path / "Images").resolve())
len_images = len(input_dir)
size = (50,50) #Size of the ROI

orderedList = [] #Array that holds the ROI grayscale values
covidList = [] #Whether the ROI is covid positive or negative











start_time = time.time()
for image in input_dir[0:len_images]:
    if image == '.DS_Store':
	    continue
    mod_path = Path(__file__).parent
    input_dir = (mod_path / "Images" / image).resolve() #Path of the covid image
    
    im = Image.open(str(input_dir))
    
    #im  = ImageOps.fit(im,(60,60),Image.ANTIALIAS) #This is done for testing purposes
    # groundTruth  = ImageOps.fit(groundTruth,(60,60),Image.ANTIALIAS)


    # plt.show()
    

    img = np.mean(im, axis=2) #Convert to grayscale



    

    xBias = 0
    yBias = 0
    while(xBias + size[0]<=img.shape[0] and yBias + size[1] <= img.shape[1]):
        crop_rectangle = (xBias,yBias,xBias + size[0],yBias + size[1])
        orderedList.append(np.mean(im.crop(crop_rectangle),axis = 2))
        # if(np.mean(im.crop(crop_rectangle))[int(size[0]/2)][int(size[1]/2)]>240):
        #     covidList.append(True)
        # else:
        #     covidList.append(False) 
        if(xBias + size[0] == img.shape[0]):
            yBias +=1
            xBias = 0
        else:
            xBias+=1

input_dir = os.listdir((mod_path / "groundTruth").resolve())
len_images = len(input_dir)            
for image in input_dir[0:len_images]:
    if image == '.DS_Store':
	    continue
    
    groundTruthDir = (mod_path / "groundTruth" / image).resolve() #Path of the ground truth
    groundTruth = Image.open(str(groundTruthDir))
    imgGround = np.mean(groundTruth, axis = 2)
    xBias = 0
    yBias = 0
    while(xBias + size[0]<=imgGround.shape[0] and yBias + size[1] <= imgGround.shape[1]):
        crop_rectangle = (xBias,yBias,xBias + size[0],yBias + size[1])
        tempImage = np.mean(groundTruth.crop(crop_rectangle),axis = 2)
        if(tempImage[int(size[0]/2)][int(size[1]/2)]>240):
            covidList.append(True)
        else:
            covidList.append(False) 
        if(xBias + size[0] == imgGround.shape[0]):
            yBias +=1
            xBias = 0
        else:
            xBias+=1


print(len(orderedList))
print(len(covidList))
imgs = Image.fromarray(np.uint8(orderedList[0]) , 'L')
plt.imshow(imgs, cmap='gray')
#plt.show()

dictionary = dict()
dictionary["image"] = orderedList
dictionary["status"] = covidList

print(sum(dictionary["status"]))



with open((mod_path / 'imageOne.pickle').resolve(),'wb') as handle:
		pickle.dump(dictionary, handle, protocol = pickle.HIGHEST_PROTOCOL)
    



print("Process finished --- %s seconds ---" % (time.time() - start_time))