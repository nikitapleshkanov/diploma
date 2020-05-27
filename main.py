import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
from scipy import ndimage
from skimage.measure import compare_ssim
import argparse
import imutils
import sys



# video1 = cv2.VideoCapture("bri.mp4")

def raskadrovka(video1):
    not_first = 0
    global img_copy1
    for number in range(0, frame_seq, int(fps)):
        video1.set(1,number)
        ret, frame = video1.read()
        # cv2.imwrite('images99'+str(number)+'.jpg', frame)
        img = frame
        gray = denoise(img)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('imagesQ99'+str(number)+'.jpg', gray)
        gray = otsu(gray)
        gray = np.float32(connected_components(gray))
        if not_first == 0:
            img_copy = gray
            ocr(gray)
        if not_first != 0:
            gray_copy = gray
            img_copy1 = img_copy
            if len(gray) > len(img_copy):
                gray_copy = gray[len(gray)-len(img_copy):]
            if len(gray) < len(img_copy):
                img_copy1 = img_copy[len(img_copy)-len(gray):]
            
            (score, diff) = compare_ssim(gray_copy, img_copy1, full=True)
            diff = (diff * 255).astype("uint8")
            
            # print("SSIM : {}".format(score), )
            

            if score < 0.80:
                ocr(gray)
                img_copy = gray


            # cv2.imshow("Original", gray_copy)
            # cv2.imshow("Modified", img_copy1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        not_first = 1
        

    return gray


def otsu(img):

    kisd,flows = cv2.threshold(img,127,127,cv2.THRESH_BINARY_INV)

    pixels = flows.tolist()

    img = img.tolist()
    for i in range(0, len(pixels)):
        for j in range(0, len(pixels[i])):
            if pixels[i][j] == 0:
                pixels[i][j] = 255
                if i-i >= 0:
                    pixels[i-1][j] = 255
                if i+1<=len(pixels)-1:
                    pixels[i+1][j] = 255
                if j-1>=0:
                    pixels[i][j-1] = 255
                if j+1<=len(pixels[i])-1:
                    pixels[i][j+1] = 255
                if i-1 >= 0 and j-1>=0:
                    pixels[i-1][j-1] = 255
                if i-1 >= 0 and j+1<=len(pixels[i])-1:
                    pixels[i-1][j+1] = 255
                if i+1<=len(pixels)-1 and j-1>=0:
                    pixels[i][j-1] = 255
                if i+1<=len(pixels)-1 and j+1<=len(pixels[i])-1:
                    pixels[i][j+1] = 255

    for i in range(0, len(pixels)):
        for j in range(0, len(pixels[i])):
            if pixels[i][j] != 255:
                img[i][j] = 0           
    return(img)

def connected_components(t):
    global count2
    global maximum
    count2 += 1 
    pixels = t
   
    img = np.asarray(t)

    img = ndimage.gaussian_filter(img, blur_radius)
    threshold = 100
    struct = ndimage.generate_binary_structure(2, 2)
    connected, obj = ndimage.label(img > threshold) 
    
    spisok = connected.tolist()
    x = []
    finalList = []
    count1 = 0
    count = 0
    obj = {}
    listik = []
    max1 = 0
    min1 = 99999
    obj2 = {}
    average = {}
    x = [x for x in spisok if sum(x)>0]
    for i in range(0,(len(x))):
        for j in range(0, (len(x[i]))):    
            if x[i][j] !=1:
                if x[i][j] not in obj.keys():
                    count += 1
                    min1 = i
                    min2 = j
                    max2 = j 
                    for k in range(i, len(x)):
                        if x[i][j] in x[k]:
                            max1 = k    
                            if max2 <list(reversed(x[k])).index(x[i][j]):
                                max2 = list(reversed(x[k])).index(x[i][j])
                            if min2 > x[k].index(x[i][j]):
                                min2 = x[k].index(x[i][j])
                        
                    obj[x[i][j]] = [min1, max1, max1-min1, (max1+min1)/2, max2-min2]
            max1 = 0
            min1 = 99999
    
        if count > 3:
            for j in obj.keys():
                if j in x[i]:
                    if j not in obj2.keys():
                        obj2[j] = obj[j]
        count = 0
        
    
    for p in obj2.keys():
        for d in obj2.keys():
            if abs(obj2[p][3]-obj2[d][3])< 5 and  obj2[p][2]>15:
                count1 += 1
        if count1>7:
            finalList.append(p)
        count1=0
    a = len(pixels)
    if  len(x)!=a:
        del pixels[0:a-len(x)-1:1]

    for i in range(0,(len(x))):
        for j in range(0, (len(x[i]))):
            if x[i][j] in finalList and x[i][j]!=0:
                pixels[i][j]=30
                if maximum == -1:
                    maximum = i
            else:
                pixels[i][j]=0
    pixels = pixels[maximum-10:]
    maximum = -1
    return pixels

def denoise(img):
    result = cv2.fastNlMeansDenoising(img)
    return result

def ocr(img):
    print(pytesseract.image_to_string(img, lang='eng'))


# raskadrovka(video1)

if __name__ == "__main__":
    path = sys.argv[1]
    video1 = cv2.VideoCapture(path)
    
    fps = video1.get(cv2.CAP_PROP_FPS)
    frame_seq = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
    time_length = frame_seq/fps
    print('fps: '+ str(fps))
    pytesseract.pytesseract.tesseract_cmd = r'C:\tesseract\tesseract.exe'
    blur_radius = 0.1
    threshold = 100

    count2 = 0
    wat = []
    img_copy1 = None
    maximum = -1
    raskadrovka(video1)