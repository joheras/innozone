#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import date
import requests
import pandas as pd
import cv2
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import numpy as np
import schedule
import time

# In[2]:


#gauth = GoogleAuth()       
#gauth.LocalWebserverAuth()       
#drive = GoogleDrive(gauth) 


# In[3]:


def align_images(image, template, maxFeatures=500, keepPercent=0.2,debug=False):
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)
    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x:x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
            matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)
    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt
    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    # return the aligned image
    return aligned


# In[4]:


listImagesURLs = ['https://web.larioja.org/files/siar/mapasOidio/RiesgoAscosporas-V1.png',
              'https://web.larioja.org/files/siar/mapasOidio/NumerovecesAscosporas-V1.png',
              'https://web.larioja.org/files/siar/mapasOidio/RiesgoConidiasActivas-V1.png',
              'https://web.larioja.org/files/siar/mapasOidio/RiesgoConidias7d-SinLeyenda-V1.png',
              'https://web.larioja.org/files/siar/mapasOidio/RiesgoConidias10d-SinLeyenda-V1.png',
              'https://web.larioja.org/files/siar/mapasOidio/RiesgoConidias14d-SinLeyenda-V1.png']

def download_image(url,date):
    imageName = url.split('/')[-1]
    local_file = 'imagenes/'+date+'-'+imageName
    data = requests.get(url)
    with open(local_file, 'wb') as file:
        file.write(data.content)

def download_images():
    today = date.today()
    datestr = today.strftime("%d-%m-%y")
    for url in listImagesURLs:
        download_image(url,datestr)


# In[5]:


def riesgoAscosporas():
    today = date.today()
    fecha = today.strftime("%d-%m-%y")
    im = cv2.imread('imagenes/'+fecha+'-'+'RiesgoAscosporas-V1.png')
    logronoPixel = (93,184)
    pixelValue = im[93,184]
    
    if ((pixelValue[0]>240) and (pixelValue[0]<255)):
        return 0
    if ((pixelValue[0]>167) and (pixelValue[0]<179)):
        return 1
    if ((pixelValue[0]>108) and (pixelValue[0]<120)):
        return 2
    if ((pixelValue[0]>65) and (pixelValue[0]<77)):
        return 3
    return 4


# In[6]:


def numeroInfecciones():
    today = date.today()
    fecha = today.strftime("%d-%m-%y")
    im = cv2.imread('imagenes/'+fecha+'-'+'NumerovecesAscosporas-V1.png')
    pixelValue = im[93,184]
    if ((pixelValue[0]>240) and (pixelValue[0]<255)):
        return '0'
    if ((pixelValue[0]>167) and (pixelValue[0]<179)):
        return '1-2'
    if ((pixelValue[0]>108) and (pixelValue[0]<120)):
        return '3-5'
    if ((pixelValue[0]>65) and (pixelValue[0]<77)):
        return '7-8'
    if ((pixelValue[0]>31) and (pixelValue[0]<43)):
        return '9-11'
    return '>12'


# In[7]:


def conideasActivas():
    today = date.today()
    fecha = today.strftime("%d-%m-%y")
    im = cv2.imread('imagenes/'+fecha+'-'+'RiesgoConidiasActivas-V1.png')
    pixelValue = im[93,184]
    if ((pixelValue[0]>31) and (pixelValue[0]<43)):
        return 1
    return 0


# In[8]:


def riesgo7d():
    today = date.today()
    fecha = today.strftime("%d-%m-%y")
    im = cv2.imread('imagenes/'+fecha+'-'+'RiesgoConidias7d-SinLeyenda-V1.png')
    pixelValue = im[144,183]
    if ((pixelValue[0]>240) and (pixelValue[0]<255)):
        return 0
    if ((pixelValue[0]>31) and (pixelValue[0]<43)):
        return 3
    if ((pixelValue[0]>65) and (pixelValue[0]<77)):
        return 2
    return 1


# In[9]:


def riesgo10d():
    today = date.today()
    fecha = today.strftime("%d-%m-%y")
    im = cv2.imread('imagenes/'+fecha+'-'+'RiesgoConidias10d-SinLeyenda-V1.png')
    pixelValue = im[144,183]
    if ((pixelValue[0]>240) and (pixelValue[0]<255)):
        return 0
    if ((pixelValue[0]>31) and (pixelValue[0]<43)):
        return 3
    if ((pixelValue[0]>65) and (pixelValue[0]<77)):
        return 2
    return 1


# In[10]:


def riesgo14d():
    today = date.today()
    fecha = today.strftime("%d-%m-%y")
    im = cv2.imread('imagenes/'+fecha+'-'+'RiesgoConidias14d-SinLeyenda-V1.png')
    pixelValue = im[144,183]
    if ((pixelValue[0]>240) and (pixelValue[0]<255)):
        return 0
    if ((pixelValue[0]>31) and (pixelValue[0]<43)):
        return 3
    if ((pixelValue[0]>65) and (pixelValue[0]<77)):
        return 2
    return 1


# In[11]:


csvPath = 'riesgoOidio.csv'

def read_and_update_csv():
    today = date.today()
    fecha = today.strftime("%d/%m/%y")
    df = pd.read_csv(csvPath)
    mydict = {'Fecha':fecha,
        'Riesgo':riesgoAscosporas(), 
        'nº Infecciones Acumuladas Ascosporas':numeroInfecciones(), 
        'Conidias Activas':conideasActivas(), 
        'Riesgo Acumulado conidias 7 días':riesgo7d(), 
        'Riesgo Acumulado conidias 10 días':riesgo10d(), 
        'Riesgo Acumulado conidias 14 días':riesgo14d()
    }
    df = df.append(mydict, ignore_index = True)
    df.to_csv(csvPath,index=None)


# In[14]:


def process():
    today = date.today()
    fecha = today.strftime("%d-%m-%y")
    
    download_images()
    read_and_update_csv()
#    if True:
#        fileList = drive.ListFile({'q': "'1kiWdUi39wsIQnGUj8XcaUYNda8UgDVq6' in parents and trashed=false"}).GetList()
#        fileID = None
#        for f in fileList:
#            if (f['title']=='riesgoOidio.csv'):
#                fileID = f['id']
#        if fileID is not None:
#            file = drive.CreateFile({'id': fileID})
#            file.Delete()
#        for url in listImagesURLs:
#            upload_file = 'imagenes/'+fecha+'-'+url.split('/')[-1]
#            gfile = drive.CreateFile({'parents': [{'id': '1JUL6UirfJ3StLCbmqr3p-Xs9WVwRAw9F'}]})
#            # Read file and set it as the content of this instance.
#            gfile.SetContentFile(upload_file)
#            gfile.Upload() # Upload the file.
#
#        gfile = drive.CreateFile({'parents': [{'id': '1kiWdUi39wsIQnGUj8XcaUYNda8UgDVq6'}]})
#        gfile.SetContentFile('riesgoOidio.csv')
#        gfile.Upload() # Upload the file.


# In[15]:


schedule.every().day.at("12:40").do(process)
while True:
    schedule.run_pending()
    time.sleep(1)

# In[ ]:




