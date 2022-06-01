from builtins import input
from google.colab.patches import cv2_imshow
from imutils import paths
import __future__
  #ensure compatibilities of codes from different python function
import argparse, gc, cv2, imutils, os, sys, time
import numpy as np
import subprocess as sp
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False


#####################
# Drawing Functions #
#####################

def draw2graphsCV(imageArray,textArray,stat=True):
  """
  side-by-side opencv image
  """
  image1=imageArray[1]
  text1=textArray[1]
  if len(imageArray)==2:
    image1=imageArray[1]
    text1=textArray[1]
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
    if len(image1.shape)==3:
      vis[:h1, :w1,:]=image1
    else:
      for i in range(0,3):
        vis[:h1, :w1,i]=image1
    if len(image2.shape)==3:
      vis[:h2, :w1:w1+w2,:]=image2
    else:
      for i in range(0,3):
        vis[:h2, w1:w1+w2,i]=image2
    cv2.putText(vis, text1, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		  255, 1)
    cv2.putText(vis, text2, (5+w1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		  255, 1)
    cv2_imshow(vis)
  else:
    cv2_imshow(image1)
  if stat==True:
    median1="{} (median)".format(round(np.median(image1),2))
    mean1="{} (mean)".format(round(np.mean(image1),2))
    sd1="{} (S.D.)".format(round(image1.std(),3))
    if len(imageArray)==2:
      median2="{} (median)".format(round(np.median(image2),2))
      mean2="{} (mean)".format(round(np.mean(image2),2))
      sd2="{} (S.D.)".format(round(image2.std(),3))
      print(">> stat of the LH graph: %s; %s; %s; %s"%(str(image1.shape),median1,mean1,sd1))
      print(">> stat of the RH graph: %s; %s; %s; %s"%(str(image2.shape),median2,mean2,sd2))
    else:
      print(">> stat of the graph: %s; %s; %s; %s"%(str(image1.shape),median1,mean1,sd1))

def drawMultiGraphs(imageArray,titleArray=None,figsize=(15,15),stat=True):
  """
  multiple subplots
  """
  tot=len(imageArray)
  col=3
  row=(tot//col)+1 if tot%col>0 else (tot//col)
  plt.figure(figsize=figsize)
  plt.axis("off")
  for i in range(len(imageArray)):
    image=imageArray[i]
    title=titleArray[i] if titleArray is not None else ""
    plt.subplot(row,col,i+1)
    plt.title(title)
    if len(image.shape)>2:
      plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else: plt.imshow(image, cmap='gray', vmin=0, vmax=255)
  plt.show()
  for i in range(len(imageArray)):
    if stat==True:
      median="{} (median)".format(round(np.median(imageArray[i]),2))
      mean="{} (mean)".format(round(np.mean(imageArray[i]),2))
      sd="{} (S.D.)".format(round(imageArray[i].std(),3))
      print(">> stat of image numbered %d of %s: %s; %s; %s"%(i+1,str(imageArray[i].shape),median,mean,sd))

#################
# Procedure Log #
#################
def logSheets(stat,record,colName):
  try:
    df=pd.read_csv(record, header=[0])
  except:
    df = pd.DataFrame(columns=[colName])
  if len(colName)==1:
    case = {colName: stat}
  else:
    case = {df.columns[i]: stat[i] for i in range(len(df.columns))}
  df=df.append(case, ignore_index = True)
  df.to_csv(record, encoding='utf-8', index=False)
  print("Log record is found in %s"%(record))

######################################
# edge/contour & other editing tools #
######################################
def auto_canny(image, sigma=0.33):
  # compute the median of the single channel pixel intensities
  v = np.median(image)
  # apply automatic Canny edge detection using the computed median
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  print(type(image))
  edged = cv2.Canny(image, lower, upper)
  # return the edged image
  return edged

def clathe(image,edged,clipRange):
  """
  contrast and brightness limiting
  """
  idC=[]
  whitePixel=[]
  equalised=gray=image
  enhancedEdged=edged
  lastChangeWhite=changeWhite=0
  if len(image.shape)>2:
    gray=cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
  for clip in clipRange:
    clip=round(clip,1)
    lastEqualised=equalised
    lastEnhancedEdged=enhancedEdged
    lastChangeWhite=changeWhite
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile))
    equalised = clahe.apply(gray)
    equalised = cv2.medianBlur(equalised, kernSize[0])
    enhancedEdged = auto_canny(equalised)
    curWhite=np.sum(enhancedEdged==255)
    totWhite=curWhite/np.sum(enhancedEdged>=0)
    claheName="clahe{}".format(clip)
    print(">> No. of white pixels (clip={}):{} ".format(clip,curWhite))
    if whitePixel:
      changeWhite=curWhite-whitePixel[-1]
    else:
      changeWhite=curWhite-np.sum(edged==255)
      if changeWhite<=0:
        clip=0
        claheName="clahe0"
        break
    if (changeWhite-lastChangeWhite<=0) or (totWhite>0.10):
      if changeWhite<=0:
        equalised=lastEqualised
        enhancedEdged=lastEnhancedEdged
        if idC[-1].item():
          clip=idC[-1].item()
          claheName="clahe{}".format(idC[-1].item())
        break
      else:
        break
    else:
      whitePixel.append(curWhite)
      idC.append(clip)
  return(equalised,enhancedEdged,clip,claheName)
  
def morphology(gray,option=1):
  """
  image restoration or contour search tool (suite b/w pics well)
  """
  if option==1:
    meth=cv2.MORPH_TOPHAT
  else:
    meth=cv2.MORPH_BLACKHAT
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,2))
  vert = cv2.morphologyEx(gray, meth, kernel)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,13))
  horiz = cv2.morphologyEx(gray, meth, kernel)
  threshH = cv2.threshold(horiz, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  threshV = cv2.threshold(vert, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  hatMask=cv2.bitwise_or(threshV, threshH)
  invMask=cv2.bitwise_not(hatMask) #invMask=255-hatMask for tuple
  return(hatMask,invMask,threshH,threshV)

def repairImage(gray,mask,invMask,params):
  """
  image restoration after morphology
  """
  result=[]
  grayMasked= cv2.bitwise_and(gray, mask)
  collectBlurry=[cv2.medianBlur(gray, kernSize[0])]
  for (diameter, sigmaColor, sigmaSpace) in params:
    collectBlurry.append(cv2.bilateralFilter(gray, diameter, sigmaColor, sigmaSpace))
  for smooth in collectBlurry:
    blurInv=cv2.bitwise_and(smooth, invMask)
    result.append(cv2.add(grayMasked,blurInv))
  return(blurInv,collectBlurry,grayMasked,result)

def resizeImage(image,synDest,imagePath,d=256,limit="max",synOrig="\/temp\/"):
    """
    # Arguments with no defined values come first
    # imagePath is a string
    # This is a reduced very of the function in pad_resize.py
    """
    (y,x)=image.shape[:2]
    if (limit=="max" and y==max(x,y)) or (limit=="min" and y==min(x,y)):
        if y!=d:
            y1=d
            scale=d/y
            x1=int(x*scale)
            method=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC
            newImage = cv2.resize(image, (x1, y1), interpolation = method)
    else:
        if x!=d:
            x1=d
            scale=d/x
            y1=int(y*scale)
            method=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC
            newImage = cv2.resize(image, (x1, y1), interpolation = method)
    newFile="{}{}".format(os.path.splitext(imagePath)[0],"_edited.png")
    newFile=re.sub(synOrig,synDest,newFile,re.I)
    dest=os.path.dirname(newFile)
    print(">> %s has been resized from %s into %s :)"%(os.path.basename(imagePath),str(image.shape),str(newImage.shape)))
    print(">> It locates in %s :)"%(dest))
    if not os.path.exists(dest):
        try:
            os.makedirs(dest)
        except FileExistsError:
            pass
    cv2.imwrite("{}".format(newFile),newImage)
    return scale

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()
