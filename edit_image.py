execfile(os.path.join(os.path.dirname(os.path.abspath(__file__)),"auxillary.py"))
from builtins import input
from google.colab.patches import cv2_imshow
from imutils import paths
import __future__
  #ensure compatibilities of codes from different python function
import argparse, cv2, gc, imutils, os, random, re, sys, time
import numpy as np
import pandas as pd
import subprocess as sp
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help=">> path to input directory of images")
ap.add_argument("-l", "--log", type=str, default=None,
	help=">> path to input log record")
ap.add_argument("-a", "--approach", type=str, default="telea", choices=["telea", "ns", "null"],
	help="inpainting algorithm to use")
ap.add_argument("-c", "--clip", type=float, default=2.0,
	help="threshold for contrast limiting")
	# recommended range: 2-5 (40 was the default)
ap.add_argument("-d", "--dim", type=int, default=256,
	help="dimension of resizing images")
ap.add_argument("-k", "--kernSize", nargs="+", type=int, default=None,
	help="kernSize for canny edge search with a string of two numbers")
ap.add_argument("-r", "--radius", type=int, default=3,help="inpainting radius")
ap.add_argument("-t", "--tile", type=int, default=8,
	help="tile grid size -- divides image into tile x time cells")
	#default setting
args = vars(ap.parse_args())


#############
# variables #
#############
clip=args["clip"]
tile=args["tile"]

"""
Other variable setup
"""
if args["kernSize"] is None:
    newKernel=""
    kernSize=deconvKern=(3,3)
    while newKernel not in [0,1]:
      newKernel=int(input(">> Happy to search edges with the default kernel, %s?\n If yes, please enter 1; otherwise 0 :) "%(str(kernSize))))
    if newKernel!=1:
        print(">> Please re-run the programme with a new kernel, e.g., (5,5)")
        sys.exit()
else: kernSize=tuple(args["kernSize"])
if args["log"] is None:
    run=""
    while run not in [0,1]:
      run=int(input(">> Location of log file has not been specified\n Enter 1 for *test*; otherwise 0 for exist :) " ))
    if run==0:
      sys.exit()
if args["approach"]!="ns":
    flags = cv2.INPAINT_TELEA
else:
    flags = cv2.INPAINT_NS
    print(">> Inpaint approach is changed to 'NS' :)")

extent=""
if args["log"] is not None:
    op=""
    while op not in range(3):
        op=int(input(">> Press '1' for saving edited image in *ready* directory; otherwise '0' for *trial* and '2' for *pooled* :) "))
    synDest=["/trial/","/ready/","/pooled/"][op]
    while extent not in [0,1,2]:
        extent=int(input(">> Enter 1 inpaint with *%s*\n **Otherwise enter 0 for *No* or 2 for *Partial* :) "%(args["approach"])))



###################
# Main Algorithms #
###################
def editing(imagePath,clipRange=np.arange(clip,clip+5,0.5)):
  image=gray=cv2.imread(imagePath)
  if len(image.shape)==3:
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.medianBlur(gray, kernSize[0])
  edged = auto_canny(blurred)
  fm=variance_of_laplacian(gray)
  if fm>200:
      clipRange=np.arange(clipRange[0],clipRange[0]+2,1)
  (equalised,enhancedEdged,clip,claheName)=clathe(image,edged,clipRange)
  (equalised,enhancedEdged,clip,claheName)=clathe(image,edged,np.arange(clip+0.1,clip+1,0.1))
  white=round(np.sum(enhancedEdged==255)/np.sum(enhancedEdged>=0),3)
  blank=np.ones(image.shape[:2], np.uint8)
  fld = cv2.ximgproc.createFastLineDetector()
  lines = fld.detect(blurred)
  inpaint=fld.drawSegments(blank,lines)
  inpaint=auto_canny(inpaint)
  cnts = cv2.findContours(inpaint.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  if cnts:
    for c in cnts:
      if np.median(image[c]) <100 or np.median(image[c]) >240:
        cv2.drawContours(image, [c], -1, (0,255,0), 1)
        if args["log"] is None or extent==1:
            cv2.drawContours(blank, [c], -1, 255, 1)
  cnts = cv2.findContours(enhancedEdged.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  if cnts:
    for c in cnts:
      if cv2.contourArea(c)<=2:
        cv2.drawContours(blank, [c], -1, 255, -1)
        cv2.drawContours(image, [c], -1, (255,0,0), -1)
      (x,y),rad = cv2.minEnclosingCircle(c)
      center = (int(x),int(y))
      rad = int(rad)
      if rad<=2:
        cv2.circle(blank,center,rad,255,-1)
        cv2.circle(image,center,rad,(0,0,255),-1)
  thres=cv2.threshold(blank, 0, 255, cv2.THRESH_OTSU)[1]
  radius=args["radius"] if (white<0.05 or args["radius"]>5) else 5
  print(">> We are using radius=%d to perform inpaint :)"%(radius))
  outTELEA = cv2.inpaint(equalised, thres, radius, flags=flags)
  if args["log"] is None:
    images=[image,gray,equalised]
    titles=[os.path.basename(imagePath),"greyscale","%s /w VoL %s"%(claheName,round(fm,2))]
    drawMultiGraphs(images,titles)  
    images=[edged,enhancedEdged,inpaint]
    titles=["edged before enhancement","edged after enhancement (/w white: %s)"%(white),"edge from fastline"]
    drawMultiGraphs(images,titles)
    images=[blank,equalised,outTELEA]
    titles=["inpaint mask","%s"%(claheName),"TELEA inpaint"]
    drawMultiGraphs(images,titles)
  else:
    if extent!=0:
      accepted=outTELEA
      print("***No inpaint***")
    else: accepted=equalised
    scale=resizeImage(accepted,synDest,imagePath,d=256,limit="max")
    if count:
      if random.randint(0, 10)%10==0 or count==0:
        images=[gray,equalised,outTELEA]
        titles=["greyscale of %s"%(os.path.basename(imagePath)),"%s"%(claheName),"%s inpaint"%(flags)]
        drawMultiGraphs(images,titles)
    brand=os.path.basename(imagePath).split("_")[0]
    return(os.path.basename(imagePath),brand,str(image.shape),round(scale,2),claheName,round(white,3),imagePath)
  gc.collect()
  
###########
# Looping #
###########
###########
# Routine #
###########
route="""
logDir=args["log"]
redList=os.path.join(logDir,"undone.csv")
whiteList=os.path.join(logDir,"readMe_xray_edited.csv")
colNames=['case','brand', 'size', 'scale', 'claheName', 'inpainted.approx','origin']
chk=os.path.splitext(imagePath)
if chk[1]=='':
  logSheets(chk[0],record=redList,colName="unreadable")
  imagePath='%s.png'%(chk[0])
try:
  stat=editing(imagePath)
  logSheets(stat,record=whiteList,colName=colNames)
except:
  if chk[1]!='':
    logSheets(imagePath,record=redList,colName="unreadable")
"""
##############
# Running... #
##############
source=imagePath=args["input"]
count=0
if os.path.isdir(source):
  for imagePath in paths.list_images(source):
    if args["log"] is None:
      editing(imagePath)
    else: exec(route)
    count+=0
else:
  if os.path.isabs(imagePath):
    if args["log"] is None:
      editing(imagePath)
    else: exec(route)

