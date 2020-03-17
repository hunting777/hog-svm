import cv2
import numpy as np
from random import randint
from non_maximum import non_max_suppression_fast as nms
def gamma_trans(img,gamma):
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)
def createTrackerByName(trackerType):  
    # Create a tracker based on tracker name  
    trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    if trackerType == trackerTypes[0]:    
        tracker = cv2.TrackerBoosting_create()  
    elif trackerType == trackerTypes[1]:     
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:    
        tracker = cv2.TrackerKCF_create()  
    elif trackerType == trackerTypes[3]:    
        tracker = cv2.TrackerTLD_create()  
    elif trackerType == trackerTypes[4]:    
        tracker = cv2.TrackerMedianFlow_create()  
    elif trackerType == trackerTypes[5]:    
        tracker = cv2.TrackerGOTURN_create()  
    elif trackerType == trackerTypes[6]:    
        tracker = cv2.TrackerMOSSE_create()  
    elif trackerType == trackerTypes[7]:    
        tracker = cv2.TrackerCSRT_create()  
    else:    
        tracker = None    
        print('Incorrect tracker name')    
        print('Available trackers are:')    
        for t in trackerTypes:      
          print(t)       
    return tracker
def sliding_window(image,stepSize,windowSize):
  for y in range(0, image.shape[0]-100, stepSize):
    for x in range(0, image.shape[1]-100, stepSize):
        yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
def area(box):
  return (abs(box[2] - box[0])) * (abs(box[3] - box[1]))
def overlaps(a, b, thresh=0.5):
  x1 = np.maximum(a[0], b[0])
  x2 = np.minimum(a[2], b[2])
  y1 = np.maximum(a[1], b[1])
  y2 = np.minimum(a[3], b[3])
  intersect = float(area([x1, y1, x2, y2]))
  return intersect / 6400 >= thresh
def compare(box1,box2):
  disappear = []
  left = []
  left_inx = []
  left1_inx = []
  left1 = []
  exist =[]
  for i in range(len(box2)):
    for j in range(len(box1)):
      if (overlaps(box2[i], box1[j], 0.5)):
        left.append(box2[i])
        left_inx.append(i)
        left1.append(box1[j])
        left1_inx.append(j)
  disappear = np.delete(box1,left1_inx,axis=0)
  exist = np.delete(box2,left_inx,axis=0)
  return disappear,left,exist
def desize(b,img):
  roi = img[int(b[1]):int(b[3]),int(b[0]):int(b[2])]
  roi = cv2.resize(roi, (80,80))
  gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
  gray = cv2.equalizeHist(gray)
  test_gradient = hog.compute(gray)
  _, result = svm2.predict(np.array([test_gradient]))
  
  #print(result[0][0])
  if result[0][0] == 1:
    return result[0][0]
  else:
    return 0

def trackingeffect(bbox,boxes2,boxes3,coordinates,img):
  a = []
  b = []
  tracker = np.zeros(len(boxes2))
  for i in range(len(bbox)):
    a = (bbox[i][0],bbox[i][1],bbox[i][0]+bbox[i][2],bbox[i][1]+bbox[i][3])
    b = (boxes2[i][0],boxes2[i][1],boxes2[i][0]+boxes2[i][2],boxes2[i][1]+boxes2[i][3])
    if (overlaps(a,b)):
      if b[1] < 0 or b[0]<0 or b[2]>352 or b[3]>288:
        tracker[i] = 1
      else:
        if (desize(b,img)):
          tracker[i] = 0
        else:
          tracker[i] = 1
    else:
      tracker[i] = 1
    coordinates.append([])
    coordinates[i].append((boxes2[i][0],boxes2[i][1]))  
  boxes3 = tracker
  return boxes3,coordinates
  """ if len(boxes3) == 0:
    boxes3 = tracker
  else:
    for i in range(len(boxes3)):
      if tracker[i]==0:
        boxes3[i] = 0
      else:
        boxes3[i] = boxes3[i] + tracker[i]
    for i in range(len(boxes3),len(tracker)):
      boxes3.append(tracker[i]) """
  
    

def offcount(track,counter1):  #计算下车人数
  thresh = 35
  if len(track) == 0:
    return counter1
  for i in range(len(track)):
    if track[i][len(track[i])-1][1] < thresh and track[i][0][1] > track[i][len(track[i])-1][1]:
      #and track[i][0][1] > 40
      counter1 = counter1 + 1
  return counter1
def upcount(counter2,left,img1):
  if len(left) == 0:
    return counter2
  for i in range(len(left)):
    if (left[i][1]+left[i][3])//2 < 40:
      if (desize(left[i],img)):
        counter2 = counter2 + 1
  return counter2

def update_bbox(boxes1,boxes2,color):
  a = []
  b = []
  to_delete = []
  bbox = []
  for i in range(len(boxes1)): 
    for j in range(len(boxes2)):
      a = boxes1[i]
      b = (boxes2[j][0],boxes2[j][1],boxes2[j][0]+boxes2[j][2],boxes2[j][1]+boxes2[j][3])
      if (overlaps(a,b)):
        to_delete.append(i)
        break
  boxes1 = np.delete(boxes1,to_delete,axis=0)
  for b in boxes2:
    roi = (b[0],b[1],b[2],b[3])
    bbox.append(roi)
  for b in boxes1:
    roi = (b[0],b[1],b[2]-b[0],b[3]-b[1])
    bbox.append(roi)
    color.append((randint(0, 255), randint(0, 255), randint(0, 255)))
  return bbox,color
svm2 = cv2.ml.SVM_load("svm.xml")
videoCapture = cv2.VideoCapture('52.avi')
videoCapture.set(cv2.CAP_PROP_POS_FRAMES,17600)
success,img1 = videoCapture.read()
rectangles = []
counter1 = 32
counter2 = 0
scale = 1
w,h = 80,80
font = cv2.FONT_HERSHEY_PLAIN
hog = cv2.HOGDescriptor((80,80),(40,40),(8,8),(8,8),9)  
for (x, y, roi) in sliding_window(img1, 8, (80, 80)):#对得到的图进行滑动窗口，(100, 40)为窗口大小，本文应取(64, 64)
    if roi.shape[1] != w or roi.shape[0] != h:         #判断是否超纲
        continue
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = gamma_trans(gray,0.8)
    test_gradient = hog.compute(gray)
    _, result = svm2.predict(np.array([test_gradient]))
    a, res = svm2.predict(np.array([test_gradient]), flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)       
    score = res[0][0]
    if result[0][0] == 1:
        if score < -1:
            rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x+w) * scale), int((y+h) * scale)
            rectangles.append([rx, ry, rx2, ry2, score])
windows = np.array(rectangles)
boxes = nms(windows,0.5)
for (x, y, x2, y2, score) in boxes:
  cv2.rectangle(img1, (int(x),int(y)),(int(x2), int(y2)),(0, 255, 0), 1)
  cv2.putText(img1, "%f" % score, (int(x),int(y)), font, 1, (0, 255, 0))
  cv2.putText(img1, "get on:%d" % 0, (15,15), font, 1, (0, 255, 0))
  cv2.putText(img1, "get off:%d" % 0, (15,35), font, 1, (0, 255, 0))
bbox = []
color = []
for b in boxes:
  roi = (b[0],b[1],b[2]-b[0],b[3]-b[1])
  bbox.append(roi)
  color.append((randint(0, 255), randint(0, 255), randint(0, 255)))
img = img1
cv2.imshow("img", img1)
cv2.waitKey(0)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
eg = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('m','p','4','v'),fps,size)
trackerType = 'CSRT'
boxes3 = []
coordinates = [[]]
exist_update = [[]]
fuck = 0
t = 40
for i in range(1,500):
  rectangles = []
  scale = 1
  w,h = 80,80
  font = cv2.FONT_HERSHEY_PLAIN
  for (x, y, roi) in sliding_window(img1, t, (80,80)):#对得到的图进行滑动窗口，(100, 40)为窗口大小，本文应取(64, 64)   
    if roi.shape[1] != w or roi.shape[0] != h:         #判断是否超纲
        continue
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = gamma_trans(gray,0.8)
    test_gradient = hog.compute(gray)
    _, result = svm2.predict(np.array([test_gradient]))
    a, res = svm2.predict(np.array([test_gradient]), flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)       
    score = res[0][0]
    if result[0][0] == 1:
      if score < -1:
        rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x+w) * scale), int((y+h) * scale)
        rectangles.append([rx, ry, rx2, ry2, score])
  windows = np.array(rectangles)
  boxes1 = nms(windows,0.5)
  disappear,left,exist = compare(boxes,boxes1)
  for l in exist:
    exist_update[fuck].append(l)
  exist_update.append([])
  fuck = fuck + 1
  multiTracker = cv2.MultiTracker_create()
  for box in bbox:
    multiTracker.add(createTrackerByName(trackerType), img, box)
  su, boxes2 = multiTracker.update(img1)
  boxes3,coordinates = trackingeffect(bbox,boxes2,boxes3,coordinates,img1)  #boxes3保存跟踪到的去情况和跟踪到的位置
  to_delete = []
  track = []
  for k in range(len(boxes3)):
    if boxes3[k] > 0:
      to_delete.append(k)  #可以在另加检测之后再删
      track.append(coordinates[k])
    
  boxes2 = np.delete(boxes2,to_delete,axis=0)
  coordinates = np.delete(coordinates,to_delete,axis=0)
  coordinates=coordinates.tolist()
  boxes3 = np.delete(boxes3,to_delete,axis=0)
  boxes3 = boxes3.tolist()
  counter1 = offcount(track,counter1)  #计算下车人数
  img = img1
  #bbox,color = update_bbox(boxes1,boxes2,color)
  if fuck > 2:
    disappear,left,exist = compare(exist_update[fuck-3],boxes)
    disappear,left,exist = compare(left,boxes1)
    counter2 = upcount(counter2,left,img1)
    bbox,color = update_bbox(left,boxes2,color)
  boxes = boxes1
  for j, newbox in enumerate(bbox):    
    p1 = (int(newbox[0]), int(newbox[1]))    
    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))    
    cv2.rectangle(img, p1, p2, color[j],1)
  multiTracker.clear()
  print(i)

  cv2.putText(img1, "get off:%d" % counter1, (15,15), font, 1, (0, 255, 0))
  cv2.putText(img1, "get on:%d" % counter2, (15,35), font, 1, (0, 255, 0))
  eg.write(img1)
  success,img1 = videoCapture.read()
