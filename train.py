import cv2
import numpy as np
import random
import xml.dom.minidom
from non_maximum import non_max_suppression_fast as nms
def gamma_trans(img,gamma):
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)
def load_images(dirname,size):
    img_list = []
    for i in range(size):
        path = dirname + str(i+1) +'.jpg'
        img = cv2.imread(path)
        img_list.append(img)
        path = dirname
    return img_list
def extract_images(path,img_list,size,wsize=(80,80)):
    extract_img = []
    #path = 'E:\\data\\post\\train-PascalVOC-export\\Annotations\\'
    for i in range(size):
        path1 = path + str(i+1) +'.xml'
        doc = xml.dom.minidom.parse(path1)
        root = doc.documentElement
        xminnode = root.getElementsByTagName("xmin")
        xmaxnode = root.getElementsByTagName("xmax")
        ymaxnode = root.getElementsByTagName("ymax") 
        yminnode = root.getElementsByTagName("ymin") 
        xmin = int(float(xminnode[0].childNodes[0].nodeValue))
        xmax = int(float(xmaxnode[0].childNodes[0].nodeValue))
        ymin = int(float(yminnode[0].childNodes[0].nodeValue))
        ymax = int(float(ymaxnode[0].childNodes[0].nodeValue))
        #roi = img_list[i][((ymin+ymax)//2-wsize[1]//2):((ymin+ymax)//2+wsize[1]//2),((xmin+xmax)//2-wsize[0]//2):((xmin+xmax)//2+wsize[0]//2)]
        roi = img_list[i][ymin:ymin+wsize[1],xmin:xmin+wsize[0]]
        #path = 'E:\\data\\post\\train-PascalVOC-export\\Annotations\\'
        if roi.shape[1] != 80 or roi.shape[0] != 80:
            continue
        extract_img.append(roi)
        

    return extract_img
def extract_neg_img(dirname,extract_neg,wsize=(80,80)):
    x = 10
    xmin,xmax,ymin,ymax = 0,0,0,0
    for i in range(15):
        path = dirname + str(i+1)+'.jpg'
        img = cv2.imread(path)
        path = dirname
        for j in range(x):
            xmin = random.randint(1,288-wsize[0])
            ymin = random.randint(1,352-wsize[1])
            xmax = xmin + wsize[0]
            ymax = ymin + wsize[1]
            roi = img[xmin:xmax,ymin:ymax]
            extract_neg.append(roi)
    return extract_neg
def computeHOGs(img_list,gradient_list,wsize=(80,80)):
    hog = cv2.HOGDescriptor((80,80),(40,40),(8,8),(8,8),9) 
    for i in range(len(img_list)):
        img = img_list[i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = gamma_trans(gray,0.8)
        gradient_list.append(hog.compute(gray))    
    return gradient_list


def sliding_window(image, stepSize, windowSize):
  for y in range(0, image.shape[0], stepSize):
    for x in range(0, image.shape[1], stepSize):
        yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])#哪个维度超纲，哪个维度就显示原图
def resize(img, scaleFactor):
  return cv2.resize(img, (int(img.shape[1] * (1 / scaleFactor)), int(img.shape[0] * (1 / scaleFactor))), interpolation=cv2.INTER_AREA)



#读取HOG特征
neg_list = []
pos_list = []
gradient_list = []
labels = []
path1 = 'E:\\data\\pos\\'
path2 = 'E:\\data\\neg\\'
path_pos = 'E:\\data\\post\\train-PascalVOC-export\\Annotations\\'
path_neg = 'E:\\data\\negtive\\negtive-PascalVOC-export\\Annotations\\'
pos_list = load_images(path1,333)
pos_list = extract_images(path_pos,pos_list,333,wsize=(80,80))
neg_list = load_images(path2,193)
neg_list = extract_images(path_neg,neg_list,193,wsize=(80,80))
neg_list = extract_neg_img(path2,neg_list,wsize=(80,80))
#neg_list = extract_neg_img(path2,wsize=(80,80))
computeHOGs(pos_list,gradient_list)
for _ in range(len(pos_list)):
    labels.append(+1)
computeHOGs(neg_list,gradient_list)
for _ in range(len(neg_list)):
    labels.append(-1)
#训练svm
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setGamma(0.001)
svm.setC(30)
svm.setKernel(cv2.ml.SVM_RBF)
svm.train(np.array(gradient_list), cv2.ml.ROW_SAMPLE, np.array(labels))
svm.save("svm.xml")

#读取图片


videoCapture = cv2.VideoCapture('52.avi')
videoCapture.set(cv2.CAP_PROP_POS_FRAMES,4203)
success,img = videoCapture.read()
rectangles = []
counter = 0
scale = 1
w,h = 80,80
font = cv2.FONT_HERSHEY_PLAIN
hog = cv2.HOGDescriptor((80,80),(40,40),(8,8),(8,8),9)  
for (x, y, roi) in sliding_window(img, 10, (80, 80)):#对得到的图进行滑动窗口，(100, 40)为窗口大小，本文应取(64, 64)
    
    if roi.shape[1] != w or roi.shape[0] != h:         #判断是否超纲
        continue
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = gamma_trans(gray,0.8)
    test_gradient = hog.compute(gray)
    _, result = svm.predict(np.array([test_gradient]))
    a, res = svm.predict(np.array([test_gradient]), flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)       
    score = res[0][0]
    if result[0][0] == 1:
        if score < -1:
            print(score)
            rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x+w) * scale), int((y+h) * scale)
            rectangles.append([rx, ry, rx2, ry2, score])
    counter += 1 

print(counter)
windows = np.array(rectangles)
boxes = nms(windows,0.5)
print(len(boxes))
for (x, y, x2, y2, score) in boxes:
  cv2.rectangle(img, (int(x),int(y)),(int(x2), int(y2)),(0, 255, 0), 1)
  cv2.putText(img, "%f" % score, (int(x),int(y)), font, 1, (0, 255, 0))
cv2.imshow("img", img)
cv2.waitKey(0)