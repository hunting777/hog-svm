import numpy as np
def area(box):
  return (abs(box[2] - box[0])) * (abs(box[3] - box[1]))

def overlaps(a, b, thresh=0.5):
  x1 = np.maximum(a[0], b[0])
  x2 = np.minimum(a[2], b[2])
  y1 = np.maximum(a[1], b[1])
  y2 = np.minimum(a[3], b[3])
  intersect = float(area([x1, y1, x2, y2]))
  return intersect / 6400 >= thresh

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh = 0.5):
  # if there are no boxes, return an empty list
  if len(boxes) == 0:
    return []

  scores = boxes[:,4]
  score_idx = np.argsort(scores)#返回scores的从小到大排序的  索引值 
  to_delete = []
  while len(score_idx) > 0:
    box = score_idx[0]
    for s in score_idx:
      if s == score_idx[0]:
        #j=j+1
        continue
      if (overlaps(boxes[s], boxes[box], overlapThresh)):
        to_delete.append(s)
        a = np.where(score_idx == s)
        score_idx = np.delete(score_idx,a)
      #j=j+1
    score_idx = np.delete(score_idx,0)
  boxes = np.delete(boxes,to_delete,axis=0)
  return boxes