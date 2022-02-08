# @title
import cv2 as cv
import numpy as np
import time


class Label:

    def __init__(self, cl=-1, tl=np.array([0., 0.]), br=np.array([0., 0.]), prob=None):
        self.__tl = tl
        self.__br = br
        self.__cl = cl
        self.__prob = prob

    def __str__(self):
        return 'Class: %d, top_left(x:%f,y:%f), bottom_right(x:%f,y:%f)' % (
            self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])

    def copy(self):
        return Label(self.__cl, self.__tl, self.__br)

    def wh(self): return self.__br - self.__tl

    def cc(self): return self.__tl + self.wh() / 2

    def tl(self): return self.__tl

    def br(self): return self.__br

    def tr(self): return np.array([self.__br[0], self.__tl[1]])

    def bl(self): return np.array([self.__tl[0], self.__br[1]])

    def cl(self): return self.__cl

    def area(self): return np.prod(self.wh())

    def prob(self): return self.__prob

    def set_class(self, cl):
        self.__cl = cl

    def set_tl(self, tl):
        self.__tl = tl

    def set_br(self, br):
        self.__br = br

    def set_wh(self, wh):
        cc = self.cc()
        self.__tl = cc - .5 * wh
        self.__br = cc + .5 * wh

    def set_prob(self, prob):
        self.__prob = prob

class DLabel(Label):

    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, 1)
        br = np.amax(pts, 1)
        Label.__init__(self, cl, tl, br, prob)


def nms(Labels, iou_threshold=.5):
    SelectedLabels = []
    Labels.sort(key=lambda l: l.prob(), reverse=True)

    for label in Labels:
        non_overlap = True
        for sel_label in SelectedLabels:
            if IOU_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break
        if non_overlap:
            SelectedLabels.append(label)

    return SelectedLabels


def IOU(tl1, br1, tl2, br2):
    wh1, wh2 = br1 - tl1, br2 - tl2
    assert ((wh1 >= .0).all() and (wh2 >= .0).all())

    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0.)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area;
    return intersection_area / union_area


def IOU_labels(l1, l2):
    return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())


def getRectPts(tlx, tly, brx, bry):
    return np.matrix([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1., 1., 1., 1.]], dtype=float)


def find_T_matrix(pts, t_pts):
    A = np.zeros((8, 9))
    for i in range(0, 4):
        xi = pts[:, i];
        xil = t_pts[:, i];
        xi = xi.T

        A[i * 2, 3:6] = -xil[2] * xi
        A[i * 2, 6:] = xil[1] * xi
        A[i * 2 + 1, :3] = xil[2] * xi
        A[i * 2 + 1, 6:] = -xil[0] * xi

    [U, S, V] = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))

    return H


def reconstruct(Iorig, I, Y, out_size, threshold=.9):
    net_stride = 2 ** 4
    side = ((208. + 40.) / 2.) / net_stride  # 7.75

    Probs = Y[..., 0]
    # print Probs
    Affines = Y[..., 2:]
    rx, ry = Y.shape[:2]
    # print Y.shape
    ywh = Y.shape[1::-1]
    # print ywh
    iwh = np.array(I.shape[1::-1], dtype=float).reshape((2, 1))
    # print iwh

    xx, yy = np.where(Probs > threshold)
    # print xx,yy

    WH = getWH(I.shape)
    MN = WH / net_stride

    # print MN

    vxx = vyy = 0.5  # alpha

    base = lambda vx, vy: np.matrix([[-vx, -vy, 1.], [vx, -vy, 1.], [vx, vy, 1.], [-vx, vy, 1.]]).T
    labels = []

    for i in range(len(xx)):
        y, x = xx[i], yy[i]
        affine = Affines[y, x]
        prob = Probs[y, x]

        mn = np.array([float(x) + .5, float(y) + .5])

        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0.)
        A[1, 1] = max(A[1, 1], 0.)
        # print A
        pts = np.array(A * base(vxx, vyy))  # *alpha
        # print pts
        pts_MN_center_mn = pts * side
        pts_MN = pts_MN_center_mn + mn.reshape((2, 1))

        pts_prop = pts_MN / MN.reshape((2, 1))

        labels.append(DLabel(0, pts_prop, prob))

    # print(labels)
    final_labels = nms(labels, .1)
    TLps = []

    if len(final_labels):
        final_labels.sort(key=lambda x: x.prob(), reverse=True)
        for i, label in enumerate(final_labels):
            t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
            ptsh = np.concatenate((label.pts * getWH(Iorig.shape).reshape((2, 1)), np.ones((1, 4))))
            H = find_T_matrix(ptsh, t_ptsh)
            Ilp = cv.warpPerspective(Iorig, H, out_size, borderValue=.0)
            # cv.imshow("frame", Iorig)
            # cv.waitKey(0)

            TLps.append(Ilp)

    return final_labels, TLps


def im2single(I):
    assert (I.dtype == 'uint8')
    return I.astype('float32') / 255.


def getWH(shape):
    return np.array(shape[1::-1]).astype(float)


def detect_lp(model, I, max_dim, net_step, out_size, threshold):
    min_dim_img = min(I.shape[:2])
    factor = float(max_dim) / min_dim_img
    # print I.shape[:2]

    w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist()
    w += (w % net_step != 0) * (net_step - w % net_step)
    h += (h % net_step != 0) * (net_step - h % net_step)
    # print w
    # print h
    Iresized = cv.resize(I, (w, h))

    T = Iresized.copy()
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))

    start = time.time()
    Yr = model.predict(T)
    Yr = np.squeeze(Yr)
    elapsed = time.time() - start
    # print(Yr)
    L, TLps = reconstruct(I, Iresized, Yr, out_size, threshold)

    return L, TLps, elapsed


# @title
# Get the names of the output layers
def getOutputsNames(net):
    """ Get the names of the output layers.

    Generally in a sequential CNN network there will be
    only one output layer at the end. In the YOLOv3
    architecture, there are multiple output layers giving
    out predictions. This function gives the names of the
    output layers. An output layer is not connected to
    any next layer.

    Args
      net : neural network
    """
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, confThreshold, nmsThreshold=0.4):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    predictions = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    if nmsThreshold:
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    else:
        indices = [[x] for x in range(len(boxes))]

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        predictions.append([classIds[i], confidences[i], [left, top, left + width, top + height]])

    return predictions


# Draw the predicted bounding box
def drawPred(frame, pred):
    classId = pred[0]
    conf = pred[1]
    box = pred[2]
    left, top, right, bottom = box[0], box[1], box[2], box[3]
    # draw bounding box
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)


def crop_region(I, label, bg=0.5):
    wh = np.array(I.shape[1::-1])

    ch = I.shape[2] if len(I.shape) == 3 else 1
    tl = np.floor(label.tl() * wh).astype(int)
    br = np.ceil(label.br() * wh).astype(int)
    outwh = br - tl

    if np.prod(outwh) == 0.:
        return None

    outsize = (outwh[1], outwh[0], ch) if ch > 1 else (outwh[1], outwh[0])
    if (np.array(outsize) < 0).any():
        pause()
    Iout = np.zeros(outsize, dtype=I.dtype) + bg

    offset = np.minimum(tl, 0) * (-1)
    tl = np.maximum(tl, 0)
    br = np.minimum(br, wh)
    wh = br - tl

    Iout[offset[1]:(offset[1] + wh[1]), offset[0]:(offset[0] + wh[0])] = I[tl[1]:br[1], tl[0]:br[0]]

    return Iout


def dknet_label_conversion(R, img_width, img_height, classes):
    WH = np.array([img_width, img_height], dtype=float)
    L = []
    for r in R:
        center = np.array(r[2][:2]) / WH
        wh2 = (np.array(r[2][2:]) / WH) * .5
        L.append(Label(ord(classes[r[0]]), tl=center - wh2, br=center + wh2, prob=r[1]))
    return L


def draw_label(I, l, color=(255, 0, 0), thickness=1):
    wh = np.array(I.shape[1::-1]).astype(float)
    tl = tuple((l.tl() * wh).astype(int).tolist())
    br = tuple((l.br() * wh).astype(int).tolist())
    cv.rectangle(I, tl, br, color, thickness=thickness)


def draw_losangle(I, pts, color=(1., 1., 1.), thickness=1):
    assert (pts.shape[0] == 2 and pts.shape[1] == 4)

    for i in range(4):
        pt1 = tuple(pts[:, i].astype(int).tolist())
        pt2 = tuple(pts[:, (i + 1) % 4].astype(int).tolist())
        cv.line(I, pt1, pt2, color, thickness)


def write2img(Img, label, strg, txt_color=(0, 0, 0), bg_color=(255, 255, 255), font_size=1):
    wh_img = np.array(Img.shape[1::-1])

    font = cv.FONT_HERSHEY_SIMPLEX

    wh_text, v = cv.getTextSize(strg, font, font_size, 3)
    bl_corner = label.tl() * wh_img

    tl_corner = np.array([bl_corner[0], bl_corner[1] - wh_text[1]]) / wh_img
    br_corner = np.array([bl_corner[0] + wh_text[0], bl_corner[1]]) / wh_img
    bl_corner /= wh_img

    if (tl_corner < 0.).any():
        delta = 0. - np.minimum(tl_corner, 0.)
    elif (br_corner > 1.).any():
        delta = 1. - np.maximum(br_corner, 1.)
    else:
        delta = 0.

    tl_corner += delta
    br_corner += delta
    bl_corner += delta

    tpl = lambda x: tuple((x * wh_img).astype(int).tolist())

    cv.rectangle(Img, tpl(tl_corner), tpl(br_corner), bg_color, -1)
    cv.putText(Img, strg, tpl(bl_corner), font, font_size, txt_color, 3)
