import numpy as np
from shapely.geometry import Polygon
import time

def intersection(g, p):
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union


def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    g[8] = (g[8] + p[8])
    return g


def weighted_merge_by_width(g, p, img_width, threshold=0.4):

    g_width = g[2] - g[0]
    p_width = p[2] - p[0]

    if g_width < img_width*threshold or p_width < img_width*threshold:
        g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    else:
        g[0] = min(g[0], p[0])
        g[1] = (g[8] * g[1] + p[8] * p[1])/(g[8] + p[8])
        g[2] = max(g[2], p[2])
        g[3] = (g[8] * g[3] + p[8] * p[3]) / (g[8] + p[8])
        g[4] = max(g[4], p[4])
        g[5] = (g[8] * g[5] + p[8] * p[5]) / (g[8] + p[8])
        g[6] = min(g[6], p[6])
        g[7] = (g[8] * g[7] + p[8] * p[7]) / (g[8] + p[8])

    g[8] = (g[8] + p[8])
    return g


def standard_nms(S, thres):

    start = time.time()

    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]
    end = time.time()-start
    print('standard_nms:'+str(end))
    return S[keep]


def nms_locality(polys, img_width, thres=0.3):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    start = time.time()

    S = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            p = weighted_merge_by_width(g, p, img_width)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])

    end = time.time()-start
    print('nms_locality:'+str(end))

    return standard_nms(np.array(S), thres)






if __name__ == '__main__':
    # 343,350,448,135,474,143,369,359
    print(Polygon(np.array([[343, 350], [448, 135],
                            [474, 143], [369, 359]])).area)
