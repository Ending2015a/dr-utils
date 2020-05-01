
import math

def clamp(n, l, r):
    l, r = (l, r) if l < r else (r, l)
    return min(max(n, l), r)

def vMul(a, h):
    return [a[0]*h, a[1]*h]

def vSub(a, b):
    return [a[0]-b[0],a[1]-b[1]]

def vDot(a, b):
    return a[0]*b[0]+a[1]*b[1]

def vLength(a):
    return math.sqrt(sum([e**2 for e in a]))

def opUnion(n):
    """
    n: a list of signed distances
    """
    return min(n)

def sdLine(p, a, b):
    """
    p: point [x, y]
    a: start of line [x, y]
    b: end of line [x, y]
    """
    pa, ba = vSub(p, a), vSub(b, a)
    h = clamp(vDot(pa, ba)/vDot(ba, ba), 0.0, 1.0)

    return vLength(vSub(pa, vMul(ba, h)))

TRACK_LINES = [
    [[-3.164394, 6.181513], [-6.304844, 2.555954]],
    [[-6.789954, 1.347433], [-6.696336, 1.934672]],
    [[-6.670804, 0.734663], [-6.789954, 1.347433]],
    [[-6.670804, 0.734663], [-6.296333, 0.232531]],
    [[-5.760159, -0.116408], [-6.296333, 0.232531]],
    [[-5.760159, -0.116408], [-5.045259, -0.465347]],
    [[-5.045259, -0.465347], [-4.168657, -0.712157]],
    [[-4.168657, -0.712157], [-3.292054, -0.754710]],
    [[-2.517580, -0.729178], [-3.292054, -0.754710]],
    [[-2.517580, -0.729178], [-1.879277, -0.729178]],
    [[-1.300550, -0.873860], [-1.879277, -0.729178]],
    [[-1.300550, -0.873860], [0.112227, -1.358971]],
    [[1.048405, -1.520674], [0.112227, -1.358971]],
    [[1.048405, -1.520674], [3.635659, -1.622802]],
    [[4.699497, -1.571738], [3.635659, -1.622802]],
    [[4.699497, -1.571738], [5.627163, -1.393013]],
    [[6.359084, -1.018542], [5.627163, -1.393013]],
    [[6.359084, -1.018542], [6.716534, -0.371729]],
    [[6.793130, 0.266574], [6.716534, -0.371729]],
    [[6.793130, 0.266574], [6.648448, 0.879345]],
    [[6.648448, 0.879345], [6.265466, 1.355944]],
    [[2.027136, 2.385739], [5.525035, 1.653819]],
    [[0.154781, 3.424045], [2.027136, 2.385739]],
    [[0.154781, 3.424045], [-0.330329, 4.079370]],
    [[-1.998427, 6.402792], [-1.487785, 6.087895]],
    [[-2.594177, 6.436834], [-1.998427, 6.402792]],
    [[-2.594177, 6.436834], [-3.164394, 6.181513]],
    [[-6.696336, 1.934672], [-6.304844, 2.555954]],
    [[-0.330329, 4.079370], [-0.926079, 5.151718]],
    [[-1.487785, 6.087895], [-0.926079, 5.151718]],
    [[5.525035, 1.653819], [6.265466, 1.355944]],
]

def sdTrack(p):
    d = [sdLine(p, line[0], line[1]) for line in TRACK_LINES]

    return opUnion(d)



STEPS_PROGRESS_RATE = 1.5

def reward_function(params):
    progress = params['progress']
    steps = params['steps']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']

    p = [params['x'], params['y']]

    def pow16_7(x):
        return -math.pow(abs(x), 2.285714) + 1

    def penalty():
        return float(progress*STEPS_PROGRESS_RATE)/max(float(steps), 1.0)

    def f(x):
        return pow16_7(x)

    norm_dist = distance_from_center/(track_width*0.5)

    # if car is on the track
    dist = sdTrack(p) if distance_from_center < track_width*0.5 else distance_from_center

    # clamp track region
    norm_dist = min(dist/(track_width*0.5), 1.0)

    reward = clamp(f(norm_dist)*penalty(), 0.0, 1.0)

    return reward
