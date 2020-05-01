
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
    [[-6.951657, 2.045311], [-6.755911, 2.887871]],
    [[-6.866550, 1.092112], [-6.951657, 2.045311]],
    [[-6.866550, 1.092112], [-6.296333, 0.232531]],
    [[-5.760159, -0.116408], [-6.296333, 0.232531]],
    [[-5.760159, -0.116408], [-5.045259, -0.465347]],
    [[-5.045259, -0.465347], [-4.168657, -0.712157]],
    [[-4.168657, -0.712157], [-3.292054, -0.754710]],
    [[-2.517580, -0.729178], [-3.292054, -0.754710]],
    [[-2.517580, -0.729178], [-1.879277, -0.729178]],
    [[-1.087782, -0.899393], [-1.879277, -0.729178]],
    [[-1.087782, -0.899393], [0.044142, -1.307906]],
    [[0.044142, -1.307906], [0.937766, -1.622802]],
    [[6.682491, -0.865350], [6.903769, -0.193004]],
    [[6.682491, -0.865350], [6.222913, -1.435567]],
    [[6.852705, 0.547427], [6.903769, -0.193004]],
    [[6.852705, 0.547427], [6.520787, 1.126155]],
    [[6.520787, 1.126155], [5.984613, 1.551690]],
    [[2.052668, 2.402761], [4.180344, 2.028290]],
    [[2.052668, 2.402761], [1.269683, 2.717657]],
    [[-2.075024, 6.470877], [-1.462253, 6.155981]],
    [[-2.721837, 6.487899], [-2.075024, 6.470877]],
    [[-2.721837, 6.487899], [-3.411204, 6.351727]],
    [[-1.462253, 6.155981], [-1.011186, 5.747468]],
    [[-4.194189, 6.028321], [-3.411204, 6.351727]],
    [[-4.194189, 6.028321], [-4.892067, 5.364486]],
    [[-5.334623, 4.853844], [-4.892067, 5.364486]],
    [[-5.334623, 4.853844], [-5.726116, 4.394266]],
    [[-6.347397, 3.577238], [-5.726116, 4.394266]],
    [[-6.347397, 3.577238], [-6.755911, 2.887871]],
    [[0.937766, -1.622802], [1.788836, -1.835570]],
    [[3.337784, -1.929188], [1.788836, -1.835570]],
    [[3.337784, -1.929188], [4.554815, -1.895145]],
    [[6.222913, -1.435567], [5.490992, -1.775995]],
    [[5.320778, 1.781479], [5.984613, 1.551690]],
    [[5.320778, 1.781479], [4.180344, 2.028290]],
    [[-0.687779, 5.194272], [-1.011186, 5.747468]],
    [[-0.687779, 5.194272], [-0.398415, 4.470862]],
    [[-0.126072, 3.807027], [-0.398415, 4.470862]],
    [[-0.126072, 3.807027], [0.427123, 3.262342]],
    [[0.427123, 3.262342], [1.269683, 2.717657]],
    [[4.554815, -1.895145], [5.490992, -1.775995]],
]

def sdTrack(p):
    d = [sdLine(p, line[0], line[1]) for line in TRACK_LINES]

    return opUnion(d)



EXP_STEPS = 120
MAX_STEPS = 200
BANDWIDTH = 0.5

def m_pow(x, power=1.8, shift_y=-1.0):
    return -math.pow(x, power) - shift_y

def m_pow16_7(x):
    return m_pow(x, power=2.285714)

def m_pow4_5(x):
    return m_pow(x, power=1.5)

def hyperbola(x, scale=0.1, shift_x=-0.2):
    return scale/(x-shift_x)

def hyper_decay(x):
    return hyperbola(x, scale=0.1, shift_x=-4.3)

def max_blend(x, y, k=8.0, k1_coef=1.0, k2_coef=2.0, fitness=2.0):
    """
    k: smoothness
    k1_coef: x portion
    k2_coef: y portion
    fitness: log base
    """

    return math.log( math.pow(fitness, k*k1_coef*x)+math.pow(fitness, k*k2_coef*y) ,fitness)/k

def f(x):
    return max_blend(m_pow4_5(abs(x)), hyper_decay(x))

def reward_function(params):
    progress = params['progress']
    steps = params['steps']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']

    p = [params['x'], params['y']]

    def smooth_penalty():
        c = float((MAX_STEPS-EXP_STEPS))
        x = float(progress)/max(float(steps), 1.0)
        return max( -c/(c+float(float(MAX_STEPS)/200.0*EXP_STEPS*x)-float(MAX_STEPS/2.0)+2.0), 0.0)

    def linear_penalty():
        return max(float(progress*(float(EXP_STEPS)/100.0))/max(steps, 1.0), 0.0)

    def one():
        return 1.0

    def penalty():
        return one()

    def rew(x):
        return f(x)*penalty()

    # if car is on the track
    dist = sdTrack(p)/(track_width*BANDWIDTH) if distance_from_center < track_width*0.5 else 1.0

    # clamp track region
    norm_dist = min(dist, 1.0)

    reward = max(f(norm_dist)*penalty(), 0.0)

    return reward
