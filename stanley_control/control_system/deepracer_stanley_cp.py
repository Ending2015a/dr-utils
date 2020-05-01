"""

Path tracking simulation with Stanley steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)

Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

"""
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
sys.path.append("../PythonRobotics/PathPlanning/CubicSpline/")

try:
    import cubic_spline_planner
except:
    raise

ACTIONS = [0.00356548, 0.12381341, -0.15538202, -0.05105542, 0.06705348, -0.34906585, 0.03286018]   # available steering angle of front wheel in radians


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
    [[-1.087782, -0.899393], [0.018610, -1.401524]],
    [[0.018610, -1.401524], [0.937766, -1.673867]],
    [[0.937766, -1.673867], [1.729261, -1.793016]],
    [[3.397359, -1.852592], [1.729261, -1.793016]],
    [[3.397359, -1.852592], [4.571836, -1.818549]],
    [[4.571836, -1.818549], [5.397375, -1.673867]],
    [[5.397375, -1.673867], [6.010145, -1.461099]],
    [[6.605895, -0.882371], [6.010145, -1.461099]],
    [[6.605895, -0.882371], [6.869726, -0.175983]],
    [[6.801641, 0.530406], [6.869726, -0.175983]],
    [[6.801641, 0.530406], [6.520787, 1.126155]],
    [[6.520787, 1.126155], [5.984613, 1.551690]],
    [[5.303757, 1.738926], [5.984613, 1.551690]],
    [[5.303757, 1.738926], [4.180344, 2.028290]],
    [[2.052668, 2.445314], [4.180344, 2.028290]],
    [[1.278194, 2.768721], [2.052668, 2.445314]],
    [[0.478188, 3.330427], [1.278194, 2.768721]],
    [[-0.100540, 3.815537], [0.478188, 3.330427]],
    [[-0.398415, 4.470862], [-0.100540, 3.815537]],
    [[-0.687779, 5.194272], [-0.398415, 4.470862]],
    [[-0.687779, 5.194272], [-1.011186, 5.747468]],
    [[-1.462253, 6.155981], [-1.011186, 5.747468]],
    [[-2.075024, 6.470877], [-1.462253, 6.155981]],
    [[-2.721837, 6.487899], [-2.075024, 6.470877]],
    [[-2.721837, 6.487899], [-3.411204, 6.351727]],
    [[-4.194189, 6.028321], [-3.411204, 6.351727]],
    [[-4.194189, 6.028321], [-4.892067, 5.364486]],
    [[-5.334623, 4.853844], [-4.892067, 5.364486]],
    [[-5.334623, 4.853844], [-5.726116, 4.394266]],
    [[-6.347397, 3.577238], [-5.726116, 4.394266]],
    [[-6.347397, 3.577238], [-6.755911, 2.887871]],
]


k = 2.5  # control gain
Kp = 5.0  # speed proportional gain
dt = 0.04  # [s] time difference
L = 0.2  # [m] Wheel base of vehicle
max_steer = np.radians(20.0)  # [rad] max steering angle

show_animation = True

class State(object):
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.

        Stanley Control uses bicycle model.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt


def pid_control(target, current):
    """
    Proportional control for the speed.

    :param target: (float)
    :param current: (float)
    :return: (float)
    """
    return Kp * (target - current)


def stanley_control(state, cx, cy, cyaw, last_target_idx):
    """
    Stanley steering control.

    :param state: (State object)
    :param cx: ([float])
    :param cy: ([float])
    :param cyaw: ([float])
    :param last_target_idx: (int)
    :return: (float, int)
    """
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy)

    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # theta_e corrects the heading error
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    # theta_d corrects the cross track error
    theta_d = np.arctan2(k * error_front_axle, state.v)
    # Steering control
    delta = theta_e + theta_d

    # Discretize the output angle
    delta_ = choose_angle(delta)

    return delta_, current_target_idx


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def calc_target_index(state, cx, cy):
    """
    Compute index in the trajectory list of the target.

    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """
    # Calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # Search nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    # Project RMS error onto front axle vector
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                      -np.sin(state.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

    return target_idx, error_front_axle

def get_ax_ay():
    '''
    Decode sequential points from TRACK_LINES.
    '''
    ax = []
    ay = []
    points = set()
    # heuristic
    ax.append(TRACK_LINES[0][1][0])
    ay.append(TRACK_LINES[0][1][1])
    points.add(tuple(TRACK_LINES[0][1]))

    for line in TRACK_LINES:
        for p in line:
            if tuple(p) not in points:
                ax.append(p[0])
                ay.append(p[1])
                points.add(tuple(p))
#    ax.append(TRACK_LINES[0][1][0])
#    ay.append(TRACK_LINES[0][1][1])

    return ax, ay

def choose_angle(a, action=None):
    '''
    Map input angle to the nearest predefine angle.
    '''
    if action is None:
        action =  ACTIONS       # global var
    d = np.inf
    idx = None
    for i, c in enumerate(action):
        if a * c > 0:
            if abs(c - a) < d:
                d = abs(c - a)
                idx = i
    return action[idx]

def main():
    """Plot an example of Stanley steering control on a cubic spline."""
        
    #  target course
    # ax = [0.0, 100.0, 100.0, 50.0, 60.0]
    # ay = [0.0, 0.0, -30.0, -20.0, 0.0]
    ax, ay = get_ax_ay()

    sta_angle = []  # log

    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=1)

    target_speed = 4  # [m/s]

    max_simulation_time = 22.0

    # Initial state
    state = State(x=6.755911, y=2.887871, yaw=np.radians(250.0), v=0.0)
    last_idx = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_idx, _ = calc_target_index(state, cx, cy)

    while max_simulation_time >= time and last_idx > target_idx:
        ai = pid_control(target_speed, state.v)
        di, target_idx = stanley_control(state, cx, cy, cyaw, target_idx)   #TODO check target_id
        state.update(ai, di)

        sta_angle.append(di)

        time += dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(ax, ay, ".y", label="course")
            
            for i in range(len(ax)):
                plt.annotate(i, (ax[i], ay[i]))
                
            plt.plot(cx, cy, ".r", label="course_interp")

            #plt.plot([x+0.3 for x in cx], [y for y in cy], ".r", label="course")
            #plt.plot([x-0.3 for x in cx], [y for y in cy], ".r")
            #plt.plot([x for x in cx], [y+0.3 for y in cy], ".r")
            #plt.plot([x for x in cx], [y-0.3 for y in cy], ".r")

            plt.plot(x, y, "-b", label="trajectory")
            plt.plot(cx[target_idx], cy[target_idx], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert last_idx >= target_idx, "Cannot reach goal"

    # Calculate pre-define angles
    # sta_angle = np.array(sta_angle)
    # sta_angle = np.clip(sta_angle, -max_steer, max_steer)
    # print('max', np.degrees(max(sta_angle)))
    # print('min', np.degrees(min(sta_angle)))
    # kmeans = KMeans(n_clusters=7, random_state=0).fit(sta_angle.reshape(-1, 1))
    # print("Center\n", kmeans.cluster_centers_)

    if show_animation:  # pragma: no cover
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(t, [iv * 3.6 for iv in v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()
