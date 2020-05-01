import os
import sys
import time
import logging

from datetime import datetime

import numpy as np
import argparse
import copy


from tkinter import *
from tkinter.colorchooser import askcolor
import dill


UNSELECTED_COLOR = '#f00'
SELECTED_COLOR = '#00f'


def sort_line(lines):
    lines = copy.deepcopy(lines)
    

    


class SDFGenerator(object):
    def __init__(self, lines):
        # lines: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        self.lines = lines

    def gen(self, filename=''):

        if not filename:
            filename = 'test.py'

        if not filename.endswith('.py'):
            filename = filename + '.py'

        with open(filename, 'w') as f:
            base_code = '''
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
'''
            f.write(base_code)

            for line in self.lines:
                f.write('    [[{:.6f}, {:.6f}], [{:.6f}, {:.6f}]],\n'.format(line[0], line[1], line[2], line[3]))

            f.write(']')

            base_code = '''

def sdTrack(p):
    d = [sdLine(p, line[0], line[1]) for line in TRACK_LINES]

    return opUnion(d)

'''
            f.write(base_code)

            f.write('''

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
''')


        print('Save SDF to: {}'.format(filename))

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            raise KeyError('Key out of bounds: {}'.format(key))

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        else:
            raise KeyError('Key out of bounds: {}'.format(key))

class PointHolder():
    def __init__(self, canvas):
        self.c = canvas
        self.id = 0
        self.point_ids = {} # Point to int(ID)
        self.adj_points = {} # int(ID) to list(int(ID))

        self.points = {} # ID to tkinter.oval
        self.connections = {} # (ID, ID) to tkinter.line

    def get_point_by_id(self, _id):
        return list(self.point_ids.keys())[list(self.point_ids.values()).index(_id)]

    def get_id_by_oval(self, oval):
        if isinstance(oval, (list, tuple)):
            _oval = None
            for o in oval:
                if self.c.type(o) == 'oval':
                    _oval = o
                    break
            oval = _oval

        if not oval:
            return None


        return list(self.points.keys())[list(self.points.values()).index(oval)]

    def get_id(self, p):
        return self.point_ids[p]

    def get_oval(self, p):
        if isinstance(p, int):
            return self.points[p]
        elif isinstance(p, Point):
            return self.points[self.point_ids[p]]

    def draw_point(self, p):
        '''
        p: ID or point
        '''

        if isinstance(p, int):
            ID = p
            p = self.get_point_by_id(ID)
        else:
            ID = self.point_ids[p]

        r = 5
        p = self.c.create_oval(int(p[0])-r, int(p[1])-r,
                               int(p[0])+r, int(p[1])+r, 
                               outline='#000', fill=UNSELECTED_COLOR, width=2)

        self.points[ID] = p

        return p

    def draw_line(self, p1, p2):
        '''
        p1: ID or point
        p2: ID or point
        '''

        if isinstance(p1, int):
            ID1 = p1
            p1 = self.get_point_by_id(ID1)
        else:
            ID1 = self.point_ids[p1]

        if isinstance(p2, int):
            ID2 = p2
            p2 = self.get_point_by_id(ID2)
        else:
            ID2 = self.point_ids[p2]

        line = self.c.create_line(p1[0], p1[1], p2[0], p2[1], width=3, fill='#060')


        ID1, ID2 = (ID1, ID2) if ID1 < ID2 else (ID2, ID1)

        self.connections[(ID1, ID2)] = line


        return line

    def add(self, p):
        '''
        p: Point
        '''

        self.id += 1
        self.point_ids[p] = self.id

        # draw oval
        self.draw_point(p)

    def move(self, p, new_pos):
        '''
        p: ID or Point
        '''
        # if input is ID
        if isinstance(p, int):
            ID = p
            p = self.get_point_by_id(ID)
        else:
            ID = self.point_ids[p]

        # move connections
        if ID in self.adj_points:
            for adj in self.adj_points[ID]:
                conn = self.connections[(adj, ID) if adj < ID else (ID, adj)]
                line = self.c.coords(conn)
                if line[0] == p.x and line[1] == p.y:
                    line[0] = new_pos[0]
                    line[1] = new_pos[1]
                else:
                    line[2] = new_pos[0]
                    line[3] = new_pos[1]

                self.c.coords(conn, line)

        # move oval
        oval = self.points[ID]
        self.c.move(oval, new_pos[0]-p[0], new_pos[1]-p[1])

        # update point 
        p[0], p[1] = new_pos[0], new_pos[1]

    def delete(self, p):
        '''
        p: ID or Point
        '''

        # if input is ID
        if isinstance(p, int):
            ID = p
            p = self.get_point_by_id(ID)
        else:
            ID = self.point_ids[p]

        # remove connections
        if ID in self.adj_points:
            for adj in self.adj_points[ID]:
                # remove ID in adjs
                self.adj_points[adj].remove(ID)

                # remove connections
                _hash = (adj, ID) if adj < ID else (ID, adj)
                conn = self.connections[_hash]
                self.c.delete(conn)
                del self.connections[_hash]

            # remove adjs
            del self.adj_points[ID]

        # remove oval
        self.c.delete(self.points[ID])
        del self.points[ID]

        # remove point
        del self.point_ids[p]

    def connect(self, p1, p2):
        '''
        p1: ID or Point
        p2: ID or Point
        '''

        # if input is ID
        if isinstance(p1, int):
            ID1 = p1
            p1 = self.get_point_by_id(ID1)
        else:
            ID1 = self.point_ids[p1]

        if isinstance(p2, int):
            ID2 = p2
            p2 = self.get_point_by_id(ID2)
        else:
            ID2 = self.point_ids[p2]

        if ID1 not in self.adj_points:
            self.adj_points[ID1] = []

        if ID2 not in self.adj_points:
            self.adj_points[ID2] = []

        if ID2 not in self.adj_points[ID1]:
            self.adj_points[ID1].append(ID2)
            self.adj_points[ID2].append(ID1)

            self.draw_line(ID1, ID2)

    def save(self, filename=None):
        if not filename:
            filename = 'savepoint.pkl'

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as f:
            dill.dump({
                'id': self.id,
                'point_ids': self.point_ids,
                'adj_points': self.adj_points
                }, f)

        print('Dump to file: {}'.format(filename))

    @classmethod
    def load(cls, canvas, filename=None):
        
        if not filename:
            filename = 'savepoint.pkl'

        print('Load from file: {}'.format(filename))

        with open(filename, 'rb') as f:
            field = dill.load(f)

        self = PointHolder(canvas)

        self.id = field['id']
        self.point_ids = field['point_ids']
        self.adj_points = field['adj_points']

        for p in self.point_ids.keys():
            self.draw_point(p)

        for ID in self.adj_points.keys():
            for adj in self.adj_points[ID]:
                if ID < adj:
                    self.draw_line(ID, adj)


        return self



class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'
    INVERSE_Y = True

    SELECTED_COLOR = '#00f'
    UNSELECTED_COLOR = '#f00'
    HOVERED_COLOR = '#ff0'


    def __init__(self, args):
        self.root = Tk()

        
        self.save_button = Button(self.root, text='Save', command=self.save_button_press)
        self.save_button.grid(row=0, column=0)

        self.load_button = Button(self.root, text='Load', command=self.load_button_press)
        self.load_button.grid(row=0, column=1)

        self.generate_button = Button(self.root, text='Generate', command=self.generate_button_press)
        self.generate_button.grid(row=0, column=2)

        
        self.gen_interp_button = Button(self.root, text='Gen Interp', command=self.gen_interp_button_press)
        self.gen_interp_button.grid(row=0, column=3)

        self.filename_entry = Entry(self.root)
        self.filename_entry.grid(row=0, column=4)

        
        self.interp_button = Button(self.root, text='Interpolate', command=self.interp_button_press)
        self.interp_button.grid(row=0, column=5)

        self.interp_entry = Entry(self.root)
        self.interp_entry.grid(row=0, column=6)



        self.add_button = Button(self.root, text='Add', command=self.add_button_press)
        self.add_button.grid(row=1, column=0)

        self.move_button = Button(self.root, text='Move', command=self.move_button_press)
        self.move_button.grid(row=1, column=1)

        self.delete_button = Button(self.root, text='Delete', command=self.delete_button_press)
        self.delete_button.grid(row=1, column=2)

        self.connect_button = Button(self.root, text='Connect', command=self.connect_button_press)
        self.connect_button.grid(row=1, column=3)

        self.reset_button = Button(self.root, text='Reset', command=self.reset)
        self.reset_button.grid(row=1, column=4)



        self.mode = 'Add'
        self.selected_point = None
        self.active_button = self.add_button
        self.activate_button(self.add_button)

        self.point_holder = None # init in reset()

        # load waypoints
        if args.npy:
            self.waypoints = np.load(args.npy)
            self.center = self.waypoints[:, 0:2]
            self.inner = self.waypoints[:, 2:4]
            self.outer = self.waypoints[:, 4:6]

            self.bounding_box = [( np.min(self.outer[:, 0]), np.min(self.outer[:, 1]) ),  # x, y
                                 ( np.max(self.outer[:, 0]) - np.min(self.outer[:, 0]),   # width
                                   np.max(self.outer[:, 1]) - np.min(self.outer[:, 1]) )] # height

            self.margin_px = 50
            self.width = args.width or 1280
            self.height = (args.width - 2*self.margin_px) * (self.bounding_box[1][1]/self.bounding_box[1][0]) + 2*self.margin_px

            
            self.center_point = np.array([self.bounding_box[0][0] + self.bounding_box[1][0]/2,
                                          self.bounding_box[0][1] + self.bounding_box[1][1]/2], dtype=np.float32)

            self.center_pixel = np.array([self.width/2.0, self.height/2.0], dtype=np.float32)

        else:
            self.waypoints = None
            self.width = args.width or 1280
            self.height = self.width/16*9

        self.c = Canvas(self.root, bg='white', width=self.width, height=self.height)
        self.c.grid(row=2, columnspan=5)

        self.reset()

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.active_button = self.add_button
        self.c.bind('<Button-1>', self.MLB_down)
        self.c.bind('<Button-3>', self.MRB_down)
        self.c.bind('<B1-Motion>', self.MLB_move)
        self.c.bind('<ButtonRelease-1>', self.MLB_up)

    def add_button_press(self):
        self.mode = 'Add'
        self.activate_button(self.add_button)

    def move_button_press(self):
        self.mode = 'Move'
        self.activate_button(self.move_button)

    def delete_button_press(self):
        self.mode = 'Delete'
        self.activate_button(self.delete_button)

    def connect_button_press(self):
        self.mode = 'Connect'
        self.activate_button(self.connect_button)

    def save_button_press(self):
        filename = self.filename_entry.get()

        self.point_holder.save(filename)

    def load_button_press(self):
        filename = self.filename_entry.get()

        self.point_holder = PointHolder.load(self.c, filename)

    def generate_button_press(self):

        lines = []
        for line in self.point_holder.connections.values():
            c = self.c.coords(line)

            a = self.pixel_to_point_transform(np.array([c[0], c[1]], dtype=np.float32))
            b = self.pixel_to_point_transform(np.array([c[2], c[3]], dtype=np.float32))

            lines.append([a[0], a[1], b[0], b[1]])

        gen = SDFGenerator(lines)

        gen.gen(self.filename_entry.get())


    def gen_interp_button_press(self):
        pass

    def interp_button_press(self):
        pass

    def activate_button(self, button):
        self.active_button.config(relief=RAISED)
        button.config(relief=SUNKEN)

        self.active_button = button

        self.unselect_point()


    def select_point(self, point):

        if not self.c.type(point) == 'oval':
            return

        if point not in self.point_holder.points.values():
            return

        self.unselect_point()

        self.selected_point = point
        self.c.itemconfig(self.selected_point, fill=self.SELECTED_COLOR)

    def unselect_point(self):
        if self.selected_point:
            self.c.itemconfig(self.selected_point, fill=self.UNSELECTED_COLOR)

            self.selected_point = None


    def filter_items(self, c, type='oval'):
        
        if c:
            if isinstance(c, (tuple, list)):
                _c = None
                for o in c:
                    if self.c.type(o) == type:
                        _c = o
                        break
                c = _c
        
        return c

    def MLB_down(self, event):

        if self.mode == 'Add':
            point = Point(event.x, event.y)
            self.point_holder.add(point)
        
            self.select_point(self.point_holder.get_oval(point))

        if self.mode == 'Delete':
            oval = self.filter_items(self.c.find_withtag(CURRENT))
            if oval:
                self.unselect_point()
                self.point_holder.delete(self.point_holder.get_id_by_oval(oval))

        if self.mode == 'Move':
            oval = self.filter_items(self.c.find_withtag(CURRENT))
            if oval:
                self.select_point(oval)

        if self.mode == 'Connect':
            oval = self.filter_items(self.c.find_withtag(CURRENT))

            if self.selected_point is None and oval:
                self.select_point(oval)

            elif oval:
                p1 = self.point_holder.get_id_by_oval(self.selected_point)
                p2 = self.point_holder.get_id_by_oval(oval)
                self.point_holder.connect(p1, p2)

                self.select_point(oval)

    def MRB_down(self, event):
        self.unselect_point()

    def MLB_move(self, event):

        if self.mode == 'Move':
            if self.selected_point is not None:
                ID = self.point_holder.get_id_by_oval(self.selected_point)

                self.point_holder.move(ID, (event.x, event.y))

    def MLB_up(self, event):
        pass

    def point_to_pixel_transform(self, point):
        '''
        Transform npy coordinate to canvas coordinate
        '''
        vec = point - self.center_point

        if self.INVERSE_Y:
            vec[1] = -vec[1]

        pixel = self.center_pixel + vec * (self.width - 2*self.margin_px)/self.bounding_box[1][0]

        return pixel


    def pixel_to_point_transform(self, pixel):
        '''
        Transform canvas coordinate to npy coordinate
        '''

        vec = (pixel - self.center_pixel) * (self.bounding_box[1][0]/(self.width - 2*self.margin_px))

        if self.INVERSE_Y:
            vec[1] = -vec[1]

        point = self.center_point + vec
        return point

    def draw_point(self, point, color='#aaaaaa', r=3):

        p = self.c.create_oval(int(point[0])-r, int(point[1])-r,
                               int(point[0])+r, int(point[1])+r, outline='#000', fill=color, width=2)
    
        return p

    def draw_line(self, p1, p2, color='#060', width=3):

        l = self.c.create_line(p1[0], p1[1], p2[0], p2[1], width=width, fill=color)

        return l


    def draw_waypoints(self):

        for point in self.center:
            pixel = self.point_to_pixel_transform(point)

            self.draw_point(pixel)

        for point in self.outer:
            pixel = self.point_to_pixel_transform(point)

            self.draw_point(pixel)

        for point in self.inner:
            pixel = self.point_to_pixel_transform(point)

            self.draw_point(pixel)

    def reset(self):
        self.c.delete('all')

        if self.waypoints is not None:
            self.draw_waypoints()
 
        self.point_holder = PointHolder(self.c)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepracer SDF painter')
    parser.add_argument('--npy', type=str, default='')
    parser.add_argument('--width', type=int, default=1280)

    args = parser.parse_args()

    Paint(args)
