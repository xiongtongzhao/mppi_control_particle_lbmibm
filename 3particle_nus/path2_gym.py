import torch
import math
import numpy as np
from numpy import linalg as LA
R=0.2
L=0.15
SN=4*R+2*math.sqrt(2)*R
SU=2*R+math.pi*R
SS=1.5*(math.pi+2)*R
def solve(semi_major, semi_minor, p):
    px = abs(p[0])
    py = abs(p[1])

    t = math.pi / 4

    a = semi_major
    b = semi_minor

    for x in range(0, 3):
        x = a * math.cos(t)
        y = b * math.sin(t)

        ex = (a*a - b*b) * math.cos(t)**3 / a
        ey = (b*b - a*a) * math.sin(t)**3 / b

        rx = x - ex
        ry = y - ey

        qx = px - ex
        qy = py - ey

        r = math.hypot(ry, rx)
        q = math.hypot(qy, qx)

        delta_c = r * math.asin((rx*qy - ry*qx)/(r*q))
        delta_t = delta_c / math.sqrt(a*a + b*b - x*x - y*y)

        t += delta_t
        t = min(math.pi/2, max(0, t))

    return [math.copysign(x, p[0]), math.copysign(y, p[1])]

def char_N1(state):
    f=abs(state[0]+L+3*R)
    return f

def closest_N1(state):
    x=-(L+3*R)
    y=state[1]
    return np.array([x,y])    
    
def char_N2(state):
    f=abs(state[0]+state[1]+2*R+L)
    return f

def closest_N2(state):
    y=(state[1]-state[0]-2*R-L)/2.0
    x=(-state[1]+state[0]-2*R-L)/2.0
    return np.array([x,y]) 
    
def char_N3(state):
    f=abs(state[0]+L+R)
    return f

def closest_N3(state):
    x=-(L+R)
    y=state[1]
    return np.array([x,y]) 

def tang_N1(state):
    x=0.0   
    y=SN   
    return  np.array([x,y])

def tang_N2(state):
    x=SN/math.sqrt(2)
    y=-x
    return np.array([x,y])

def tang_N3(state):
    x=0.0    
    y=SN    
    return np.array([x,y])

def char_U1(state):
    f=abs(state[0]+R)
    return f

def closest_U1(state):
    x=-R
    y=state[1]
    return np.array([x,y])

def char_U2(state):
    f=abs(state[0]**2+state[1]**2-R**2)
    return f

def closest_U2(state):
    center=[0.0,0.0]
    [x,y]=center+R*(state-center)/LA.norm(state-center)
    return np.array([x,y])

def char_U3(state):
    f=abs(state[0]-R)
    return f

def closest_U3(state):
    x=R
    y=state[1]
    return np.array([x,y])

def tang_U1(state):
    x=0
    y=-SU    
    return np.array([x,y])

def tang_U2(state):
    x=-(state[1])*SU/R
    y=(state[0]*SU/R)
    return np.array([x,y])

def tang_U3(state):

    x=0    
    y=SU   
    return np.array([x,y])

def char_S1(state):
    f=abs((state[0]-2.0*R-L)**2+((state[1]-0.5*R)**2)*4-R**2)    
    return f

def closest_S1(state):
    return np.array(solve(R,0.5*R,[state[0]-2.0*R-L,state[1]-0.5*R]))+[2.0*R+L,0.5*R]

def char_S2(state):
    f=abs((state[0]-2.0*R-L)**2+((state[1]-0.5*R)**2)*4-R**2)
    return f

def closest_S2(state):
    return np.array(solve(R,0.5*R,[state[0]-2.0*R-L,state[1]-0.5*R]))+[2.0*R+L,0.5*R]  
    
def char_S3(state):
    f=abs((state[0]-2.0*R-L)**2+((state[1]+0.5*R)**2)*4-R**2) 
    return f

def closest_S3(state):
    return np.array(solve(R,0.5*R,[state[0]-2.0*R-L,state[1]+0.5*R]))+ [2.0*R+L,-0.5*R]  
    
def tang_S1(state):
    x=-4*((state[1]-0.5*R))
    y=((state[0]-2.0*R-L))
    tang=np.array([x,y])
    tang=tang*SS/LA.norm(tang)
    return tang

def tang_S2(state):
    x=-4*((state[1]-0.5*R))
    y=((state[0]-2.0*R-L))
    tang=np.array([x,y])
    tang=tang*SS/LA.norm(tang)    
    return tang

def tang_S3(state):
    x=4*((state[1]+0.5*R))
    y=-((state[0]-2.0*R-L))
    tang=np.array([x,y])
    tang=tang*SS/LA.norm(tang)    
    return tang





