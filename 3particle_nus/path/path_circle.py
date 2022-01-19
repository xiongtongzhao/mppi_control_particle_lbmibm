import torch
import math

R1=0.4
R2=0.5
R3=0.6
SO1=2*math.pi*R1
SO2=2*math.pi*R2
SO3=2*math.pi*R3

def char_O(state,R):
    f=abs(state[:,0]**2+state[:,1]**2-R**2)
    return f


def tang_O(state,R):
    x=-(state[:,1]/R).view(-1,1)
    y=(state[:,0]/R).view(-1,1)   
    return  torch.cat((x,y),dim=1)





