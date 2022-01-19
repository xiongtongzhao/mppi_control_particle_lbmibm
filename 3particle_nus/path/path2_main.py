import torch
import math

R=0.2
L=0.15
SN=4*R+2*math.sqrt(2)*R
SU=2*R+math.pi*R
SS=1.5*(math.pi+2)*R

def char_N1(state):
    f=abs(state[:,0]+L+3*R)
    return f

def char_N2(state):
    f=abs(state[:,0]+state[:,1]+2*R+L)
    return f

def char_N3(state):
    f=abs(state[:,0]+L+R)
    return f

def tang_N1(state):
    s=state.shape[0]
    x=(torch.zeros(s)).view(-1,1)    
    y=(torch.ones(s)*SN).view(-1,1)    
    return  torch.cat((x,y),dim=1)

def tang_N2(state):
    s=state.shape[0]
    x=(torch.ones(s)*SN/math.sqrt(2)).view(-1,1)
    y=-x
    return  torch.cat((x,y),dim=1)

def tang_N3(state):
    s=state.shape[0]
    x=(torch.zeros(s)).view(-1,1)     
    y=(torch.ones(s)*SN).view(-1,1)    
    return  torch.cat((x,y),dim=1)

def char_U1(state):
    f=abs(state[:,0]+R)
    return f

def char_U2(state):
    f=abs(state[:,0]**2+state[:,1]**2-R**2)
    return f

def char_U3(state):
    f=abs(state[:,0]-R)
    return f

def tang_U1(state):
    s=state.shape[0]
    x=(torch.zeros(s)).view(-1,1)
    y=-(torch.ones(s)*SU).view(-1,1)    
    return torch.cat((x,y),dim=1)

def tang_U2(state):
    x=-(state[:,1]*SU/R).view(-1,1)
    y=(state[:,0]*SU/R).view(-1,1)
    return torch.cat((x,y),dim=1)

def tang_U3(state):
    s=state.shape[0]
    x=(torch.zeros(s)).view(-1,1)    
    y=(torch.ones(s)*SU).view(-1,1)    
    return torch.cat((x,y),dim=1)

def char_S1(state):
    f=abs((state[:,0]-2.0*R-L)**2+((state[:,1]-0.5*R)**2)*4-R**2)
    return f

def char_S2(state):
    f=abs((state[:,0]-2.0*R-L)**2+((state[:,1]-0.5*R)**2)*4-R**2)
    return f

def char_S3(state):
    f=abs((state[:,0]-2.0*R-L)**2+((state[:,1]+0.5*R)**2)*4-R**2)
    return f

def tang_S1(state):
    x=-4*((state[:,1]-0.5*R)).view(-1,1)
    y=((state[:,0]-2.0*R-L)).view(-1,1)
    tang=torch.cat((x,y),dim=1)
    tang=SS*tang/(tang.norm(dim=1)).view(-1,1)
    return tang

def tang_S2(state):
    x=-4*((state[:,1]-0.5*R)).view(-1,1)
    y=((state[:,0]-2.0*R-L)).view(-1,1)
    tang=torch.cat((x,y),dim=1)
    tang=SS*tang/(tang.norm(dim=1)).view(-1,1)  
    return tang

def tang_S3(state):
    x=4*((state[:,1]+0.5*R)).view(-1,1)
    y=-((state[:,0]-2.0*R-L)).view(-1,1)
    tang=torch.cat((x,y),dim=1)
    tang=SS*tang/(tang.norm(dim=1)).view(-1,1)     
    return tang





