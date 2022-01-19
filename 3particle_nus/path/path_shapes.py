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

def char_triangle(state,L,n,m):
    if n==1:
        if m==0:
            f=abs(math.sqrt(3)*state[:,0]+state[:,1]-L/math.sqrt(3))
        if m==1:
            f=abs(math.sqrt(3)*state[:,0]-state[:,1]+L/math.sqrt(3))
        if m==2:
            f=abs(state[:,1]+0.5*L/math.sqrt(3))            
    if n==2:
        if m==2:
            f=abs(math.sqrt(3)*state[:,0]+state[:,1]-L/math.sqrt(3))
        if m==0:
            f=abs(math.sqrt(3)*state[:,0]-state[:,1]+L/math.sqrt(3))
        if m==1:
            f=abs(state[:,1]+0.5*L/math.sqrt(3))        

    if n==3:
        if m==1:
            f=abs(math.sqrt(3)*state[:,0]+state[:,1]-L/math.sqrt(3))
        if m==2:
            f=abs(math.sqrt(3)*state[:,0]-state[:,1]+L/math.sqrt(3))
        if m==0:
            f=abs(state[:,1]+0.5*L/math.sqrt(3))        

    return f

def tang_triangle(state,L,n,m):
    s=state.shape[0]
    if n==1:
        if m==0:
            x=(-torch.ones((s,1))).view(-1,1)
            y=(math.sqrt(3)*torch.ones((s,1))).view(-1,1)
        if m==1:
            x=(-torch.ones((s,1))).view(-1,1)
            y=(-math.sqrt(3)*torch.ones((s,1))).view(-1,1)
        if m==2:
            x=torch.ones((s,1)).view(-1,1)
            y=torch.zeros((s,1)).view(-1,1)
    
    if n==2:
        if m==2:
            x=(-torch.ones((s,1))).view(-1,1)
            y=(math.sqrt(3)*torch.ones((s,1))).view(-1,1)
        if m==0:
            x=(-torch.ones((s,1))).view(-1,1)
            y=(-math.sqrt(3)*torch.ones((s,1))).view(-1,1)
        if m==1:
            x=torch.ones((s,1)).view(-1,1)
            y=torch.zeros((s,1)).view(-1,1)    
    
    if n==3:
        if m==1:
            x=(-torch.ones((s,1))).view(-1,1)
            y=(math.sqrt(3)*torch.ones((s,1))).view(-1,1)
        if m==2:
            x=(-torch.ones((s,1))).view(-1,1)
            y=(-math.sqrt(3)*torch.ones((s,1))).view(-1,1)
        if m==0:
            x=torch.ones((s,1)).view(-1,1)
            y=torch.zeros((s,1)).view(-1,1)    
    
    return  torch.cat((x,y),dim=1)    
    
def char_square(state,L,n,m):
    if n==1:
        if m==0:
            f=abs(state[:,0]-0.5*L)
        if m==1:
            f=abs(state[:,1]-0.5*L)
        if m==2:
            f=abs(state[:,0]+0.5*L)
        if m==3:
            f=abs(state[:,1]+0.5*L)
            
    if n==2:
        if m==3:
            f=abs(state[:,0]-0.5*L)
        if m==0:
            f=abs(state[:,1]-0.5*L)
        if m==1:
            f=abs(state[:,0]+0.5*L)
        if m==2:
            f=abs(state[:,1]+0.5*L)
    
    if n==3:
        if m==2:
            f=abs(state[:,0]-0.5*L)
        if m==3:
            f=abs(state[:,1]-0.5*L)
        if m==0:
            f=abs(state[:,0]+0.5*L)
        if m==1:
            f=abs(state[:,1]+0.5*L)
            
    if n==4:
        if m==1:
            f=abs(state[:,0]-0.5*L)
        if m==2:
            f=abs(state[:,1]-0.5*L)
        if m==3:
            f=abs(state[:,0]+0.5*L)
        if m==0:
            f=abs(state[:,1]+0.5*L)
            
    return f
        
def tang_square(state,L,n,m):
    s=state.shape[0]
    if n==1:
        if m==0:
            x=torch.zeros((s,1)).view(-1,1)
            y=torch.ones((s,1)).view(-1,1)
        if m==1:
            x=-torch.ones((s,1)).view(-1,1)
            y=torch.zeros((s,1)).view(-1,1)
        if m==2:
            x=torch.zeros((s,1)).view(-1,1)
            y=-torch.ones((s,1)).view(-1,1)
        if m==3:
            x=torch.ones((s,1)).view(-1,1)
            y=torch.zeros((s,1)).view(-1,1)
    
    if n==2:
        if m==3:
            x=torch.zeros((s,1)).view(-1,1)
            y=torch.ones((s,1)).view(-1,1)
        if m==0:
            x=-torch.ones((s,1)).view(-1,1)
            y=torch.zeros((s,1)).view(-1,1)
        if m==1:
            x=torch.zeros((s,1)).view(-1,1)
            y=-torch.ones((s,1)).view(-1,1)
        if m==2:
            x=torch.ones((s,1)).view(-1,1)
            y=torch.zeros((s,1)).view(-1,1)    
        
    if n==3:
        if m==2:
            x=torch.zeros((s,1)).view(-1,1)
            y=torch.ones((s,1)).view(-1,1)
        if m==3:
            x=-torch.ones((s,1)).view(-1,1)
            y=torch.zeros((s,1)).view(-1,1)
        if m==0:
            x=torch.zeros((s,1)).view(-1,1)
            y=-torch.ones((s,1)).view(-1,1)
        if m==1:
            x=torch.ones((s,1)).view(-1,1)
            y=torch.zeros((s,1)).view(-1,1)
            
    if n==4:
        if m==1:
            x=torch.zeros((s,1)).view(-1,1)
            y=torch.ones((s,1)).view(-1,1)
        if m==2:
            x=-torch.ones((s,1)).view(-1,1)
            y=torch.zeros((s,1)).view(-1,1)
        if m==3:
            x=torch.zeros((s,1)).view(-1,1)
            y=-torch.ones((s,1)).view(-1,1)
        if m==0:
            x=torch.ones((s,1)).view(-1,1)
            y=torch.zeros((s,1)).view(-1,1)        
        
    return torch.cat((x,y),dim=1)    
        
        
        
        
        
        
        
        
        
        
        
        
        