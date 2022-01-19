import torch
import math

R=0.25
RU=1.0*R
SN=2.0*R+R*math.sqrt(2)
SU=(1.0*RU*math.sqrt(3)+RU*math.pi/3.0)
SS=(R*math.sqrt(5.0-2.0*math.sqrt(3))+R*2.0*math.pi/3.0)
SSL=(math.sqrt(5.0-2.0*math.sqrt(3)))
def func_circle(state):
    r=0.5
    f=state.norm(dim=1)**2-r**2
    return f

def tang_circle(state):
    x=-state[:,1].view(-1,1)
    y=state[:,0].view(-1,1)
    return torch.cat((x,y),dim=1)

def func_line(state):
        
    f=state[:,0]-state[:,1]
    return f

def tang_line(state):
    s=state.shape[0]
    x=torch.ones(s).view(-1,1)
    y=x
    return torch.cat((x,y),dim=1)

def func_doubleline(state,X_ini,X_tar):
    f=abs((state[:,0]-X_ini[:,0])*(state[:,1]-X_tar[:,1])-(state[:,0]-X_tar[:,0])*(state[:,1]-X_ini[:,1]))+abs((state[:,2]-X_ini[:,2])*(state[:,3]-X_tar[:,3])-(state[:,2]-X_tar[:,2])*(state[:,3]-X_ini[:,3]))
    return f

def tang_doubleline(state,X_ini,X_tar):
    return X_tar-state

def pathlength(X_ini,X_tar):
    d=X_tar-X_ini
    
    return torch.cat((d[:,:2].norm(dim=1).view(-1,1),d[:,2:].norm(dim=1).view(-1,1)),dim=1)


def func_para_line2(state):
    f=abs(state[:,0]+0.4)+abs(state[:,2]+state[:,3])
    return f

def tang_para_line2(state,reward1,reward2):
    s=state.shape[0]
    x1=(torch.zeros(s)*(0.7-reward1)).view(-1,1)
    y1=(torch.ones(s)*(0.7-reward1)).view(-1,1)
    x2=-(torch.ones(s)*(0.5*math.sqrt(2))*(0.5*math.sqrt(2)-reward2)).view(-1,1) 
    y2=(torch.ones(s)*(0.5*math.sqrt(2))*(0.5*math.sqrt(2)-reward2)).view(-1,1)
    return torch.cat((x1,y1,x2,y2),dim=1)

def func_para_circle2(state):
    f=abs(state[:,0]**2+state[:,1]**2-0.49)+abs(state[:,2]**2+state[:,3]**2-0.49)
    return f

def tang_para_circle2(state,reward1,reward2):
    x1=-(state[:,1]).view(-1,1)
    y1=(state[:,0]).view(-1,1)
    x2=-(state[:,3]).view(-1,1)
    y2=(state[:,2]).view(-1,1)     
    
    '''
    x1=-(state[:,1]*(0.7*math.pi-reward1)).view(-1,1)
    y1=(state[:,0]*(0.7*math.pi-reward1)).view(-1,1)
    x2=-(state[:,3]*(0.7*math.pi-reward2)).view(-1,1)
    y2=(state[:,2]*(0.7*math.pi-reward2)).view(-1,1)
    '''
    return  torch.cat((x1,y1,x2,y2),dim=1)  

def char_N1(state):
    f=abs(state[:,0]+(R/2.0+2*R))
    return f

def char_N2(state):
    f=abs(state[:,0]+state[:,1]+(R/2.0+2*R)-1.5*R)
    return f

def char_N3(state):
    f=abs(state[:,0]+(R/2.0+R))
    return f

def tang_N1(state):
    s=state.shape[0]
    x=(torch.zeros(s)).view(-1,1)    
#    x=-((state[:,0]+R/2.0+2*R)*SN).view(-1,1)
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
#    x=-((state[:,0]+R/2.0+R)*SN).view(-1,1)
    y=(torch.ones(s)*SN).view(-1,1)    
    return  torch.cat((x,y),dim=1)

def char_U1(state):
    f=abs(state[:,0]+RU/2.0)
    return f

def char_U2(state):
    f=abs(state[:,0]**2+(state[:,1]-1.5*R)**2-RU**2)
    return f

def char_U3(state):
    f=abs(state[:,0]-RU/2.0)
    return f

def tang_U1(state):
    s=state.shape[0]
#    x=-((state[:,0]+R/2.0)*SU).view(-1,1)
    x=(torch.zeros(s)).view(-1,1)
    y=-(torch.ones(s)*SU).view(-1,1)    
    return torch.cat((x,y),dim=1)

def tang_U2(state):
    #r=state.norm(dim=1)
    x=-((state[:,1]-1.5*R)*SU/RU).view(-1,1)
    y=(state[:,0]*SU/RU).view(-1,1)
    return torch.cat((x,y),dim=1)

def tang_U3(state):
    s=state.shape[0]
    x=(torch.zeros(s)).view(-1,1)    
#    x=-((state[:,0]-R/2.0)*SU).view(-1,1)
    y=(torch.ones(s)*SU).view(-1,1)    
    return torch.cat((x,y),dim=1)

def char_S1(state):
    f=abs((state[:,0]-2.0*R)**2+(state[:,1]-0.5*R)**2-R**2)
    return f

def char_S2(state):
    f=abs((math.sqrt(3)-1)*state[:,0]+state[:,1]-1.5*R+(2.5-2.0*math.sqrt(3))*R)
    return f

def char_S3(state):
    f=abs((state[:,0]-2.0*R)**2+(state[:,1]-1.5*R)**2-R**2)
    return f

def tang_S1(state):
    x=-((state[:,1]-0.5*R)*SS/R).view(-1,1)
    y=((state[:,0]-2.0*R)*SS/R).view(-1,1)
    return torch.cat((x,y),dim=1)

def tang_S2(state):
    s=state.shape[0]
    x=(torch.ones(s)*SS/SSL).view(-1,1)
    y=(torch.ones(s)*(1.0-math.sqrt(3))*SS/SSL).view(-1,1)
    return  torch.cat((x,y),dim=1)

def tang_S3(state):
    x=((state[:,1]-1.5*R)*SS/R).view(-1,1)
    y=-((state[:,0]-2.0*R)*SS/R).view(-1,1)
    return torch.cat((x,y),dim=1)   
'''
def ratio_N1(state):
    y=state[1]
    ratio=(y-0.5*R)/SN
    return ratio

def ratio_N2(state):
    y=state[]
    
'''   
    
def char_NN1(state):
    f=abs(state[:,0]+(R/2.0))
    return f

def char_NN2(state):
    f=abs(state[:,0]+state[:,1]+R/2.0-1.5*R)
    return f

def char_NN3(state):
    f=abs(state[:,0]+(R/2.0-R))
    return f

def tang_NN1(state):
    s=state.shape[0]
    x=(torch.zeros(s)).view(-1,1)    
#    x=-((state[:,0]+R/2.0+2*R)*SN).view(-1,1)
    y=(torch.ones(s)*SN).view(-1,1)    
    return  torch.cat((x,y),dim=1)
    
def tang_NN2(state):
    s=state.shape[0]
    x=(torch.ones(s)*SN/math.sqrt(2)).view(-1,1)
    y=-x
    return  torch.cat((x,y),dim=1)

def tang_NN3(state):
    s=state.shape[0]
    x=(torch.zeros(s)).view(-1,1)     
#    x=-((state[:,0]+R/2.0+R)*SN).view(-1,1)
    y=(torch.ones(s)*SN).view(-1,1)    
    return  torch.cat((x,y),dim=1)    
    
def char_NNN1(state):
    f=abs(state[:,0]-(1.5*R))
    return f

def char_NNN2(state):
    f=abs(state[:,0]+state[:,1]-1.5*R-1.5*R)
    return f

def char_NNN3(state):
    f=abs(state[:,0]-2.5*R)
    return f

def tang_NNN1(state):
    s=state.shape[0]
    x=(torch.zeros(s)).view(-1,1)    
#    x=-((state[:,0]+R/2.0+2*R)*SN).view(-1,1)
    y=(torch.ones(s)*SN).view(-1,1)    
    return  torch.cat((x,y),dim=1)
    
def tang_NNN2(state):
    s=state.shape[0]
    x=(torch.ones(s)*SN/math.sqrt(2)).view(-1,1)
    y=-x
    return  torch.cat((x,y),dim=1)

def tang_NNN3(state):
    s=state.shape[0]
    x=(torch.zeros(s)).view(-1,1)     
#    x=-((state[:,0]+R/2.0+R)*SN).view(-1,1)
    y=(torch.ones(s)*SN).view(-1,1)    
    return  torch.cat((x,y),dim=1)     
    
    
    
    
    
    
    
    


