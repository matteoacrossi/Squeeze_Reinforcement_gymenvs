import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random as rand
import numpy as np
from scipy.linalg import sqrtm
import math
import sympy as sym

#definizione dei parametri (ributto qui tutti i conti delle matrici che servono per il calcolo)
k=1 #loss rate
eta=1 #eta della misura
              
#setto sigma ambiente Sb e sigma misura Sm
Sb=np.identity(2)
z=1e300 #qui abbozzo il lim per z a infinito della sigma di squeezing
Sm=np.array([[1/z,0],[0,z]])
              
#canale rumoroso per la misura con efficenza eta
Xs=np.identity(2)*(eta**(-0.5))
Ys=(eta**(-1)-1)*np.identity(2)
Sm=Xs.dot(Sm.dot(Xs.T))+Ys
              
#matrice simplettica
Sy=np.array([[0,1],[-1,0]])
              
#imposto i blocchi per la matrice Hc anche se mi sa non serve
zero=np.zeros([2,2])
C=(k**0.5)*Sy.T
                    
#matrice di drift
D=Sy.dot(C.dot(Sb.dot(C.T.dot(Sy.T))))
              
#imposto la matrice 1/(sigma ambiente+sigma misura)^0.5
SIGMA=sqrtm(np.linalg.inv(Sb+Sm))
              
#imposto le matrici di dinamica monitorata (ho lasciato sotto commento il caso perfetto, ovvero quello dove il limite per z a infinito è preso esatto)
B=C.dot(Sy.dot(SIGMA))#np.array([ [-((eta*k)**0.5),0],[0,0] ] ) #
E=Sy.dot(C.dot(Sb.dot(SIGMA)))


class SqueezeEnvq(gym.Env):
      
      metadata = {'render.modes': ['human']} #non so bene a che serva ma per ora lo tengo
      #provo a definire degli attributi che dovrebbero essere le cose che vanno tenute in memoria in esecuzione
      
      A=np.zeros((2,2))
      Hs=np.zeros((2,2))
      dt=0
      Y=np.zeros((2,2))
      P=np.array([[1,0],[0,0]])
      K=np.zeros((2,2))
      sigmacss=np.zeros((2,2))
      eps=5e-2
    
      #queste servono per il feedback
      l=1 
      F=np.array([[l,0],[0,l]])
      Q=np.zeros((2,2))
      Qinv=np.zeros((2,2))
    
      #queste sono le inizializzazioni della dinamica per lo step
      r=np.ones(2)
      sc=np.identity(2)
      current_reward=0
      Done=False
        
      #parametri di impostazione
      monkey=False
      plot=False
      eval=False
      
      #bozza per risoluzione simbolica Riccati stationary eq da separare in altro file per controllo, o da mettere nel file che fa i plot
      #Fv = sym.Matrix(F)
      #Yv = sym.Matrix(sym.MatrixSymbol('Yv', 2, 2))
      #Qv = sym.Matrix(Qinv)
      #Pv = sym.Matrix(P)
      #Av = sym.Matrix(A)
      #Yv=sym.solve(Av.T*Yv+Yv*Av+Pv-Yv*Fv*Qv*Fv.T*Yv, Yv)
      #print(Yv[3])
      #Yv=np.array(list(Yv[3]))
      #Yv=np.array( [ [Yv[0],Yv[1]],[Yv[2],Yv[3]] ])
      #Yv=Yv.astype(np.float)

      def __init__(self,q=1e-4,X=0.499*k,dt=1e-4,monkey=False,plot=False,eval=False,randq=False):
              
              super(SqueezeEnvq, self).__init__() 
              
              #setto le impostazioni
              self.randq=randq
              self.monkey=monkey  
              self.dt=dt
              self.plot=plot
              self.eval=eval
              a=(k-2*X)/k
              b=k/(k-2*X)
              self.sigmacss=np.array([[a,0],[0,b]])
              self.eps=5e-2
              self.time=0
            
              #setto il passo di integrazione
              self.dt=dt
        
              #setto la matrice Q di costo del feedback e la sua inversa
              self.Q=q*np.identity(2)
              self.Qinv=np.linalg.inv(self.Q)
        
              #matrice dell'hamiltoniana di Squeezing
              self.Hs=-X*np.array([[0,1],[1,0]])
                
              #matrice A dell'evoluzione
              self.A=Sy.dot(self.Hs)+0.5*Sy.dot(C.dot(Sy.dot(C.T)))  
            
              if self.monkey==True:
                self.action_space = spaces.Box( 
                   low=-np.inf , high=np.inf ,shape=(1,), dtype=np.float32)
              
              if self.monkey==False:
                self.action_space = spaces.Box( 
                   low=-np.inf , high=np.inf ,shape=(4,), dtype=np.float32)
                            
              self.observation_space = spaces.Box(
                   low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
              
              
      
      def step(self, action):
              
              eps=self.eps
              sigmacss=self.sigmacss
              plot=self.plot
              monkey=self.monkey
              eval=self.eval
              Q=self.Q
              dt=self.dt 
              A=self.A
              F=self.F
              time=self.time
              r=self.r
              sc=self.sc
              
              #estraggo l'incremento dw
              dw=np.random.randn(2)*(dt**0.5)
              
              #momento primo con feedback, dall'azione scelta faccio la matrice K
              if self.monkey==True:                
                K=np.array([[action[0],0],[0,0]])
              
              if self.monkey==False:
                action=np.array([[action[0],action[1]],[action[2],action[3]]])
                K=action  
                
              #definisco u e aggiorno il momento primo
              u=-K.dot(r)
              delta=K-self.K
              self.K=K
              dr=A.dot(r)*dt+(2**(-0.5))*(E-sc.dot(B)).dot(dw)+F.dot(u)*dt
              r=r+dr
              
              #aggiorno la matrice di covarianza
              sc=sc+dt*((A.dot(sc)+sc.dot(A.T)+D)-(E-sc.dot(B)).dot((E-sc.dot(B)).T))
              
              #funzione costo
              h=0.5*sc[0,0]+r[0]**2 + u.T.dot(Q.dot(u))
                
              costoQ= u.T.dot(Q.dot(u))
              costoP=h-costoQ
                
              if abs(np.linalg.norm(delta))<=1 or eval==True or time<=1:
                costoagg=0
              else:
                costoagg=0.1*np.linalg.norm(delta)
              
              if plot==False:
              
                if h>=10:
                 print(h)
                 h=(4e4-time)*10
                 print(h)
                 self.current_reward=-h
                 self.Done=True
                
                else:
                 self.current_reward=-h-costoagg
               
              if plot==True:
                 self.current_reward=-h
                 
              
              #provo a dare un criterio per smettere dopo un po' che siamo abbastanza vicini allo steady state
              distance=(sc[0,0]-sigmacss[0,0])**2
              
              if distance<=eps:
                 self.Done=True
                
              #alla fine salvo il momento primo e la matrice di covarianza e le metto nell'output
              time+=1
              self.time=time
              self.r=r
              self.sc=sc
              output=[r[0],r[1],sc[0,0],sc[0,1],sc[1,0],sc[1,1]]
              output=np.array(output)
              return output , self.current_reward , self.Done ,{'costoP': costoP, 'costoQ':costoQ,'K(t)':K,'incr':delta,'rewagg':-h-costoagg}
    
      

      def reset(self):
              
              self.time=0
              plot=self.plot
              randq=self.randq
              if randq==True:
                    q=np.random.uniform(1e-4,1e-2)
                    #setto la matrice Q di costo del feedback e la sua inversa
                    self.Q=q*np.identity(2)
                    self.Qinv=np.linalg.inv(self.Q)
              #reinizializzo delle cose
              if plot==True:
                a=0.5
                d=0.5
                n=2
                
              if plot==False:
                a=np.random.uniform(-1,1)
                d=np.random.uniform(-1,1)
                n=np.random.uniform(0,10)
             
              #setto i momenti primi iniziali
              self.r=np.array([a,d])
              
              #setto le covarianze iniziali e i 'Done'
              self.sc=(2*n+1)*np.array([[1,0],[0,1]])
              self.Done=False
              
              r=self.r
              sc=self.sc
              
              #resetto le reward
              self.current_reward=0
              self.K=np.zeros((2,2))
              
              #resetto a zero la matrice Y per il feedback e preparo l'output
              self.Y=0*np.identity(2)
              
              output=[r[0], r[1], sc[0,0], sc[0,1], sc[1,0], sc[1,1]]
              output=np.array(output)
              return output
          
      
      
      def render(self,mode='human'):
              print(self.momento_primo,self.current_reward)
    

