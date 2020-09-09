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
X=k*0.499 #accoppiamento hamiltoniana del sistema qui l'ho impostato sul valore 'critico'
              
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
              
#matrice dell'hamiltoniana di Squeezing
Hs=-X*np.array([[0,1],[1,0]]) 
              
#imposto i blocchi per la matrice Hc anche se mi sa non serve
zero=np.zeros([2,2])
C=(k**0.5)*Sy.T
              
              
#matrice A dell'evoluzione
A=Sy.dot(Hs)+0.5*Sy.dot(C.dot(Sy.dot(C.T)))
                    
#matrice di drift
D=Sy.dot(C.dot(Sb.dot(C.T.dot(Sy.T))))
              
#imposto la matrice 1/(sigma ambiente+sigma misura)^0.5
SIGMA=sqrtm(np.linalg.inv(Sb+Sm))
              
#imposto le matrici di dinamica monitorata (ho lasciato sotto commento il caso perfetto, ovvero quello dove il limite per z a infinito è preso esatto)
B=C.dot(Sy.dot(SIGMA))#np.array([ [-((eta*k)**0.5),0],[0,0] ] ) #
E=Sy.dot(C.dot(Sb.dot(SIGMA)))#B#
      
#queste servono per il feedback
l=1 
F=np.array([[l,0],[0,l]])
q=1e-4
Q=q*np.identity(2)
Qinv=np.linalg.inv(Q)
             
#imposto la sigmac steady state, per definire uno stop
a=(k-2*X)/k
b=k/(k-2*X)
sigmacss=np.array([[a,0],[0,b]])
eps=5e-2 #serve tipo definizione di limite per dire quando siamo in steady state


class SqueezeEnv(gym.Env):
      
      metadata = {'render.modes': ['human']} #non so bene a che serva ma per ora lo tengo
      #provo a definire degli attributi che dovrebbero essere le cose che vanno tenute in memoria in esecuzione
      momento_primo=np.ones(2)
      momento_primo_2=np.ones(2)
      matrice_covarianza=np.identity(2)
      matrice_covarianza_2=np.identity(2)
      current_reward=0
      current_reward_2=0
      Done=False
      Done_2=False
      dt=1e-4
      Y=0*np.identity(2)
      P=np.array([[1,0],[0,0]])
      
      #bozza per risoluzione simbolica Riccati stationary eq
      Fv = sym.Matrix(F)
      Yv = sym.Matrix(sym.MatrixSymbol('Yv', 2, 2))
      Qv = sym.Matrix(Qinv)
      Pv = sym.Matrix(P)
      Av = sym.Matrix(A)
      Yv=sym.solve(Av.T*Yv+Yv*Av+Pv-Yv*Fv*Qv*Fv.T*Yv, Yv)
      #print(Yv[3])
      Yv=np.array(list(Yv[3]))
      Yv=np.array( [ [Yv[0],Yv[1]],[Yv[2],Yv[3]] ])
      Yv=Yv.astype(np.float)
          
      

      def __init__(self):
              
              super(SqueezeEnv, self).__init__() #questo non ho capito a che serve ma lo tengo
              
              #tentativo di dare spazi di azioni e osservazioni, per come lo vorrei fare lo spazio delle azioni è un float da -1 a 1 per u[0] e un altro analogo per u[1]
              #mentre lo spazio delle osservazioni è una matrice 3x2 dove la prima riga rende il momento primo e le altre due righe sono la matrice di covarianza. Per ora non ho idea se ho capito come funziona la notazione
              self.action_space = spaces.Box(
                  low=-np.inf , high=np.inf ,shape=(4,), dtype=np.float32)
              #print(self.action_space)    
              
              self.observation_space = spaces.Box(
                  low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
              
              
      
      def step(self, action):
              print(prova)
              dt=self.dt 
              #print(action)
              ##richiamo il momento primo e aggiorno lo stato d'ambiente (inizializzato a ogni step a zero,zero) 
              rbcm=np.zeros(2)
              rc=self.momento_primo
              rbcm=rbcm+Sy.dot(C.T.dot(rc))*(dt**0.5)
              
              #estraggo rm per poi definire l'incremento dw
              #rm2=np.random.multivariate_normal(rbcm, (Sb+Sm)/2)
              dwm=np.random.randn(2)*(dt**0.5)#((SIGMA).dot(rm2-rbcm))*(dt**0.5)
              
              #momento primo con feedback, dall'azione scelta faccio la matrice K, poi riprendo sc da matrice_covarianza
              action=np.array([[action[0],action[1]],[action[2],action[3]]])
              #print(action)
              sc=self.matrice_covarianza
              #definisco u e aggiorno il momento primo
              u=np.array(action).dot(rc)
              drc=A.dot(rc)*dt+(2**(-0.5))*(E-sc.dot(B)).dot(dwm)+F.dot(u)*dt
              rc=rc+drc
              
              #aggiorno la matrice di covarianza
              sc=sc+dt*((A.dot(sc)+sc.dot(A.T)+D)-(E-sc.dot(B)).dot((E-sc.dot(B)).T))
              
              #funzione costo
              h=0.5*sc[0,0]+rc[0]**2 + u.T.dot(Q.dot(u)) #fatta a mente ma mi sembra venga così per la P 1,0,0,0
              costoQ= u.T.dot(Q.dot(u))
              costoP=h-costoQ
              self.current_reward=-h#(h+0.001)**-4
                  
              #provo a dare un criterio per smettere dopo un po' che siamo abbastanza vicini allo steady state
              distance=(sc[0,0]-sigmacss[0,0])**2#+abs(sc[1,1]-sigmacss[1,1])
              if distance<=eps:
                  self.Done=True
              #alla fine salvo il momento primo e la matrice di covarianza e le metto nell'output
              self.momento_primo=rc
              self.matrice_covarianza=sc
              output=[rc[0],rc[1],sc[0,0],sc[0,1],sc[1,0],sc[1,1]]
              output=np.array(output)
              return output , self.current_reward , self.Done , {'costoP': costoP, 'costoQ':costoQ,'K(t)':action}
    
      #provo a mettere l'agente "imparato"
      def optimal_agent(self):
            
              dt=self.dt 
              
              #aggiorno il momento primo dello stato d'ambiente 
              rbcm=np.zeros(2)
              rc=self.momento_primo_2
              rbcm=rbcm+Sy.dot(C.T.dot(rc))*(dt**0.5)
              
              #rm2=np.random.multivariate_normal(rbcm, (Sb+Sm)/2)
              dwm=np.random.randn(2)*(dt**0.5)#((SIGMA).dot(rm2-rbcm))*(dt**0.5)
              
              #Y=self.Yv
              Y=self.Y
              P=self.P
              #momento primo con feedback
              sc=self.matrice_covarianza_2
              Y=Y+dt*( Y.dot(A.T)+A.dot(Y) + P - Y.dot(F.dot(Qinv.dot((F.T).dot(Y)))))
              Kopt=Qinv.dot((F.T).dot(Y))
              Ab=A-F.dot(Kopt)
              drc=((Ab).dot(rc))*dt+( E-sc.dot(B)).dot(dwm)+Sy.dot(C.dot(rbcm))*(dt**0.5)
              rc=rc+drc
              u=Kopt.dot(rc)
              self.Y=Y
              #print(Kopt)
              
              #matrice di covarianza
              sc=sc+dt*((A.dot(sc)+sc.dot(A.T)+D)-(E-sc.dot(B)).dot((E-sc.dot(B)).T))
              
              #funzione costo
              h=0.5*sc[0,0]+rc[0]**2 + u.T.dot(Q.dot(u)) #fatta a mente ma mi sembra venga così per la P 1,0,0,0
              costoQ= u.T.dot(Q.dot(u))
              costoP=h-costoQ
              self.current_reward_2=-h#(h+0.001)**-4
                  
              #provo a dare un criterio per smettere dopo un po' che siamo abbastanza vicini allo steady state
              distance=(sc[0,0]-sigmacss[0,0])**2#+abs(sc[1,1]-sigmacss[1,1])
              if distance<=eps:
                  self.Done_2=True
                  
              self.momento_primo_2=rc
              self.matrice_covarianza_2=sc
              output=[rc[0],rc[1],sc[0,0],sc[0,1],sc[1,0],sc[1,1]]
              return np.array(output) , self.current_reward_2 , self.Done_2 , {'costoP': costoP, 'costoQ':costoQ,'K(t)':Kopt}

      def reset(self):
                            
              #reinizializzo delle cose
              #parto da punti diversi ogni volta e vediamo
              #a=0.5
              #d=0.5
              #n=5
              
              a=rand.uniform(-1,1)
              d=rand.uniform(-1,1)
              n=rand.uniform(0,10)
              
              #setto i momenti primi iniziali
              self.momento_primo=np.array([a,d])
              self.momento_primo_2=np.array([a,d])
              
              #setto le covarianze iniziali e i 'Done'
              self.matrice_covarianza=(2*n+1)*np.array([[1,0],[0,1]])
              self.Done=False
              self.matrice_covarianza_2=(2*n+1)*np.array([[1,0],[0,1]])
              self.Done_2=False
              
              rc=self.momento_primo
              sc=self.matrice_covarianza
              
              #resetto a zero la matrice Y per il feedback e preparo l'output
              self.Y=0*np.identity(2)
              output=[rc[0], rc[1], sc[0,0], sc[0,1], sc[1,0], sc[1,1]]
              output=np.array(output)
              
              #resetto le reward
              self.current_reward=0
              self.current_reward_2=0
              
              return output
          
      
      
      def render(self,mode='human'):
              print(self.momento_primo,self.current_reward)
    

