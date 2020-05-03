#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:47:10 2020

@author: nicolas
"""
from numpy import exp
from math import sqrt
import numpy as np
from numpy import random
from math import pi
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
from tabulate import tabulate
from scipy.stats import t
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from function_memoire import*

########### FONCTIONS RELATIVES A LA FONCTION :
    
    
# Density of Noise
def pn(x, m0, s0):
      return norm.pdf(x, m0, s0);

def pn_t(x,n):
      return t.pdf(x,n);

## density of  posterior  unormalized:
def pm0(x, m , s ):
      return exp(-0.5*((x-m)/s)**2);
    
## tensorflow:

def h(c):
    
    return(sum(pm0*c/(pm0*c + pn))/len(x)+sum(pn/(pn+pm0*c))/len(x))

    
def h_min():
    return(sum(pm0*c/(pm0*c + pn))/len(x)+sum(pn/(pn+pm0*c))/len(x))

############ CLASSS GRADIENT:
    
class Gradient:


    # instance attribute
    def __init__(self, cte,mu,sigma, error_mu,error_sigma, error_cte, ctes,mus,sigmas):
        
              self.cte = cte
              self.mu = mu
              self.sigma = sigma
        
              self.error_mu = error_mu
              self.error_sigma = error_sigma
              self.error_cte = error_cte 
        
              self.ctes = ctes
              self.mus = mus
              self.sigmas = sigmas
            
class Grad:
    
    def __init__(self, cte,std):
        
        self.cte = cte
        self.std = std


class Gradient_t:


    # instance attribute
    def __init__(self, cte, n , error_mu,error_sigma, error_cte, ctes,mus,sigmas):
              self.cte = cte
              self.n = n
              
              self.error_mu = error_mu
              self.error_sigma = error_sigma
              self.error_cte = error_cte 
        
              self.ctes = ctes
              self.mus = mus
              self.sigmas = sigmas

            
def kl_divergence(p, q): #distance de Kullback-Leibler
    return np.mean(np.where(p != 0, np.log(p / q), 0))

def f(k, i): #fonction pour dessiner en 3D 
    return L[np.argwhere(L_mu==k)[0][0], np.argwhere(L_sigma==i)[0][0]]
        
 #############################@@@@@@@@@@@@@@@@@ FONCTIONS 1D @@@@@@@@@@@@@@@@@@@@############################       
############# NCE_1:

def NCEDescent1D_T(X1,batch_size, m, s,n, nu = 1):
    

    Y = np.random.standard_t(n,int(batch_size*nu))
    X = np.hstack((X1,Y))


    C1=np.ones(batch_size)
    c= np.zeros(nu*batch_size)
    C = np.hstack((C1,c))

    Z = np.log(pm0(X,m,s)/pn_t(X,n))

    mydata = {}
    mydata['C'] = C
    mydata['Z'] = Z


    mylogit = smf.glm(formula='C ~ 1', offset = Z-np.log(nu) , data=mydata, family=sm.families.Binomial())
    res = mylogit.fit()
    
    result = Grad(exp(res.params[0]),res.bse[0])
    
    
    return(result)


def NCEDescent1D(X1,size, m, s,mu_noise, sigma_noise, nu = 1):
    

    Y = np.random.normal (mu_noise, sigma_noise, nu*size)
    X = np.hstack((X1,Y))


    C1=np.ones(size)
    c= np.zeros(nu*size)
    C = np.hstack((C1,c))

    Z = np.log(pm0(X,m,s)/pn(X,mu_noise,sigma_noise))

    mydata = {}
    mydata['C'] = C
    mydata['Z'] = Z


    mylogit = smf.glm(formula='C ~ 1', offset = Z-np.log(nu) , data=mydata, family=sm.families.Binomial())
    res = mylogit.fit()
    
    result = Grad(exp(res.params[0]),res.bse[0])
    
    
    return(result)


################### GAN 1D:
    
def GANDescent(x_batches, m, s,mu_init , sigma_init, cte_init , learning_rate = [0.01,0.01], max_iters = 500, nu=1):    
    
    m0 = mu_init 
    s0 =sigma_init
    cte = cte_init
    
    error_mu = [] 
    error_sigma = []
    error_cte = [] 
    
    mus = [m0]
    sigmas = [s0]
    ctes = [cte]

    batch_size= len(x_batches[0])
     
    for itr in range(max_iters): 
        
        for x in x_batches:
            
            z= random.normal( 0, 1, int(batch_size)) 
            q = m0+s0*z
            
            #Gradient in respect to the constant
            grad_cte = np.sum( 1/cte - pm0(x,m,s)/(cte*pm0(x,m,s)+ pn(x, m0,s0)) )/batch_size - (1/ batch_size)*np.sum( pm0(q,m,s)/(cte*pm0(q,m,s)+ pn(q,m0,s0)) )
            cte = cte + learning_rate[0] * grad_cte
            error_cte.append( (grad_cte) ) 
            

            #Gradient in respect to noise parameters
           
            grad_mu = -(q-m0)/s0**2 +((q-m0)*norm.pdf(q, m0, s0)/s0**2 + 
                        (q-m)*cte*exp(-0.5*((q-m)/s)**2)/s**2)/(cte*pm0(q,m,s) + norm.pdf(q,m0,s0))
            grad_sigma = grad_mu*z
            
            grad_mu = np.sum(grad_mu)/batch_size
            grad_sigma = np.sum(grad_sigma)/batch_size
            
            m0 = m0 - learning_rate[1] * grad_mu
            s0 = s0 - learning_rate[1] * grad_sigma
            
            error_mu.append( (grad_mu) ) 
            error_sigma.append((grad_sigma))
            
            ctes.append(cte)
            mus.append(m0)
            sigmas.append(s0)
            
            if ( abs(ctes[-1] - ctes[-2]) < 1e-6 and abs(mus[-1]-mus[-2])<1e-6 and abs(sigmas[-1]-sigmas[-2])<1e-6):
                return Gradient(cte,m0,s0, error_mu,error_sigma, error_cte, ctes,mus,sigmas)
            
  
    result = Gradient(cte,m0,s0, error_mu,error_sigma, error_cte, ctes,mus,sigmas)
    return result




##### BONUS ###

def NCE_Descent1D(x_batches, m, s,mu_init =10, sigma_init=10, cte_init = 0.2, learning_rate = [0.01, 0.01],max_iters = 50, nu = 1):    
    
    m0 = mu_init 
    s0 =sigma_init
    cte = cte_init
    
    #not used for nce
    error_mu = [] 
    error_sigma = []
    error_cte = [] 
    
    mus = []
    sigmas = []
    ctes = [cte_init]
     
    
    batch_size= len(x_batches[0])

    
    for itr in range(max_iters): 
        
        for x in x_batches:
            
            z= random.normal( 0, 1, int(batch_size*nu) )
            q = m0+s0*z 
            
            #Gradient in respect to the constant
            pm0_x = pm0(x,m,s)
            pm0_q = pm0(q,m,s)
            
            grad_cte = np.sum( 1/cte - pm0_x/(cte*pm0_x+ nu*pn(x, m0,s0)) )/batch_size - (1/ batch_size)*np.sum( pm0_q/(cte*pm0_q+nu*pn(q,m0,s0)) )
            cte = cte + learning_rate[0] * grad_cte
            error_cte.append( grad_cte ) 
            
            ctes.append(cte)
            #mus.append(mu)
            #sigmas.append(sigma)
            
            if (tf.norm(grad_cte)<1e-6):
                print(itr)
                return Gradient(cte,m0,s0, error_mu,error_sigma, error_cte, ctes,mus,sigmas)
    
    print(itr)    
            
    result = Gradient(cte,m0,s0, error_mu,error_sigma, error_cte, ctes,mus,sigmas)
    return result

def NCEDescent1D_t(x_batches, m, s,n, cte_init = 0.2, learning_rate = [0.01, 0.01],max_iters = 50, nu = 1):    
    

    cte = cte_init
    
    #not used for nce
    error_mu = [] 
    error_sigma = []
    error_cte = [] 
    
    mus = []
    sigmas = []
    ctes = [cte_init]
     
    
    batch_size= len(x_batches[0])

    
    for itr in range(max_iters): 
        
        for x in x_batches:
            
            q= random.standard_t(n,int(batch_size*nu))
            
            #Gradient in respect to the constant
            pm0_x = pm0(x,m,s)
            pm0_q = pm0(q,m,s)
            
            grad_cte = np.sum( 1/cte - pm0_x/(cte*pm0_x+ nu*pn_t(x, n)) )/batch_size - (1/ batch_size)*np.sum( pm0_q/(cte*pm0_q+nu*pn_t(q,n)) )
            cte = cte + learning_rate[0] * grad_cte
            error_cte.append( grad_cte ) 
            
            ctes.append(cte)
            #mus.append(mu)
            #sigmas.append(sigma)
            
            if (tf.norm(grad_cte)<1e-9):
                print("convergé au bout de ", itr, "itérations")
                return Gradient_t(cte,n, error_mu,error_sigma, error_cte, ctes,mus,sigmas)
    
    print(" a fait le max d'itération")    
            
    result = Gradient_t(cte,n, error_mu,error_sigma, error_cte, ctes,mus,sigmas)
    return result
############# other descent algorithms:
    
    
            
def NCERMS1D(x_batches, m, s,mu_init =10, sigma_init=10, cte_init = 0.2, learning_rate = [0.01, 0.01],max_iters = 50, nu = 1):    
    
    m0 = mu_init 
    s0 =sigma_init
    cte = tf.Variable(cte_init,dtype='float32')
    print(cte)
    #not used for nce
    error_mu = [] 
    error_sigma = []
    error_cte = [] 
    lr=learning_rate[0]
    mus = []
    sigmas = []
    ctes = [cte_init]
     
    batch_size= len(x_batches[0])
    
    opt= tf.optimizers.RMSprop(learning_rate=lr)

    
    for itr in range(max_iters): 
        
        for x in x_batches:
            z= random.normal( 0, 1, int(batch_size*nu) )
            q = m0+s0*z 
            #print('q===',q)
            #Gradient in respect to the constant
            pm0_x = pm0(x,m,s)
            pm0_q = pm0(q,m,s)
            pn_q  = pn(q,m0,s0)
            pn_x  = pn(x,m0,s0)
            #print('PMO_x', pm0_x)
            #print('PMO_q===',pm0_q)
            #print('PN_Q====',pn_q)
            #print('PN_X===', pn_x)
            
            
            def h_min():
                return(-tf.math.reduce_mean(tf.math.log(pm0_x*cte/(pm0_x*cte + nu*pn_x)))-nu*tf.math.reduce_mean(tf.math.log(pn_q/(nu*pn_q+pm0_q*cte))))
                
            def h(c):
                return(-tf.math.reduce_mean(tf.math.log(pm0_x*cte/(pm0_x*cte + nu*pn_x)))-nu*tf.math.reduce_mean(tf.math.log(pn_q/(nu*pn_q+pm0_q*cte))))
                #print('H_min_x===',pm0_x*cte/(pm0_x*cte + pn_x))
                #print('H_min_y====',pn_q/(pn_q+pm0_q*cte))
                
                
    
            #def h(cte):
                #return(sum(tf.math.log(pm0_x*cte/(pm0_x*cte + pn_x)))/len(x)+sum(tf.math.log(pn_q/(pn_q+pm0_q*cte)))/len(x))

            #print('CTE===',cte)
            opt.minimize(h_min,var_list=[cte])
            print("CTE3=====",cte.numpy())
            ctes.append(cte.numpy())
            if (tf.norm(ctes[-1]-ctes[-2])<1e-5):
                return Gradient(cte,m0,s0, error_mu,error_sigma, error_cte, ctes,mus,sigmas)
            
    result = Gradient(cte,m0,s0, error_mu,error_sigma, error_cte, ctes,mus,sigmas)
    return result

def NCEAdam1D(x_batches, m, s,mu_init =10, sigma_init=10, cte_init = 0.2, learning_rate = [0.01, 0.01],max_iters = 50, nu = 1):    
    
    m0 = mu_init 
    s0 =sigma_init
    cte = tf.Variable(cte_init,dtype='float32')
    print(cte)
    #not used for nce
    error_mu = [] 
    error_sigma = []
    error_cte = [] 
    lr= learning_rate[0]
    
    mus = []
    sigmas = []
    ctes = [cte_init]
     
    batch_size= len(x_batches[0])
    
    opt= tf.optimizers.Adam(learning_rate = lr)

    
    for itr in range(max_iters): 
        
        for x in x_batches:
            z= random.normal( 0, 1, int(batch_size*nu) )
            q = m0+s0*z 
            #print('q===',q)
            #Gradient in respect to the constant
            pm0_x = pm0(x,m,s)
            pm0_q = pm0(q,m,s)
            pn_q  = pn(q,m0,s0)
            pn_x  = pn(x,m0,s0)
            #print('PMO_x', pm0_x)
            #print('PMO_q===',pm0_q)
            #print('PN_Q====',pn_q)
            #print('PN_X===', pn_x)
            
            
            def h_min():
                return(-tf.math.reduce_mean(tf.math.log(pm0_x*cte/(pm0_x*cte + nu*pn_x)))-nu*tf.math.reduce_mean(tf.math.log(pn_q/(nu*pn_q+pm0_q*cte))))
                
            def h(c):
                return(-tf.math.reduce_mean(tf.math.log(pm0_x*cte/(pm0_x*cte + nu*pn_x)))-nu*tf.math.reduce_mean(tf.math.log(pn_q/(nu*pn_q+pm0_q*cte))))
                #print('H_min_x===',pm0_x*cte/(pm0_x*cte + pn_x))
                #print('H_min_y====',pn_q/(pn_q+pm0_q*cte))
                
                
    
            #def h(cte):
                #return(sum(tf.math.log(pm0_x*cte/(pm0_x*cte + pn_x)))/len(x)+sum(tf.math.log(pn_q/(pn_q+pm0_q*cte)))/len(x))

            #print('CTE===',cte)
            opt.minimize(h_min,var_list=[cte])
            print("CTE3=====",cte.numpy())
            ctes.append(cte.numpy())
            if (tf.norm(ctes[-1]-ctes[-2])<1e-5):
                return Gradient(cte,m0,s0, error_mu,error_sigma, error_cte, ctes,mus,sigmas)
            
    result = Gradient(cte,m0,s0, error_mu,error_sigma, error_cte, ctes,mus,sigmas)
    return result

def NCEMom1D(x_batches, m, s,mu_init =10, sigma_init=10, cte_init = 0.2, learning_rate = [0.01, 0.01],max_iters = 50, nu = 1):    
    
    m0 = mu_init 
    s0 =sigma_init
    cte = tf.Variable(cte_init,dtype='float32')
    print(cte)
    #not used for nce
    error_mu = [] 
    error_sigma = []
    error_cte = [] 
    lr= learning_rate[0]
    
    mus = []
    sigmas = []
    ctes = [cte_init]
     
    batch_size= len(x_batches[0])
    
    opt= tf.optimizers.SGD(learning_rate = lr, momentum=0.9)

    
    for itr in range(max_iters): 
        
        for x in x_batches:
            z= random.normal( 0, 1, int(batch_size*nu) )
            q = m0+s0*z 
            #print('q===',q)
            #Gradient in respect to the constant
            pm0_x = pm0(x,m,s)
            pm0_q = pm0(q,m,s)
            pn_q  = pn(q,m0,s0)
            pn_x  = pn(x,m0,s0)
            #print('PMO_x', pm0_x)
            #print('PMO_q===',pm0_q)
            #print('PN_Q====',pn_q)
            #print('PN_X===', pn_x)
            
            
            def h_min():
                return(-tf.math.reduce_mean(tf.math.log(pm0_x*cte/(pm0_x*cte + nu*pn_x)))-nu*tf.math.reduce_mean(tf.math.log(pn_q/(nu*pn_q+pm0_q*cte))))
                
            def h(c):
                return(-tf.math.reduce_mean(tf.math.log(pm0_x*cte/(pm0_x*cte + nu*pn_x)))-nu*tf.math.reduce_mean(tf.math.log(pn_q/(nu*pn_q+pm0_q*cte))))
                #print('H_min_x===',pm0_x*cte/(pm0_x*cte + pn_x))
                #print('H_min_y====',pn_q/(pn_q+pm0_q*cte))
                
                
    
            #def h(cte):
                #return(sum(tf.math.log(pm0_x*cte/(pm0_x*cte + pn_x)))/len(x)+sum(tf.math.log(pn_q/(pn_q+pm0_q*cte)))/len(x))

            #print('CTE===',cte)
            opt.minimize(h_min,var_list=[cte])
            print("CTE3=====",cte.numpy())
            ctes.append(cte.numpy())
            if (tf.norm(ctes[-1]-ctes[-2])<1e-5):
                return Gradient(cte,m0,s0, error_mu,error_sigma, error_cte, ctes,mus,sigmas)
            
    result = Gradient(cte,m0,s0, error_mu,error_sigma, error_cte, ctes,mus,sigmas)
    return result

