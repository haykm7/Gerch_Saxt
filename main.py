import numpy as np
import random
import math
import matplotlib.pyplot as plt

L = 70
N = 5
M = 5

#mu, sigma = 0, 0.1
#s1 = np.random.normal(mu, sigma, N*M)
#s2 = np.random.normal(mu+1, sigma+1, N*M)

TransmX = np.zeros((M,N), dtype = 'complex_')
TransmX_Real = np.zeros((M,N), dtype = 'complex_')


P = np.exp(1j*np.random.uniform(0,2*np.pi,(L,N)))
#for index, value in np.ndenumerate(P):
    #P[index] = np.exp(1j*random.uniform(0, 2*math.pi))


TransmX = np.random.normal(size=(N,M)) + 1j * np.random.normal(size=(N,M))
TransmX_Real = np.random.normal(size=(N,M)) + 1j * np.random.normal(size=(N,M))

#TransmX = np.random.randn(N,M)
#TransmX_Real = np.random.randn(N,M)
#for index, value in np.ndenumerate(TransmX):
#    TransmX[index] = s1[i]
#    TransmX_Real[index] = s2[i]
#    i+=1


E_out_real = np.square(np.abs(np.matmul(P,np.transpose(TransmX_Real))))
Intensity = np.sqrt(E_out_real)
E = np.matmul(P,np.transpose(TransmX))
E_new = 0
E_vect = []
TransmX_new = 0
TransmX_vect = [TransmX]
#print(TransmX_vect[0])
#print(np.corrcoef(TransmX_vect[-1],TransmX_vect[-1])[0][0])
cond = True
_ = 0
iteration_vect = []
correlation_vect = []

while cond:
    E_new = np.multiply(Intensity,np.exp(1j*np.angle(np.matmul(P,np.transpose(TransmX)))))
    E_vect.append(E_new)
    TransmX_new = np.transpose(np.matmul(np.linalg.pinv(P),E_new))
    TransmX_vect.append(TransmX_new)
    if _ < 1:
        pass
    _+=1
    iteration_vect.append(_)
    if _ >= 2:
        correlation_vect.append(np.abs(np.corrcoef(TransmX_vect[-1],TransmX_Real)[0][5]))
        if np.abs(np.corrcoef(TransmX_vect[-1],TransmX_vect[-3])[0][5]) >= 0.9999999:
            #print(TransmX_vect[-1])
            #print(np.corrcoef(TransmX_vect[-1],TransmX_Real)[0][0])
            print(f'Number of iterations is {_}')
            cond = False
    E = E_vect[-1]
    TransmX = TransmX_vect[-1]
#print(E_vect)
print(np.abs(np.corrcoef(TransmX_vect[-1],TransmX_Real)[0][5]))
print(correlation_vect)
plt.plot(iteration_vect[:-1],correlation_vect)
plt.show()



def lambda_enumeration():
    T0 = np.random.normal(size=(N,M)) + 1j * np.random.normal(size=(N,M))
    T_target  = np.random.normal(size=(N,M)) + 1j * np.random.normal(size=(N,M))

    lambda_array = np.zeros(N,dtype='complex_')
    i = 0
    lambda_array = T_target / T0
    #for index in np.ndenumerate(T_target):
    #    lambda_array[i] = T_target[index]/T0[index]
    #    i+=1
    print(np.abs(lambda_array))
lambda_enumeration()
