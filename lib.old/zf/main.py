import numpy as np
from numpy.linalg import svd

sigma = 1
n_streams = 4

# cria n_streams streams de simbolos complexos
s = np.sqrt(sigma/2)*(np.random.randn(n_streams,1) + np.random.randn(n_streams,1)*(1j))

#print ("simbols: ",s)
print ("simbols.shape: ",s.shape)

#cria um canal complexo gaussiano circular considerando num_tx antenas de transmissao e num_rx antenas de recepcao
num_tx = 16
num_rx = 9
h = np.sqrt(sigma/2)*(np.random.randn(num_tx, num_rx) + np.random.randn(num_tx, num_rx)*(1j))

#print ("h ", h)
print ("h.shape: ", h.shape)

# calculando o precoder considerando a info de h por meio do svd(h)
u, d, v = svd(h, full_matrices=True)

u_f = v.conj().T

print ("u_f: ", u_f.shape)

print(d)











#m = v.conj().T
#m = m[0:n_streams,:]

#print ("m.shape: ", m.shape)


#z_opt = u.conj().T
#z_opt = z_opt[:,0:n_streams]
#print("z_opt.shape: ", z_opt.shape)

#x = np.matmul(s.T, m)
#print ("x.shape: ", x.shape)

#
#y = np.matmul(x, h)
#print ("y.shape: ", y.shape)
#
#y_est = np.matmul(y, z_opt)
#print ("y_est.shape: ", y_est.shape)
#
#
#print ("s: ", np.abs(s))
#print ("y_est: ", np.abs(y_est.T))
