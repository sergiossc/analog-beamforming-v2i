import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

Lt = 16
Lr = 16

nt = 32
nr = 32


rx_slots = np.linspace(-1, 1, nr+1)
rx_slot = 1/Lr # 0.5
half_rx_slot = rx_slot/2 # 0.25
rx_slots_max = rx_slots + half_rx_slot
rx_slots_min = rx_slots - half_rx_slot
print (f'RX SLOTS:')
print (f'rx_slots_max:\n{rx_slots_max}')
print (f'rx_slots_min:\n{rx_slots_min}')


tx_slots = np.linspace(-1, 1, nt+1)
tx_slot = 1/Lt # 0.5
half_tx_slot = tx_slot/2 # 0.25
tx_slots_max = tx_slots + half_tx_slot
tx_slots_min = tx_slots - half_tx_slot


print (f'TX SLOTS:')
print (tx_slots_max)
print (tx_slots_min)



paths = []
n_paths = 500
for n in range(n_paths):
    pathgain = np.sqrt(nr * nt) * np.random.rand() * np.exp(1j * 0)
    #aoa = np.random.choice(np.linspace(np.deg2rad(80), np.deg2rad(100), num=50))
    aoa = np.random.choice(np.linspace(0, 2 * np.pi, num=100))
    aod = np.random.choice(np.linspace(0, 2 * np.pi, num=100))
    #aod = np.random.choice(np.linspace(np.deg2rad(80), np.deg2rad(100), num=50))
    p = {'pathgain': pathgain, 'aoa': aoa, 'aod':aod}
    paths.append(p)



h_a_dict = {}
for k in range(nr+1):
    for l in range(nt+1):
        h_a_dict[(k,l)] = []

for p in paths:
    aoa = p['aoa']
    aod = p['aod']
    pathgain = p['pathgain']

    print ('AoA')
    cos_aoa = np.cos(aoa)
    print (f'input: {cos_aoa}')
    
    rx_index = []

    if (cos_aoa > rx_slots_min[0] and cos_aoa < rx_slots_max[0]) or (cos_aoa > rx_slots_min[-1] and cos_aoa < rx_slots_max[-1]):
        print (f'output: 0, {nr}')
        rx_index.append(0)
        rx_index.append(nr)
    else:
        for n in range(1, nr):
            v_min = rx_slots_min[n]
            v_max = rx_slots_max[n]
            if cos_aoa > v_min and cos_aoa < v_max:
                print (f'output: {n}')
                rx_index.append(n)

    print ('AoD')
    cos_aod = np.cos(aod)
    print (f'input: {cos_aod}')
    
    tx_index = []
    if (cos_aod > tx_slots_min[0] and cos_aod < tx_slots_max[0]) or (cos_aod > tx_slots_min[-1] and cos_aod < tx_slots_max[-1]):
        print (f'output: 0, {nt}')
        tx_index.append(0)
        tx_index.append(nt)
    else:
        for n in range(1, nt):
            v_min = tx_slots_min[n]
            v_max = tx_slots_max[n]
            if cos_aod > v_min and cos_aod < v_max:
                print (f'output: {n}')
                tx_index.append(n)

    for k in rx_index:
        for l in tx_index:
            #h_a[k, l] += pathgain
            h_a_dict[(k,l)].append(pathgain)

print (h_a_dict) 

h_a = np.zeros((nr+1, nt+1), dtype=complex)

for k in range(nr):
    for l in range(nt):
        mean_pathgain = np.mean(np.abs(h_a_dict[(k,l)]))
        if np.isnan(mean_pathgain):
            pass
        else:
            print (f'mean_pathgain: {mean_pathgain}')
            print ('------')
            h_a[k,l] = mean_pathgain


x = np.arange(nr+1)

y = np.arange(nt+1)
X, Y = np.meshgrid(x, y)
Z = np.abs(h_a.T)

fig = plt.figure(figsize=(10, 6))
 
ax1 = fig.add_subplot(111, projection='3d')
#surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax1.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
ax1.set_xlabel('RX bins')
ax1.set_ylabel('TX bins')
ax1.set_zlabel('|h_a|')
ax1.set_title('gist_earth color map')
surf1 = ax1.plot_surface(X, Y, Z, cmap=cm.viridis)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
plt.show()

