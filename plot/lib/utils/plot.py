import numpy as np
from numpy.linalg import norm
#import matplotlib 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.constants import c
import sys
import pandas as pd
from mayavi import mlab


def plot_histogram(sets, title, filename):
    fig, ax = plt.subplots()
    data = []
    for cw_id, sample_ids in sets.items():
        for sample_id in sample_ids:
            data.append(cw_id)
    pd.Series(data).value_counts().plot('bar')
    ax.set_title(title)
    #filename = 'histogram.png'
    fig.savefig(filename)
    #plt.show()

def plot_rate(opt_rate, est_rate, opt_label, est_label, filename):
    fig, ax = plt.subplots()
    opt_r = np.array([v for k,v in opt_rate.items()])
    est_r = np.array([v for k,v in est_rate.items()])
    diff_r = opt_r - est_r
    #fig = plt.figure()
    #fig.add_subplot(111)
    ax.set_title('Rate in bits/s/Hz')
    ax.plot(opt_r, label=opt_label)
    ax.plot(est_r, label=est_label)
    ax.plot(diff_r, label='diff rate')
    ax.legend()
    #plt.show()
    fig.savefig(filename)

def plot_paths(id_scene, chn_paths, tx_transceiver, rx_transceiver):

    departure_theta = []
    arrival_theta = []

    for chn_path in chn_paths:
        departure_theta.append(float(chn_path['departure_theta']))
        arrival_theta.append(float(chn_path['arrival_theta']))

    departure_theta = np.deg2rad(departure_theta)
    arrival_theta = np.deg2rad(arrival_theta)

    fig = plt.figure()
    # ... at tx transceiver
    #fig.add_subplot(211, projection='polar')
    fig.add_subplot(111, projection='polar')
    plt.title('Tx' + tx_transceiver['label'] + ' DoD paths')
    for t in departure_theta:
        plt.polar((0, t), (0, 1))

    # ... at rx transceiver
    #fig.add_subplot(212, projection='polar')
    #plt.title('Rx' + rx_transceiver['label'] + ' DoA paths')
    #for t in arrival_theta:
    #    plt.polar((0, t), (0, 1))

    plt.title(str(id_scene))
    plt.show()

def plot_devices(transceivers, rotate_axis=True):
    gridsize = (3, 1)
    fig = plt.figure(figsize=(7,7))
    ax1 = plt.subplot2grid(gridsize, (0, 0))
    ax1.set_title('XY')
    ax2 = plt.subplot2grid(gridsize, (1, 0))
    ax2.set_title('XZ')
    ax3 = plt.subplot2grid(gridsize, (2, 0))
    ax3.set_title('YZ')

    posx = []
    posy = []
    posz = []
    device_label = []

    for transceiver in transceivers:
        px = float(transceiver['posx'])
        py = float(transceiver['posy'])
        pz = float(transceiver['posz'])
        d_label = transceiver['type']+transceiver['label']

        posx.append(px)
        posy.append(py) #float(transceiver['posy']))
        posz.append(pz) #float(transceiver['posz']))
        device_label.append(d_label) #transceiver['type']+transceiver['label'])
        ax2.text(px, py, d_label)

    ax2.plot(posx, posy, 'bs')

    #plt.title(str(id_scene))
    if rotate_axis: 
        ax2.xlim(1000, 0)
        ax2.ylim(1000, 0)
    plt.show()

def plot_sample(training_set):
    for sample_id, sample in training_set.items():
        x_axis = []
        y_axis = []

        sample_shape = sample.shape
        x = sample_shape[0]
        y = sample_shape[1]

        for i in range (x):
            for j in range (y):
                x_axis.append(sample[i,j].real)
                y_axis.append(sample[i,j].imag)
        plt.plot(x_axis, y_axis, 's')        
    plt.show()

def plot_samples(sets, training_set):
    for cw_id, samples_id in sets.items():
        x_axis = []
        y_axis = []
        samples = [training_set[sample_id] for sample_id in samples_id]
        for sample in samples:
            sample_shape = sample.shape
            x = sample_shape[0]
            y = sample_shape[1]
   
            for i in range(x):
                for j in range(y):
                    x_axis.append(sample[i,j].real)
                    y_axis.append(sample[i,j].imag)

        plt.plot(x_axis, y_axis, 's')        
    plt.show()

def plot_performance(distortion_by_round, graph_title, filename):
    fig, ax = plt.subplots()
    for r, mean_distortion in distortion_by_round.items():
        ax.plot(mean_distortion, label='#cw: ' + str(2**r))
    plt.ylabel('distortion (MSE)')
    plt.xlabel('# iterations')
    plt.title(graph_title)
    plt.legend()
    #plt.show()
    #fig_filename = 'training_performance.png'
    fig.savefig(filename)

def PlotPattern(codebook, phased_array):
    for cw_id, cw in codebook.items():
        w = cw #np.array([w1, w2])
        theta = np.arange(-180, 180)
        gain_db = np.zeros(len(theta))
        for i in range(len(theta)):
            t = np.deg2rad(theta[i])
            a = [np.exp(-1j * 2 * np.pi * n * phased_array.element_spacing * np.sin(t)/phased_array.wave_length) for n in range(len(w))]
            product = np.matmul(w.T, a)
            gain_db[i] = 10 * np.log10((np.abs(product) ** 2)/np.abs(np.matmul(w.conj().T, w)))
        
        plt.plot(theta, gain_db, '-', label=str(cw_id))

    plt.ylabel('Gain(dB)')
    plt.xlabel('Angle(deg)')
    plt.legend()
    plt.show()

def RateCDF(rate):

    max_rate = np.max(rate)
    axis_rate = np.arange(0,max_rate+1, 0.00001)
    axis_rate = np.around(axis_rate, decimals=5)
    histogram_sample_rate = {}
    for r in axis_rate:
        histogram_sample_rate[r] = []
    rate = np.around(rate, decimals=5)
    for r in rate:
        histogram_sample_rate[r].append(r)
    length_rate = len(rate)
    cdf_rate = {}
    before_k = -1
    for k, v in histogram_sample_rate.items():
        
        if len(v)>0:
            cdf_rate[k] = len(v)/length_rate
            if before_k == -1:
                before_k = k
            else:
                cdf_rate[k] = len(v)/length_rate + cdf_rate[before_k]
                before_k = k
    x_cdf_rate = []
    y_cdf_rate = []
    for k, v in cdf_rate.items():
        x_cdf_rate.append(k)
        y_cdf_rate.append(v)

    return x_cdf_rate, y_cdf_rate
#my_rate = np.random.rand(1000) * 10
#ddPlotRateCDF(my_rate) 


#theta_interval = np.arange(-90,90)
#theta_interval = (theta_orig - 90) * -1
#theta_interval = theta_orig
#num_theta = 2
#theta = np.random.choice(theta_interval, num_theta, replace=False)

#complex_numbers = np.array([1 * np.exp(1j * np.deg2rad(t)) for t in theta])

#for complex_number in complex_numbers:
#    plt.plot([0,complex_number.real], [0,complex_number.imag])
#plt.savefigfig('angles.png')

def PlotPattern2():
    """
    ref: Balanis 2016, chp.06
    ref: https://www.mtt.org/wp-content/uploads/2019/01/beamform_mmw_antarr.pdf
    ref: http://www.waves.utoronto.ca/prof/svhum/ece422/notes/15-arrays2.pdf
    ref: https://www.ece.mcmaster.ca/faculty/nikolova/antenna_dload/current_lectures/L13_Arrays1.pdf
    """
    num_tx_elements = 16
    #fc = 60 * (10 ** 9) #60 GHz
    fc = 60 * (10 ** 9) #60 GHz
    wavelength = c/fc #in meters
    #element_spacing =  wavelength/2 #in meters
    element_spacing =  .5 * wavelength #5 / (10 ** 3) #wavelength/2 #in meters
    k = 2 * np.pi * 1/wavelength

    theta = np.arange(0, 180, 1)

    array_vec = np.arange(num_tx_elements)

    w = np.ones(num_tx_elements, dtype=complex)
    alpha = np.random.choice(theta)
    #alpha = 90 #np.random.choice(theta)
    print ('alpha: ', alpha)
    count = 1
    for i in range(len(w)):
        #t = np.random.choice(theta)
        t = alpha * count
        count += 1
        w[i] = 1 * np.exp(-1j * np.deg2rad(t))
    print (w)
    print (np.abs(w))

    product = np.zeros(len(theta), dtype=complex)
    f_psi = np.zeros(len(theta), dtype=complex)
    psi = np.zeros(len(theta))
    for i in range(len(theta)):
        t = np.deg2rad(theta[i])
        p = k * element_spacing * np.cos(t)
        psi[i] = p
        array_factor = np.array([np.exp(1j * n * p) for n in array_vec])

        product[i] = np.matmul(w.T, array_factor)
        product[i] = product[i]/np.matmul(w.conj().T, w)

        af0 = np.sum(array_factor) 
        af1 = af0 * np.exp(1j * p)
        af2 = af0 - af1 
        af3 = af2/(1 - np.exp(1j * p))
        f_psi[i] = af3#/len(array_vec)


    fig = plt.figure()

    angles = np.angle(product)
    mag = np.abs(product)
    fig.add_subplot(311, projection='polar')
    plt.polar(psi, mag)

    fig.add_subplot(312)
    plt.plot(psi, np.abs(product), label='AF x w')
    plt.legend()

    fig.add_subplot(313)
    plt.plot(psi, np.abs(f_psi), label='f(psi)')

    plt.legend()
    plt.show()

#PlotPattern2()
def PParttern(id_scene, aods, tx_array):
    class path:
        def __init__(self, id, aod):
            self.id = id
            self.aod = np.deg2rad(aod)
            self.at = None
    
    theta = np.arange(0,180)
    
    num_tx_elements = tx_array.size
    wavelength = tx_array.wave_length  #in meters
    element_spacing = tx_array.element_spacing #in meters
    array_vec = np.arange(num_tx_elements)
    k = 2 * np.pi * 1/wavelength
    #at_res = np.array([np.exp(-1j * 2 * np.pi * element_spacing * (1/wavelength) * np.cos(np.angle(aod))) for aod in aods])
    #at_res = np.array([np.exp(-1j * np.angle(aod)) for aod in aods])
    at_res = aods
    
    
    prod = np.zeros(len(theta), dtype=complex)
    f_psi = np.zeros(len(theta), dtype=complex)
    psi = np.zeros(len(theta))
    for i in range(len(theta)):

        t = np.deg2rad(theta[i])
        p = k * element_spacing * np.cos(t)
        psi[i] = p
        af = np.array([np.exp(1j * n * p) for n in range(len(array_vec))])

        af0 = np.sum(af) 
        af1 = af0 * np.exp(1j * p)
        af2 = af0 - af1 
        af3 = af2/(1 - np.exp(1j * p))
        f_psi[i] = af3/len(array_vec)

        product = np.matmul(at_res.T, af)
        product = product/np.matmul(at_res.conj().T, at_res)#print ('product: \n', product)
        prod[i] = product #/len(array_vec)
    #plt.plot(theta, prod)
    #plt.legend()
    #plt.savefigfig('steering.png')
    #plt.show()
    fig = plt.figure()

    angles = np.deg2rad(theta)
    mag = np.abs(prod)
    fig.add_subplot(111, projection='polar')
    #fig.add_subplot(111)
    plt.polar(angles, mag)
    #plt.polar(psi, f_psi)
    plt.title(str(id_scene))
    plt.show()

#aods = np.array(np.deg2rad([10, 20, 30, 40]))
#PParttern(aods)
def AntennaPartern(w, tx_array):
    theta = np.arange(0,180)
   
    num_tx_elements = tx_array.size
    wavelength = tx_array.wave_length  #in meters
    element_spacing = tx_array.element_spacing #in meters
    tx_array_vec = np.arange(num_tx_elements)
   
    at = w
    prod = np.zeros(len(theta), dtype=complex)
    for i in range(len(theta)):
        t = np.deg2rad(theta[i])
        af = np.array([np.exp(1j * 2 * np.pi * n * element_spacing * (1/wavelength) * np.cos(t)) for n in range(len(tx_array_vec))])
        product = np.matmul(at.T, af)
        product = (np.abs(product) ** 2)/np.abs(np.matmul(at.conj().T, at))#print ('product: \n', product)
        prod[i] = product/num_tx_elements
    plt.plot(theta, np.abs(prod), label='w')
    plt.legend()
    #plt.savefigfig('steering.png')
    plt.show()

def pattern3d(phased_array, w, title):

    #fc = 60 * (10 ** 9)
    wavelength = phased_array.wave_length
    d = phased_array.element_spacing
    k = 2 * np.pi * (1/wavelength)

    precision = 50  # of samples
    t = np.linspace(0, np.pi, precision) # Elevation angles
    p = np.linspace(0, 2*np.pi, precision) # Azimuth angles
    
    theta, phi = np.meshgrid(t, p)
    
    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(theta) * np.sin(phi)
    Z = np.cos(theta) 
    
    num_x = int(np.sqrt(phased_array.size))
    num_y = int(np.sqrt(phased_array.size))
    num_z = int(1)
    
    af = 0
    for x in range(num_x):
        for y in range(num_y):
            for z in range(num_z):
                #af = (w[num_v, num_h] * np.exp(-1j * k * d * ((num_v * u) + (num_h * v)))) + af
                delay = (x * X) + (y * Y) + (z * Z)
                af = af + (w[x,y,z] * np.exp(-1j * k * d * delay))
                #af = np.exp(-1j * k * d * n * u) * np.exp(-1j * k * d * m * v) + af
    ##af = np.abs(af) 
    X = af * X
    Y = af * Y
    Z = af * Z 
    #z = af * z
    #z = np.abs(normalized_af * z)
    #z = normalized_af * z
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    #ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
    ax.plot_surface(X, Y, Z, color='b')
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),linewidth=0, antialiased=False, alpha=0.5)

    ax.view_init(45, 45)
    #ax.plot_surface(theta, phi, af, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title(title);
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #plt.savefigfig('my3dplot.png')
    plt.show()

def plot_pattern(tx_array, w=None):
    dtheta = np.linspace(0, 2*np.pi, 360)
    dphi = np.linspace(0, 2*np.pi, 360)
    theta, phi = np.meshgrid(dtheta, dphi)
    
    X = np.sin(theta)*np.cos(phi)
    Y = np.sin(theta)*np.sin(phi)
    Z = np.cos(theta)
    
    # So, I chose put antenna elements at x-y axis
    tx_array_x = 1
    #tx_array_x = int(np.sqrt(tx_array.size))
    tx_array_y = 1
    #tx_array_y = int(np.sqrt(tx_array.size))
    #tx_array_z = 1 
    tx_array_z = int(tx_array.size)
    
    wavelength = tx_array.wave_length
    d = tx_array.element_spacing
    k = 2 * np.pi * (1/wavelength)
    
    af = 0 
    for x in range(tx_array_x):
       for y in range(tx_array_y):
           for z in range(tx_array_z):
               delay = (x * X) + (y * Y) + (z * Z)
               if w is None:
                   af = af + (np.exp(-1j * k * d * delay))
               else:
                   af = af + (w[x,y] * np.exp(1j * k * d * delay))
                   #af = af + (np.exp(1j * k * d * delay)) 
    
    af = np.abs(af)
    #X = af * X
    #Y = af * Y
    Z = af * Z
    
    # View it.
    if w is None:
        s = mlab.mesh(X, Y, Z)
    else:
        s = mlab.mesh(X, Y, Z)
        
    #h = plt.contourf(X, Y, Z)
    #plt.show()
    mlab.colorbar(orientation='vertical')
    mlab.axes()
    mlab.show()


def plot_cb_pattern(tx_array, cb):
    dtheta = np.linspace(0, 2*np.pi, 100)
    dphi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(dtheta, dphi)
    
    X = np.sin(theta)*np.cos(phi)
    Y = np.sin(theta)*np.sin(phi)
    Z = np.cos(theta)
    
    # So, I chose put antenna elements at x-y axis
    tx_array_x = int(np.sqrt(tx_array.size))
    tx_array_y = int(np.sqrt(tx_array.size))
    tx_array_z = 1 
    
    wavelength = tx_array.wave_length
    d = tx_array.element_spacing
    k = 2 * np.pi * (1/wavelength)
  

    range_colors = np.linspace(0.2,0.8,2*len(cb))
    mat_colors = np.random.choice(range_colors, (len(cb), 3))
    pallet_colors = [(l[0], l[1], l[2]) for l in mat_colors]
    count_color = 0

    for cw_id, cw in cb.items():
        w = np.array(cw).reshape(int(np.sqrt(tx_array.size)) , int(np.sqrt(tx_array.size)))
        #plot_pattern(tx_array, cw)
 
        af = 0 
        for x in range(tx_array_x):
           for y in range(tx_array_y):
               for z in range(tx_array_z):
                   delay = (x * X) + (y * Y) + (z * Z)
                   af = af + (w[x,y] * np.exp(1j * k * d * delay))
    
        af = np.abs(af)
        #X = af * X
        #Y = af * Y
        #Z = af * Z
        
        # View it.
        ##s = mlab.mesh(af*X, af*Y, af*Z, color=pallet_colors[count_color], opacity=1.0)
        plt.contourf(af*X, af*Y, af*Z)
        plt.show()
        count_color += 1
        #s = mlab.surf(X, Y, Z)
        
    ##mlab.colorbar(orientation='vertical')
    ##mlab.axes()
    ##mlab.show()

def plot_codebook(tx_array=None, cb=None, dim=2):
    theta = np.linspace(-np.pi/3, np.pi/3) 
    num_of_elements = 32
    distance = 0.5
    for t in theta:
        array_vector = 1/np.sqrt(num_of_elements) * np.exp(np.arange(num_of_elements) * 1j * 2 * np.pi * distance * np.sin(t)) 
    if tx_array is None:
        pass


if __name__ == '__main__':
    plot_codebook()
    
#    dtheta = np.linspace(0, 2*np.pi, 100)
#    dphi = np.linspace(0, 2*np.pi, 100)
#    theta, phi = np.meshgrid(dtheta, dphi)
#    
#    X = np.sin(theta)*np.cos(phi)
#    Y = np.sin(theta)*np.sin(phi)
#    Z = np.cos(theta)
#    
#    # So, I chose put antenna elements at x-y axis
#    tx_array_x = int(np.sqrt(tx_array.size))
#    tx_array_y = int(np.sqrt(tx_array.size))
#    tx_array_z = 1 
#    
#    wavelength = tx_array.wave_length
#    d = tx_array.element_spacing
#    k = 2 * np.pi * (1/wavelength)
#  
#
#    range_colors = np.linspace(0.2,0.8,2*len(cb))
#    mat_colors = np.random.choice(range_colors, (len(cb), 3))
#    pallet_colors = [(l[0], l[1], l[2]) for l in mat_colors]
#    count_color = 0
#
#    for cw_id, cw in cb.items():
#        w = np.array(cw).reshape(int(np.sqrt(tx_array.size)) , int(np.sqrt(tx_array.size)))
#        #plot_pattern(tx_array, cw)
# 
#        af = 0 
#        for x in range(tx_array_x):
#           for y in range(tx_array_y):
#               for z in range(tx_array_z):
#                   delay = (x * X) + (y * Y) + (z * Z)
#                   af = af + (w[x,y] * np.exp(1j * k * d * delay))
#    
#        af = np.abs(af)
#        #X = af * X
#        #Y = af * Y
#        #Z = af * Z
#        
#        # View it.
#        ##s = mlab.mesh(af*X, af*Y, af*Z, color=pallet_colors[count_color], opacity=1.0)
#        plt.contourf(af*X, af*Y, af*Z)
#        plt.show()
#        count_color += 1
#        #s = mlab.surf(X, Y, Z)
#        
#    ##mlab.colorbar(orientation='vertical')
#    ##mlab.axes()
#    ##mlab.show()
#
