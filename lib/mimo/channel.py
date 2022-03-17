# Define o modelo de canal a ser utilizado
import numpy as np
from numpy import matmul
import matplotlib.pyplot as plt


def calc_omega(theta, phi, k, d):
    omegax = k * d * np.sin(theta) * np.cos(phi)
    omegay = k * d * np.sin(theta) * np.sin(phi)
    omegaz = k * d * np.cos(theta)
    return omegax, omegay, omegaz

def scatteringchnmtxfromtextfiles(paths, tx_transceiver, rx_transceiver, tx_array, rx_array):
    """
    Estima o canal ataves atraves do modelo geometrico narrowband utilizando os dados de ray-tracing, utilizando angulo de elevacao e azimute. Retorna a matriz de canal.
    """
    tx_size = tx_array.size
    rx_size = rx_array.size

    #h = np.zeros((tx_size, rx_size), dtype=complex) # define a matriz de canal com tamanho num_tr por num_rx.
    h = np.zeros((tx_size), dtype=complex) # define a matriz de canal com tamanho num_tr por num_rx.
    #factor = np.sqrt(tx_size * rx_size) * (1/len(paths))
    #factor = np.sqrt(tx_size) * (1/len(paths))
    factor = (1/len(paths)) # * (1/tx_size)

    tx_spacing = tx_array.element_spacing
    tx_wavelength = tx_array.wave_length

    rx_spacing = rx_array.element_spacing
    rx_wavelength = rx_array.wave_length
 
    if (tx_array.formfactory=="UPA" and rx_array.formfactory=="UPA"):
        pass

    if (tx_array.formfactory=="ULA" and rx_array.formfactory=="ULA"):
        for n in range(len(paths)):

            departure_theta = np.deg2rad(float(paths[n]['departure_theta']))
            arrival_theta = np.deg2rad(float(paths[n]['arrival_theta']))

            tx_array_vec = np.arange(tx_size)
            rx_array_vec = np.arange(rx_size)

            k_t = 2 * np.pi * (1/tx_wavelength)
            k_r = 2 * np.pi * (1/rx_wavelength)
            psi_t = k_t * tx_spacing * np.cos(departure_theta)
            psi_r = k_r * rx_spacing * np.cos(arrival_theta)
            #steering_vec = (1/np.sqrt(tx_size)) * np.array([np.exp(-1j * n * psi_t) for n in tx_array_vec])
            steering_vec = np.array([np.exp(-1j * n * psi_t) for n in tx_array_vec])
            receive_vec = (1/np.sqrt(rx_size)) * np.array([np.exp(-1j * n * psi_r) for n in rx_array_vec])
            
            complex_gain = float(rx_transceiver['pathgain']) * np.exp(-1j * psi_t)
            #print('complex_gain:\n', complex_gain)
            h = h + steering_vec
    return  factor * h
    #return  h
def scatteringchnmtxfromhdf5files(paths, rx_array, tx_array):
    """
    Input: paths from RT and device information of TX and RX.
    Output: channel matrix w\ Nrx by Ntx shape. It is get throught geometric channel design.
    """

    rx_size = rx_array.size #
    tx_size = tx_array.size # size is the number of elements in the transceivers

    h = np.zeros((rx_size, tx_size), dtype=complex)

    rx_spacing = rx_array.element_spacing
    tx_spacing = tx_array.element_spacing

    rx_wavelength = rx_array.wave_length
    tx_wavelength = tx_array.wave_length

    k_rx = 2 * np.pi * (1/rx_wavelength)
    k_tx = 2 * np.pi * (1/tx_wavelength)

    # at receiving process...
    aoa_theta = np.array([float(p['arrival_theta']) for p in paths])# theta is elevation angle
    aoa_phi = np.array([float(p['arrival_phi']) for p in paths]) # phi is azimuth angle

    # at transmition process...
    aod_theta = np.array([float(p['departure_theta']) for p in paths])# 
    aod_phi = np.array([float(p['departure_phi']) for p in paths])
    #print (f'len(aoa_theta): {len(aoa_theta)}')
    #print (f'len(aoa_phi): {len(aoa_theta)}')
    #print (f'len(aod_theta): {len(aod_theta)}')
    #print (f'len(aod_phi): {len(aod_phi)}')
    
    # from each path
    #complex_gain = np.array([float(p['received_power'])*np.exp(1j * 0) for p in paths]) # 
    complex_gain = np.array([float(p['received_power'])*np.exp(1j * p['phase']) for p in paths])
 
    factor = np.sqrt(rx_size * tx_size) #
 
    if (rx_array.formfactory=="UPA" and tx_array.formfactory=="UPA"):
        arrival_omegax, arrival_omegay, arrival_omegaz = calc_omega(aoa_theta, aoa_phi, k_rx, rx_spacing)
        departure_omegax, departure_omegay, departure_omegaz = calc_omega(aod_theta, aod_phi, k_tx, tx_spacing)

        # @ RX
        num_rx_x = int(np.sqrt(1)) # number of element of rx at x_axis
        num_rx_y = int(np.sqrt(rx_size)) # number of element of rx at y_axis
        num_rx_z = int(np.sqrt(rx_size)) # number of element of rx at z_axis
        # @ TX
        num_tx_x = int(np.sqrt(1)) # number of element of tx at x_axis
        num_tx_y = int(np.sqrt(tx_size)) # number of element of tx at y_axis
        num_tx_z = int(np.sqrt(tx_size)) # number of element of tx at z_axis


        # Stimating the channel..
        #complex_gain = [p['received_power']*np.exp(1j * p['phase']) for p in paths] 
        #complex_gain = complex_gain/np.sqrt(np.sum(complex_gain.conj() * complex_gain))
#        received_power = np.array([p['received_power'] for p in paths])
#        received_power = received_power/np.sqrt(np.sum(received_power * received_power))

        #complex_gain = np.array([received_pwr*np.exp(1j * 0) for received_pwr in received_power]) 
        #complex_gain = np.array([p['received_power']*np.exp(1j * 0) for p in paths]) 
        #factor = np.sqrt(rx_size * tx_size) #
        #factor = np.sqrt(rx_array.size * tx_array.size) * (1/len(paths))  * (np.linalg.norm(complex_gain)/np.sum(complex_gain))

        for n in range(len(paths)):
       
            # @ RX
            rx_vecx = np.exp(1j * arrival_omegay[n] * np.arange(num_rx_z))
            rx_vecy = np.exp(1j * arrival_omegax[n] * np.arange(num_rx_y))
            ar = (1/np.sqrt(num_rx_z * num_rx_y)) * np.kron(rx_vecy, rx_vecx)
            #ar = np.kron(rx_vecy, rx_vecx)
            ar = np.array(ar).reshape(rx_size,1)
            # @ TX
            tx_vecx = np.exp(1j * departure_omegay[n] * np.arange(num_tx_z))
            tx_vecy = np.exp(1j * departure_omegax[n] * np.arange(num_tx_y))
            at = (1/np.sqrt(num_tx_z * num_tx_y)) * np.kron(tx_vecy, tx_vecx)
            #at = np.kron(tx_vecy, tx_vecx)
            at = np.array(at).reshape(tx_size,1)
            #print (f'ar.shape: {ar.shape}') 
            #print (f'at.shape: {at.shape}') 
            # Channel contrib of this path:
            #h_contrib = np.matmul(ar.conj().T, at)
            h_contrib = ar * at.conj().T
        
            #h = h + complex_gain[n] * h_contrib
            h = h + complex_gain[n] * h_contrib
        
        h = factor * h
        #print (f'h.shape: {h.shape}')

    #print (f'===============================')
    if (tx_array.formfactory=="ULA" and rx_array.formfactory=="ULA"):
        for n in range(len(paths)):

            #departure_theta = float(paths[n]['departure_theta'])
            #print ('departure_theta(deg): ', np.rad2deg(departure_theta))
            #arrival_theta = float(paths[n]['arrival_theta'])
            #print ('arrival_theta(deg): ', np.rad2deg(arrival_theta))

            tx_array_vec = np.arange(tx_size)
            rx_array_vec = np.arange(rx_size)

            at = np.array([np.exp(-1j * 2 * np.pi * k * tx_spacing * (1/tx_wavelength) * np.cos(aod_theta)) for k in range(len(tx_array_vec))])
            at = at * (1/np.sqrt(len(tx_array_vec)))
            at = np.matrix(at)

            ar = np.array([np.exp(-1j * 2 * np.pi * k * rx_spacing * (1/rx_wavelength) * np.cos(aoa_theta)) for k in range(len(rx_array_vec))])
            ar = ar * (1/np.sqrt(len(rx_array_vec)))
            ar = np.matrix(ar)

            #complex_gain = pathsdd[n]['received_power'] * np.exp(-1j * np.deg2rad(np.random.choice(theta)))

            outer_product = ar * at.conj().T
            h = h + (complex_gain[n] * outer_product)
        h = h * factor
    return  h


def richscatteringchnmtx(num_tx, num_rx):
    """
    Ergodic channel. Fast, frequence non-selective channel: y_n = H_n x_n + z_n.  
    Narrowband, MIMO channel
    PDF model: Rich Scattering
    Circurly Simmetric Complex Gaussian from: 
         https://www.researchgate.net/post/How_can_I_generate_circularly_symmetric_complex_gaussian_CSCG_noise
    """
    sigma = 1
    #my_seed = 2323
    #np.random.seed(my_seed)
    h = np.sqrt(sigma/2)*(np.random.randn(num_rx, num_tx) + np.random.randn(num_rx, num_tx) * 1j)
    #h = np.sqrt(sigma/2)*np.random.randn(num_tx, num_rx)
    return h



def plot_paths(chn_paths):
    print('chn_paths:\n', chn_paths)
 
    departure_theta = []
    arrival_theta = []
    for chn_path in chn_paths:
        departure_theta.append(float(chn_path['departure_theta']))
        arrival_theta.append(float(chn_path['arrival_theta']))
 
    fig = plt.figure()
    fig.add_subplot(111, projection='polar')
    plt.title('Tx ' + ' DoD paths')
    for t in departure_theta:
        plt.polar((0, t), (0, 1))
    #plt.title(str(id_scene))
    plt.show()

#def get_channel(paths, tx_array, rx_array):
##class path:
##    def __init__(self, id, aod):
##        self.id = id
##        self.aod = np.deg2rad(aod)
##        self.at = None
#
#
#class Path:
#    def __init__(self, id, aod, aoa, received_power):
#        self.id = str(id)
#        self.aod = np.deg2rad(aod)
#        self.aoa = np.deg2rad(aoa)
#        self.received_power = np.power(10,received_power/10) # convert dB in W
#        self.at = None
# 
#    def to_string(self):
#        print ('id: ', self.id)
#        print ('aod: ', np.rad2deg(self.aod))
#        print ('aoa: ', np.rad2deg(self.aoa))
#        print ('received_power: ', self.received_power)
# 
##def array_factor(p, array_vec):
##    factor = 1/len(array_vec) # no cado do uso pra estimar o canal considerando TX e RX, usar 1/sqrt(len(array_vec).
##    af = np.array([np.exp(-1j *  n * p) for n in array_vec])
##    #af = np.array([np.exp(-1j * n * p) for n in tx_array])
##    return factor * af
#
#
#theta = np.arange(0,180)
#num_paths = 2
##my_seed = np.random.choice(np.arange(100000))
#my_seed = 633
#np.random.seed(my_seed)
#print ('my_seed: \n', my_seed)
#paths = []
#for n in range(num_paths):
#    aod = np.random.choice(theta)
#    aoa = np.random.choice(theta)
#    received_power = np.random.choice([0.000001, 0.01, 1.0])
#    p = Path(n, aod, aoa, received_power)
#    paths.append(p)
#
#num_tx_elements = 4
#num_rx_elements = 4
#
#fc = 60 * (10 ** 9) #60 GHz
#wavelength = c/fc #in meters
#element_spacing = wavelength/2 #in meters
#tx_array_vec = np.arange(num_tx_elements)
#rx_array_vec = np.arange(num_rx_elements)
#k = 2 * np.pi * 1/wavelength
#at_res = np.zeros(len(tx_array_vec))
#
#
#h = np.zeros((num_rx_elements, num_tx_elements))
#received_power_list = [p.received_power for p in paths]
#factor = np.sqrt(num_rx_elements * num_tx_elements) * np.linalg.norm(received_power_list)/np.sum(received_power_list)
#
#for p in paths:
#
#    print('path_id: \n', p.id)
#    aod = p.aod
#    aoa = p.aoa
#    print ('aod: ', aod)
#    print ('aoa: ', aoa)
#    at = np.array([np.exp(-1j * 2 * np.pi * n * element_spacing * (1/wavelength) * np.cos(aod)) for n in range(len(tx_array_vec))])
#    ar = np.array([np.exp(-1j * 2 * np.pi * n * element_spacing * (1/wavelength) * np.cos(aoa)) for n in range(len(rx_array_vec))])
#    complex_gain = p.received_power * np.exp(-1j * np.deg2rad(np.random.choice(theta)))
#    print ('abs.complex_gain: \n', np.abs(complex_gain))
#    at_res = at_res + (at * complex_gain)
#    p.at = at
#
#    outer_product = np.outer(ar.conj().T, at)
#    h = h + (complex_gain * outer_product)
#at_res = at_res * factor
#h = h * factor
#
#
#p_res = Path('p_res', 0, 0, 0)
#p_res.at = at_res
#paths.append(p_res)
#
#print ('h', h)
#u, s, vh = svd(h)
#print ('u', u)
#print ('s', s)
#print ('vh', vh)
#print ('vh[0]', vh[0])
#w1 = vh[0]
#print ('w1', w1)
#w2 = p_res.at
#print ('w2', w2)
#p_svd = Path('p_svd', 0, 0, 0)
#p_svd.at = w1
#paths.append(p_svd)
#
#
#
#for p in paths:
#    at = p.at
#    prod = np.zeros(len(theta), dtype=complex)
#    for i in range(len(theta)):
#        t = np.deg2rad(theta[i])
#        af = np.array([np.exp(1j * 2 * np.pi * n * element_spacing * (1/wavelength) * np.cos(t)) for n in range(len(tx_array_vec))])
#        product = np.matmul(at.T, af)
#        product = (np.abs(product) ** 2)/np.abs(np.matmul(at.conj().T, at))#print ('product: \n', product)
#        prod[i] = product/len(tx_array_vec)
#    plt.plot(theta, prod, label=p.id)
#    plt.legend()
##plt.savefig('steering.png')
#plt.show()
