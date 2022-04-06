import json
import numpy as np
import sys
import matplotlib.pyplot as plt



if __name__ == "__main__":
    pass
    pathfile_json = sys.argv[1]
    with open(pathfile_json) as result:
        result_data = result.read()
        result_d = json.loads(result_data)
    print (result_d.keys())

    snr_db = np.array(result_d['snr_db'])
    capacity_opt_egt_mean = np.array(result_d['capacity_opt_egt_mean'])
    capacity_opt_egt_b1_mean = np.array(result_d['capacity_opt_egt_b1_mean'])
    capacity_opt_egt_b2_mean = np.array(result_d['capacity_opt_egt_b2_mean'])
    capacity_opt_egt_b3_mean = np.array(result_d['capacity_opt_egt_b3_mean'])
    capacity_opt_egt_b4_mean = np.array(result_d['capacity_opt_egt_b4_mean'])

    capacity_est_egt_mean = np.array(result_d['capacity_est_egt_mean'])
    capacity_est_egt_b1_mean = np.array(result_d['capacity_est_egt_b1_mean'])
    capacity_est_egt_b2_mean = np.array(result_d['capacity_est_egt_b2_mean'])
    capacity_est_egt_b3_mean = np.array(result_d['capacity_est_egt_b3_mean'])
    capacity_est_egt_b4_mean = np.array(result_d['capacity_est_egt_b4_mean'])
    
    print (f'mean opt_egt - est_egt: {np.mean( capacity_opt_egt_mean - capacity_est_egt_mean  )}')
    print (f'mean est_egt - est_egt_b1: {np.mean( capacity_est_egt_mean - capacity_est_egt_b1_mean )}')

    x = snr_db
    #y = capacity_opt_egt_mean
    #plt.plot(x , y, marker='p')

    xvals = np.linspace(x[0], x[-1], 20000)

    capacity_opt_egt_mean_interp = np.interp(xvals, x, capacity_opt_egt_mean)
    capacity_opt_egt_b1_mean_interp = np.interp(xvals, x, capacity_opt_egt_b1_mean)
    capacity_opt_egt_b2_mean_interp = np.interp(xvals, x, capacity_opt_egt_b2_mean)
    capacity_opt_egt_b3_mean_interp = np.interp(xvals, x, capacity_opt_egt_b3_mean)
    capacity_opt_egt_b4_mean_interp = np.interp(xvals, x, capacity_opt_egt_b4_mean)

    capacity_est_egt_mean_interp = np.interp(xvals, x, capacity_est_egt_mean)
    capacity_est_egt_b1_mean_interp = np.interp(xvals, x, capacity_est_egt_b1_mean)
    capacity_est_egt_b2_mean_interp = np.interp(xvals, x, capacity_est_egt_b2_mean)
    capacity_est_egt_b3_mean_interp = np.interp(xvals, x, capacity_est_egt_b3_mean)
    capacity_est_egt_b4_mean_interp = np.interp(xvals, x, capacity_est_egt_b4_mean)

    #plt.plot(xvals, capacity_opt_egt_mean_interp, marker='*')
    #plt.plot(xvals, capacity_est_egt_mean_interp, marker='*')
    #plt.plot(xvals, capacity_est_egt_b1_mean_interp, marker='*')

    #plt.show()
    

    
    xmin = 3 #capacity_opt_egt_mean_interp[100]
    print (xmin)
    xmax = 10 #capacity_est_egt_b1_mean_interp[-100]
    print (xmax)

    cvals = np.linspace(xmin, xmax, 100)
    x_opt = []
    x_opt_b1 = []
    x_opt_b2 = []
    x_opt_b3 = []
    x_opt_b4 = []

    x_est = []
    x_est_b1 = []
    x_est_b2 = []
    x_est_b3 = []
    x_est_b4 = []



    precision = 3
    
    for cval in cvals:
        #print ('----')
        #print (cval)
        for n in range(len(xvals)):
            if (round(cval, precision) == round(capacity_opt_egt_mean_interp[n], precision)):
                 #print (xvals[n])
                 #x_opt.append([round(cval,3), xvals[n]])
                 x_opt.append(xvals[n])
                 break


    for cval in cvals:
        #print ('----')
        #print (cval)
        for n in range(len(xvals)):


            if (round(cval,precision) == round(capacity_opt_egt_b1_mean_interp[n], precision)):
                 #print (xvals[n])
                 #x_est_b1.append([round(cval,3), xvals[n]])
                 x_opt_b1.append(xvals[n])
                 break

    for cval in cvals:
        #print ('----')
        #print (cval)
        for n in range(len(xvals)):


            if (round(cval, precision) == round(capacity_opt_egt_b2_mean_interp[n], precision)):
                 #print (xvals[n])
                 #x_est_b1.append([round(cval,3), xvals[n]])
                 x_opt_b2.append(xvals[n])
                 break


    for cval in cvals:
        #print ('----')
        #print (cval)
        for n in range(len(xvals)):


            if (round(cval, precision) == round(capacity_opt_egt_b3_mean_interp[n], precision)):
                 #print (xvals[n])
                 #x_est_b1.append([round(cval,3), xvals[n]])
                 x_opt_b3.append(xvals[n])
                 break


    for cval in cvals:
        #print ('----')
        #print (cval)
        for n in range(len(xvals)):


            if (round(cval, precision) == round(capacity_opt_egt_b4_mean_interp[n], precision)):
                 #print (xvals[n])
                 #x_est_b1.append([round(cval,3), xvals[n]])
                 x_opt_b4.append(xvals[n])
                 break






    for cval in cvals:
        #print ('----')
        #print (cval)
        for n in range(len(xvals)):

            if (round(cval, precision) == round(capacity_est_egt_mean_interp[n], precision)):
                 #print (xvals[n])
                 #x_est.append([round(cval,3), xvals[n]])
                 x_est.append(xvals[n])
                 break


    for cval in cvals:
        #print ('----')
        #print (cval)
        for n in range(len(xvals)):


            if (round(cval,precision) == round(capacity_est_egt_b1_mean_interp[n], precision)):
                 #print (xvals[n])
                 #x_est_b1.append([round(cval,3), xvals[n]])
                 x_est_b1.append(xvals[n])
                 break

    for cval in cvals:
        #print ('----')
        #print (cval)
        for n in range(len(xvals)):


            if (round(cval, precision) == round(capacity_est_egt_b2_mean_interp[n], precision)):
                 #print (xvals[n])
                 #x_est_b1.append([round(cval,3), xvals[n]])
                 x_est_b2.append(xvals[n])
                 break


    for cval in cvals:
        #print ('----')
        #print (cval)
        for n in range(len(xvals)):


            if (round(cval, precision) == round(capacity_est_egt_b3_mean_interp[n], precision)):
                 #print (xvals[n])
                 #x_est_b1.append([round(cval,3), xvals[n]])
                 x_est_b3.append(xvals[n])
                 break


    for cval in cvals:
        #print ('----')
        #print (cval)
        for n in range(len(xvals)):


            if (round(cval, precision) == round(capacity_est_egt_b4_mean_interp[n], precision)):
                 #print (xvals[n])
                 #x_est_b1.append([round(cval,3), xvals[n]])
                 x_est_b4.append(xvals[n])
                 break


    x_opt = np.array(x_opt)
    x_opt_b1 = np.array(x_opt_b1)
    x_opt_b2 = np.array(x_opt_b2)
    x_opt_b3 = np.array(x_opt_b3)
    x_opt_b4 = np.array(x_opt_b4)

    x_est = np.array(x_est)
    x_est_b1 = np.array(x_est_b1)
    x_est_b2 = np.array(x_est_b2)
    x_est_b3 = np.array(x_est_b3)
    x_est_b4 = np.array(x_est_b4)

    print ('----')
    print (len(x_opt))
    print (len(x_opt_b1))
    print (len(x_opt_b2))
    print (len(x_opt_b3))
    print (len(x_opt_b4))
 
    print (len(x_est))
    print (len(x_est_b1))
    print (len(x_est_b2))
    print (len(x_est_b3))
    print (len(x_est_b4))
    

    print (f'mean opt_egt - opt_egt_b1 (dB): {np.mean( np.sqrt( (x_opt - x_opt_b1) ** 2  ) )}')
    print (f'mean opt_egt - opt_egt_b2 (dB): {np.mean( np.sqrt( (x_opt - x_opt_b2) ** 2  ) )}')
    print (f'mean opt_egt - opt_egt_b3 (dB): {np.mean( np.sqrt( (x_opt - x_opt_b3) ** 2  ) )}')
    print (f'mean opt_egt - opt_egt_b4 (dB): {np.mean( np.sqrt( (x_opt - x_opt_b4) ** 2  ) )}')

    print (f'mean opt_egt - est_egt (dB): {np.mean( np.sqrt( (x_opt - x_est) ** 2 )  )}')

    print (f'mean est_egt - est_egt_b1 (dB): {np.mean( np.sqrt( (x_est - x_est_b1) ** 2  ) )}')
    print (f'mean est_egt - est_egt_b2 (dB): {np.mean( np.sqrt( (x_est - x_est_b2) ** 2  ) )}')
    print (f'mean est_egt - est_egt_b3 (dB): {np.mean( np.sqrt( (x_est - x_est_b3) ** 2  ) )}')
    print (f'mean est_egt - est_egt_b4 (dB): {np.mean( np.sqrt( (x_est - x_est_b4) ** 2  ) )}')
