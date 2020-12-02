import numpy as np
import rytools.cholla as ca
import matplotlib.pyplot as plt
import rytools.units

cgs = rytools.units.get_cgs()

out = {}
out['fail'] = False

########## Settings

#simDirectory = r'/data/groups/ramirez-ruiz/rcastroy/'
simDirectory = r'./'

#simDirectory += 'q1e3x3_512_LOWLMAX/'
#simDirectory += 'chollaTidesTest/'
#simDirectory += 'chollaNoRelax/'
#simDirectory += 'chollaNoRelaxSimple/'
#simDirectory += 'chollaNoRelaxSimpleRegularCFL/'
#simDirectory += 'chollaSimpleStrongDamp/'
#simDirectory += 'chollaStrongDamp/'
#simDirectory += 'chollaNoRelaxRegularCFL/'

##########

#Load first file just to get properties of the simulation
#ds = ca.ChollaDataset(simDirectory, 0, [])

#Load file with orbit information
data = np.loadtxt(simDirectory + 'orbit_evolution.log', skiprows = 1).T

#TODO: Make the table reading automatic
out['t'     ] = data[0, :]
out['xstar' ] = np.array([ data[ 1, :], data[ 2, :], data[ 3, :] ]).T
out['vstar' ] = np.array([ data[ 4, :], data[ 5, :], data[ 6, :] ]).T
out['xBH'   ] = np.array([ data[ 7, :], data[ 8, :], data[ 9, :] ]).T
out['vBH'   ] = np.array([ data[10, :], data[11, :], data[12, :] ]).T
out['xFrame'] = np.array([ data[13, :], data[14, :], data[15, :] ]).T
out['vFrame'] = np.array([ data[16, :], data[17, :], data[18, :] ]).T
out['aFrame'] = np.array([ data[19, :], data[20, :], data[21, :] ]).T

#Switch from local frame to global frame
out['xstarglobal'] = out['xstar'] + out['xFrame']

#Compute acceleration of the center of mass
accx = np.gradient(out['vstar'][:, 0], out['t'], edge_order = 2)
accy = np.gradient(out['vstar'][:, 1], out['t'], edge_order = 2)
accz = np.gradient(out['vstar'][:, 2], out['t'], edge_order = 2)

out['astar'] = np.array([ accx, accy, accz ]).T
out['astarglobal'] = out['astar'] + out['aFrame']

out['absacc'] = np.linalg.norm(out['astarglobal'], axis = 1)
out['r'] = np.linalg.norm(out['xstarglobal'] - out['xBH'], axis = 1)

q = 1e3
Mstar = 1.989e33
eta = - ( cgs['GNEWT'] * Mstar * q * pow(out['r'], 5) + out['absacc'] * pow(out['r'], 7) ) / ( 9 * cgs['GNEWT'] * cgs['GNEWT'] * Mstar * q * q )
print(eta)

fig, ax = plt.subplots()
ax.plot(out['t'][out['t'] <= 40000], eta[out['t'] <= 40000])
ax.set_xlim(0, 40000)
plt.show()
plt.close()
