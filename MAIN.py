import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
import random
import time

#np.set_printoptions(threshold = np.inf)

########################################################################
#                     GRID AND TEMPORAL PARAMETERS                     #
########################################################################

xmin, ymin, t0 = -5, -5, 0
xmax, ymax, T = 5, 5, 15
Nx, Ny, Nt = 101, 101, 51
dx, dy, dt = (xmax-xmin)/(Nx-1), (ymax-ymin)/(Ny-1), T/(Nt-1)
x, y, t = np.linspace(xmin, xmax, Nx), np.linspace(ymin, ymax, Ny), np.linspace(0,T,Nt)
X, Y = np.meshgrid(x, y, indexing="ij")

# CONTROL FUNCTION

Na_mod, Na_angle = 51, 52
a_mod = np.linspace(0, 1, Na_mod)
a_angle = np.linspace(0, 2*np.pi, Na_angle, endpoint=False)

def mask_island(X, Y, r=0.01):
	mask = (X**2+Y**2)<r**2
	X_island = X[mask]
	Y_island = Y[mask]
	xmid_island = np.mean(X_island)
	ymid_island = np.mean(Y_island)
	return mask, xmid_island, ymid_island

# MAIN PICTURE (TARGET, OBSTACLES AND SWIMMER FIRST POSITION)
mask_island, xmid_island, ymid_island = mask_island(X,Y)
X_island = X[mask_island]
Y_island = Y[mask_island]
x0 = random.uniform(xmin, xmax)
y0 = random.uniform(ymin, ymax)

########################################################################
#                            FUNCTIONS                                 #
########################################################################

def semi_lagrangian_god(V, x, Nx, y, Ny, Nt, dt, a_mod, a_angle, X = X, Y = Y, mask_island = mask_island, big_num = 1e20, lamb = 1, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax):
	driftx = drift(X, Y)[0]
	drifty = drift(X, Y)[1]
	for n in range(Nt-2, -1, -1):
		V[n, :, :] = V[n+1, :, :]
		V[n, mask_island] = 0
		xobj = X[:,:,np.newaxis, np.newaxis] + (a_mod[:, np.newaxis]*np.cos(a_angle))[np.newaxis, np.newaxis, :, :] + driftx
		yobj = Y[:,:,np.newaxis, np.newaxis] + (a_mod[:, np.newaxis]*np.sin(a_angle))[np.newaxis, np.newaxis, :, :] + drifty
		mask_borders = (xobj<xmin)+(xobj>xmax)+(yobj<ymin)+(yobj>ymax)
		index_x = x.searchsorted(xobj)
		index_x = np.where(index_x==Nx, Nx-1, index_x)
		index_y = y.searchsorted(yobj)
		index_y = np.where(index_y==Ny, Ny-1, index_y)
		(x1,y1,x2,y2) = (X[index_x-1,index_y-1],Y[index_x-1,index_y-1],X[index_x,index_y],Y[index_x,index_y])
		(w11,w12,w21,w22) = bilinear_interpolation(xobj,yobj,x1,y1,x2,y2)
		V_interp = w11*V[n+1 ,index_x-1, index_y-1]+w21*V[n+1 ,index_x, index_y-1]+w12*V[n+1 ,index_x-1, index_y]+w22*V[n+1 ,index_x, index_y]
		value = dt*running_cost(xobj, yobj, n*dt, a_mod)+np.exp(-lamb*dt)*V_interp
		value[mask_borders] = big_num
		value0 = np.min(value, axis=(2,3))
		V[n] = value0*np.invert(mask_island)
	return V

def semi_lagrangian_swimmer(V, x0, y0, x, y, Nt, dt, a_mod, Na_mod, a_angle, Na_angle, big_num = 1e3, lamb = 0):
	x_swimmer = x0
	y_swimmer = y0
	x_swimmer_array = np.array([x_swimmer])
	y_swimmer_array = np.array([y_swimmer])
	for n in range(Nt):
		value0 = big_num
		driftx = drift(x_swimmer, y_swimmer)[0]
		drifty = drift(x_swimmer, y_swimmer)[1]
		for m in range(Na_mod):
			for p in range(Na_angle):
				a_m = a_mod[m]
				phi = a_angle[p]
				xobj = x_swimmer + a_m*np.cos(phi) + driftx
				yobj = y_swimmer + a_m*np.sin(phi) + drifty
				if (xobj<X[0][0])or(xobj>X[-1][-1])or(yobj<Y[0][0])or(yobj>Y[-1][-1]):
							continue
				else:		
					index_x = x.searchsorted(xobj)
					index_y = y.searchsorted(yobj)
					(x1,y1,x2,y2) = (X[index_x-1,index_y-1],Y[index_x-1,index_y-1],X[index_x,index_y],Y[index_x,index_y])
					(w11,w12,w21,w22) = bilinear_interpolation(xobj,yobj,x1,y1,x2,y2)
					V_interp = w11*V[n ,index_x-1, index_y-1]+w21*V[n ,index_x, index_y-1]+w12*V[n ,index_x-1, index_y]+w22*V[n ,index_x, index_y]
					value = dt*running_cost(xobj, yobj, n*dt, a_m)+np.exp(-lamb*dt)*V_interp
					if value < value0:
						value0 = value
						mod_star = a_m
						ang_star = phi
		x_swimmer = x_swimmer+dt*(mod_star*np.cos(ang_star)+driftx)
		y_swimmer = y_swimmer+dt*(mod_star*np.sin(ang_star)+drifty)
		x_swimmer_array = np.append(x_swimmer_array, x_swimmer)
		y_swimmer_array = np.append(y_swimmer_array, y_swimmer)
	return x_swimmer_array, y_swimmer_array

def bilinear_interpolation(x,y,x1,y1,x2,y2):
	full_area = (x2-x1)*(y2-y1)
	(a,b,c,d) = (x2-x,y2-y,y-y1,x-x1)
	(w11,w12,w21,w22) = (a*b,a*c,d*b,d*c)/(full_area)
	return (w11,w12,w21,w22)

def exit_cost(V, X, Y, big_num = 1e3, mask = mask_island):
	V[-1,mask] = 0
	V[-1, np.invert(mask)] = big_num/3
	return V

def running_cost(x, y, t, a, w_target=1/4, w_t=1/4, w_a=1/4, x0 = xmid_island, y0 = ymid_island):
	return w_target*np.sqrt((x-x0)**2+(y-y0)**2)

def drift(x, y, mod = 0):
	return [-mod/np.sqrt(2),-mod/np.sqrt(2)]

########################################################################
#                            MAIN CODE                                 #
########################################################################

V = np.zeros((Nt, Nx, Ny))
V = exit_cost(V, X, Y)
start_time = time.time()
V = semi_lagrangian_god(V, x, Nx, y, Ny, Nt, dt, a_mod, a_angle)
end_time = time.time()
print("Computing time for S-L V calculation", end_time-start_time)
x_swimmer_array, y_swimmer_array = semi_lagrangian_swimmer(V, x0, y0, x, y, Nt, dt, a_mod, Na_mod, a_angle, Na_angle, big_num = 1e3, lamb = 1)

########################################################################
#                             GRAPHS                                   #
########################################################################

plt.style.use('dark_background')
fig, ax = plt.subplots()
plt.axis('equal')
ax.plot(X,Y, "k,")
ax.plot(X_island, Y_island, "ys")
line, = ax.plot(x0, y0, "mo", label="Nadador")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
def update(frame):
	line.set_data((x_swimmer_array[frame], y_swimmer_array[frame]))
	ax.contourf(X, Y, V[frame], cmap="jet")
	return ax, line
ani = animation.FuncAnimation(fig=fig, func=update, frames=Nt, interval=100, repeat=False)
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'RESULTS/')
sample_file_name = "main.mp4"
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
ani.save(results_dir + sample_file_name, fps=10)
plt.show()

#PREVIOUS PLOT
"""
plt.style.use('dark_background')
fig, ax = plt.subplots()
ax.plot(X,Y, "k,")
ax.plot(X_island, Y_island, "ys")
line, = ax.plot(x0, y0, "mo", label="Nadador")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
for n in range(Nt):
	line.set_data((x_swimmer_array[n], y_swimmer_array[n]))
	ax.contourf(X, Y, V[n], cmap="jet")
	plt.pause(0.1)
plt.show()
"""