# -*- coding: utf-8 -*-
"""
Eric O'Neill 2018 CBE 710 simulation project
"""
#import needed libraries
import Ising_MC_functions as jon
import numpy as np
import random
import matplotlib.pyplot as plt

"""///////////////////////////////FUNCTIONS/////////////////////////////////"""

#this function calculates the total energy of a lattice.
#it assumes there is no external magnetic field H=0.
# E = -sum_i^N mu*H*s_i - (J/2)sum_i^N sum_j^` s_i*s_j
def Calculate_E(tot_num_lattice, pos, neighbors, spins, nearest_neighbors, J):
    #set the energy J, nearest neighbor numbers, and initialize the energy
    energy = 0
    #for each point on the lattice 
    for i in range(tot_num_lattice):
        #iterate over each points nearest neighbors to calculate energies
        for j in range(nearest_neighbors):
            #spin of the current lattice site times the spin of its neighbor
            #for all of the nearest neighbors
            energy += spins[i]*spins[neighbors[i,j]]
    #the final energy is scaled by J and divided by 2 to exclude double counting
    energy = energy * -J/2
    return energy





# this function calculates the change in energy by changing the orientation of 
# one spin
def Delta_E(tot_num_lattice, site_index, neighbors, spins, nearest_neighbors, J):
    #initialize the change in energy due to change in spin
    dE = 0
    #loop over the lattice site's nearest neighbors
    for j in range(nearest_neighbors):
        # for each nearest neighbor, flip the spin s_i and calculate 
        # the energy contribution due to that site and its neighbors
        dE = dE - spins[site_index]*spins[neighbors[site_index,j]]
    # The change in energy due to flipping one spin is dE = Enew - Eold
    # Eold = -Enew in our case so dE = 2*Enew    
    dE = -J * dE * 2
    return dE





# this function calculates the probability of moving to the new state given 
# the old energy and the new energy
def Calculate_alpha(E_old, E_new):
    #Take the min(1,exp(-beta(Enew-Eold)))
    if 1 <= np.exp(-E_new + E_old):
        return 1
    else:
        return np.exp(-E_new + E_old)
"""/////////////////////////////////////////////////////////////////////////"""




"""///////////////////////////////MAIN//////////////////////////////////////"""

#set the values for J and the nearest neighbors
J = 2
nearest_neighbors = 4

# Assume we start in a configuration rm. We choose this config randomly. 
# Call the functions from Ising_MC_functions.py provided on Canvas
pos,neighbors = jon.AssembleSquareLattice(20, 20)
tot_num_lattice,spins = jon.InitializeLattice(20,20)

#Calculate the total energy of the system
energy = Calculate_E(tot_num_lattice, pos, neighbors, spins, nearest_neighbors, J)

#initialize the vector for plotting energy and magnetization
energy_plot_x = [0]
energy_plot_y = [energy/tot_num_lattice]
mag_plot_x = [0]
mag_plot_y = [sum(spins)/tot_num_lattice]

#monte-carlo steps and sampling time
steps = 500000
sample_time = 500

magnetization = 0
energy_avg = 0
energy_squared_avg = 0
count = 0

#iterate over the chosen number of MonteCarlo steps
for i in range(steps):
    #Generate a trial configuration r_n by changing one random spin
    v = random.randint(0, tot_num_lattice-1)
    
    #calculate the change in energy due to flipping one spin
    dE = Delta_E(tot_num_lattice, v, neighbors, spins, nearest_neighbors, J)

    #Calculate how likely we are to accept the new configuration
    alpha = Calculate_alpha(energy, (energy + dE))
    
    #generate a random number to decide if we accept the new config
    u = random.uniform(0, 1)
    
    #accept the new config if u < alpha
    if u < alpha:
        #permanantly change the spin and the total energy
        spins[v] = -spins[v]
        energy = energy + dE
        
        
    #Sample the total energy only certain times
    if (i % sample_time == 0):
        #update the plotting vectors
        energy_plot_x.append(i)
        energy_plot_y.append(energy)
        mag_plot_x.append(i)
        mag_plot_y.append(sum(spins)/tot_num_lattice)
        if i >= 150000:
            energy_avg += energy
            energy_squared_avg += energy*energy
            count = count+1
        
    magnetization = magnetization + sum(spins)
    
    
        
        
        
#plot the results and show the final configurations
   
avg_M = magnetization/steps/tot_num_lattice   

avg_E = energy_avg/count
avg_E2 = energy_squared_avg/count

Cv = (avg_E2 - (avg_E*avg_E))


Cv2 = (np.var(energy_plot_y[301:]))

Cv3 = (np.var(energy_plot_y))

print('average magnetization =' , avg_M)

print('Heat Capacity', Cv)

print('HC CV' , Cv2)
    
print('HC CV' , Cv3)
jon.PlotLatticeConfiguration(pos, spins, 'J=2')   

plt.figure(2)
plt.scatter(energy_plot_x,energy_plot_y) 

plt.figure(3)
plt.scatter(mag_plot_x,mag_plot_y) 
