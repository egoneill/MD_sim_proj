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
def Calculate_alpha(E_old, E_new, T):
    #Take the min(1,exp(-beta(Enew-Eold)))
    if 1 <= np.exp((1/T)*(-E_new + E_old)):
        return 1
    else:
        return np.exp((1/T)*(-E_new + E_old))
"""/////////////////////////////////////////////////////////////////////////"""




"""///////////////////////////////MAIN//////////////////////////////////////"""

#initialize plotting vectors
CV_plot_x = np.zeros(60)
CV_plot_y = np.zeros(60)
mag_plot_x = np.zeros(60)
mag_plot_y = np.zeros(60)


 # Assume we start in a configuration rm. We choose this config randomly. 
    # Call the functions from Ising_MC_functions.py provided on Canvas
pos,neighbors = jon.AssembleSquareLattice(20, 20)
tot_num_lattice,spins1 = jon.InitializeLattice(20,20)
l = 0
m = 0

#initial magnetization
sum_spins = sum(spins1)


#iterate over all temperatures
for T in np.arange(0.1,6.0,0.1):
    #set the values for J and the nearest neighbors
    J = 1
    nearest_neighbors = 4
    
    spins = spins1
   
    energy_plot_y = np.zeros(2000)
    
    #Calculate the total energy of the system
    energy = Calculate_E(tot_num_lattice, pos, neighbors, spins, nearest_neighbors, J)
    
    #initialize the vector for plotting energy and magnetization
    
    
    #monte-carlo steps and sampling time
    steps = 1000000
    sample_time = 500
    
    #initialize magnetization and energies
    magnetization = 0
    energy_avg = 0
    energy_squared_avg = 0
    count = 0
    
    k = 0
    
    #iterate over the chosen number of MonteCarlo steps
    for i in range(steps):
        #Generate a trial configuration r_n by changing one random spin
        v = random.randint(0, tot_num_lattice-1)
        
        #calculate the change in energy due to flipping one spin
        dE = Delta_E(tot_num_lattice, v, neighbors, spins, nearest_neighbors, J)
    
        #Calculate how likely we are to accept the new configuration
        alpha = Calculate_alpha(energy, (energy + dE), T)
        
        #generate a random number to decide if we accept the new config
        u = random.uniform(0, 1)
        
        #accept the new config if u < alpha
        if u < alpha:
            #permanantly change the spin and the total energy
            spins[v] = -spins[v]
            sum_spins = sum_spins + 2*(spins[v])
            energy = energy + dE
            
            
            
             #Sample the total energy only certain times
        if (i % sample_time == 0):
            #update the plotting vectors
            energy_plot_y[k] = energy
            k += 1
    
                
            
        #update the magnetization
        magnetization = magnetization + sum_spins
        
            
            
            
    #plot the results and show the final configurations
       
    
    #jon.PlotLatticeConfiguration(pos, spins, 'J=2')  
    avg_M = np.abs(magnetization/steps/tot_num_lattice)
    mag_plot_x[m] = T
    mag_plot_y[m] = avg_M
    
    Cv = (np.var(energy_plot_y[301:]))

    CV_plot_x[m] = T
    CV_plot_y[m] = Cv

    m += 1

    
#jon.PlotLatticeConfiguration(pos, spins, 'J=2')   

plt.figure(2)
plt.plot(mag_plot_x,mag_plot_y) 

plt.figure(3)
plt.plot(CV_plot_x,CV_plot_y) 
