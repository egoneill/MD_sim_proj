#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 13:54:47 2018

@author: oneill
"""
import MD_functions as MD
import numpy as np
import matplotlib.pyplot as plt
import time
import random



#this function calculates the forces between all particles and returns an 
#nxnx3 array of the forces and the potential energy of the system. find the force
#experienced by particle i by summing the ith row of the array
def CalculateForces2(pos, epsilon, sigma, num_particles, L):
    
    #initialize force vector, potential energy, vector of x,y,z distances
    #and the actual distance
    forces = np.zeros([num_particles, num_particles, 3])
    pot_energy = 0
    delxyz = np.zeros([num_particles,num_particles, 3])
    rij = np.zeros([num_particles,num_particles])
            
    
    #use broadcasting arrays to make an nxnx3 array containing the x-x, y-y, z-z distances
    #for all pairs of particles, this also implements the minimum image convention
    #to calculate the x,y,z distance w.r.t the periodic boundary conditions
    delxyz = (pos[np.newaxis,:,:] - pos[:,np.newaxis,:]) - np.rint((pos[np.newaxis,:,:] - pos[:,np.newaxis,:])/L[0])*L[0]
    
    #use broadcasting arrays to make an nxn array containing the distances btwn all particles
    rij = np.linalg.norm(delxyz, axis=2)
    
    #square the distances for use in force eqns below
    rij = rij**2
    
    #turn 0s to 1 because dividing by 0 causes trouble... 
    rij[rij == 0] = 1
    
    #calculate the forces in the three directions
    forces[:,:,0] = -(48*epsilon/rij)*((sigma**12)/(rij**6)  -  (0.5)*((sigma**6)/(rij**3)))*delxyz[:,:,0]
    forces[:,:,1] = -(48*epsilon/rij)*((sigma**12)/(rij**6)  -  (0.5)*((sigma**6)/(rij**3)))*delxyz[:,:,1]
    forces[:,:,2] = -(48*epsilon/rij)*((sigma**12)/(rij**6)  -  (0.5)*((sigma**6)/(rij**3)))*delxyz[:,:,2]
    
    #calculate the potential energy and divide by 2 to avoid overcounting
    pot_energy = 0.5*np.sum(4*epsilon*((sigma**12)/(rij**6)  -  ((sigma**6)/(rij**3))))
    
    
    return forces, pot_energy

#this function calculates the new positions of particles and returns an nx3 array of 
#particle positions. 
def NewPositions(pos, vels, forces, timestep, mass, num_particles, L):
    
    #uses the equation from notes to calculate new positions.
    #sum along first axis of force array to get forces.
    pos = pos + vels*timestep + 0.5*mass*np.sum(forces, axis=1)*timestep*timestep
   
    #implement the periodic boundary conditions for all three directions
    pos[:,0] = pos[:,0] - np.floor(pos[:,0]/L[0])*L[0]
    pos[:,1] = pos[:,1] - np.floor(pos[:,1]/L[1])*L[1]
    pos[:,2] = pos[:,2] - np.floor(pos[:,2]/L[2])*L[2]
    
    return pos

#This function computes the half velocity step. returns an nx3 array of particle
#half velocities
def HalfVelocity(vels, mass, timestep, forces, num_particles):
    
    #initialize half velocity vector
    halfvels = np.zeros([num_particles, 3])
    #use half step velocity formula
    halfvels = vels + 0.5*mass*np.sum(forces, axis=1)*timestep
    
    return halfvels


#This function computes the velocities of all particles. returns an nx3 array of 
#particle velocities
def UpdateVels(vels, halfvels, mass, timestep, forces, num_particles):
    
    #use velocity formula from notes
    vels = halfvels + 0.5*mass*np.sum(forces, axis=1)*timestep
    
    return vels
    
#this function calculates the Kinetic energy of the system by summing the 
#kinetic energy of each particle. returns a scalar
def KineticEnergy(vels, mass):
    #initialize kinetic energy
    k = 0
    #loop over all velocities and calculate the kinetic energy
    for i in range(len(vels)):
        k += 0.5*mass*(vels[i,0]**2 + vels[i,1]**2 + vels[i,2]**2)
    
    return k


def RadialDist(pos, bins, L, dr, num_particles):
    
    delxyz = np.zeros([num_particles,num_particles, 3])
    rij = np.zeros([num_particles,num_particles])
            
    
    #use broadcasting arrays to make an nxnx3 array containing the x-x, y-y, z-z distances
    #for all pairs of particles, this also implements the minimum image convention
    #to calculate the x,y,z distance w.r.t the periodic boundary conditions
    delxyz = (pos[np.newaxis,:,:] - pos[:,np.newaxis,:]) - np.rint((pos[np.newaxis,:,:] - pos[:,np.newaxis,:])/L[0])*L[0]
    
    #use broadcasting arrays to make an nxn array containing the distances btwn all particles
    rij = np.linalg.norm(delxyz, axis=2)
    
    rij = np.rint(rij/dr)
    rij = rij.astype(int)
    
    for i in range(num_particles):
        j = i+1
        while j < num_particles:
            bins[rij[i,j]] += 1
            j+=1
    
    return bins




"""///////////////////////////////MAIN//////////////////////////////////////"""

#Checking the efficiency of the simulation
start1 = time.time()

#THERMOSTAT
andersen_themostat = 1
collision_frequency = 0.1

#initialize variables#
pos = []
mass = 1
simulation_steps = 50000
#how often to sample system
vid_sample_frames = 50
#how long the plot vectors should be
frames = simulation_steps/vid_sample_frames
timestep = .001
num_particles = 125
epsilon = 1
kB = 1
density = 3
#lattice diamater
sigma = (1/((density)**(1/3)))
#sigma in force eqn
force_sigma = 1
Temp = 1.2*(epsilon/kB)
pot_energy = 0

#initialize the lattice of particle positions
pos, box_size = MD.AssembleCubeLattice(sigma, density, 5,5,5)

#move the particles off the boundary to start
pos[:,0] = pos[:,0] + 0.001
pos[:,1] = pos[:,1] + 0.001
pos[:,2] = pos[:,2] + 0.001

#initialize the particle velocities
vels = MD.AssignInitialVelocities(Temp,kB, mass, num_particles)

#calculate the initial forces on the particles
forces, pot_energy = CalculateForces2(pos, epsilon, force_sigma, num_particles, box_size)

#initialize animation array
video = np.zeros([int(frames) + 1, num_particles,3])
frame_index = 0


# initialize plotting Variables
KE_plot = np.zeros(int(frames) + 1)
PE_plot = np.zeros(int(frames) + 1)
Total_energy_plot = np.zeros(int(frames) + 1)
time_plot = np.zeros(int(frames) + 1)
Temp_plot = np.zeros(int(frames) + 1)

speed_range_min = 30000
speed_range_max = 35000
speed_index = 0
Speed_plot = np.zeros((speed_range_max - speed_range_min)*num_particles)

dr = 0.02
maxdist = (((box_size[0])/2)**2 + ((box_size[1])/2)**2 + ((box_size[2])/2)**2)**(0.5)
bins = np.zeros(int(np.ceil(maxdist/dr) ))


#load initial energies and positions into plotting arrays
video[frame_index,:,:] = pos
KE_plot[frame_index] = KineticEnergy(vels, mass)
PE_plot[frame_index] = pot_energy
Total_energy_plot[frame_index] = KE_plot[frame_index] + PE_plot[frame_index]
frame_index += 1


times_vels = 0


#MAIN LOOP

for i in range(0, simulation_steps):
    #calculate the new positions from initial velocities and forces
    pos = NewPositions(pos,vels,forces,timestep,mass,num_particles, box_size)
    
    #calculate the half velocity update
    halfvels = HalfVelocity(vels, mass, timestep, forces, num_particles)
    
    #generate the new forces and calculate the potential energy    
    forces, pot_energy = CalculateForces2(pos, epsilon, force_sigma, num_particles, box_size)

    #finish the velocity update
    vels = UpdateVels(vels, halfvels, mass, timestep, forces, num_particles)
    
    
    #implements the Andersen Thermostat if boolean is true
    if(andersen_themostat == 1):
        for j in range(num_particles):
            v = random.uniform(0, 1)
            if v < collision_frequency*timestep:
                times_vels += 1
                #re-set velocities based on the boltzman distribution with
                #mean = 0 and stdev = sqrt(kT/m)
                vels[j,:] = np.random.randn(3)*(kB*Temp*(1/mass))**(1/2)
                
                
                #remove center of mass velocity from each particle
                com_vel = np.zeros([1, 3])
                for l in range(num_particles):
                    com_vel = com_vel + vels[l, :]
                com_vel = com_vel / num_particles
                for l in range(num_particles):
                    vels[l,:] = vels[l, :] - com_vel
                
                #scale to correct temp
                cur_temp = 0.66 * KineticEnergy(vels,mass) / num_particles
                scale_factor = np.sqrt(Temp /cur_temp) 
                # multiply velocities by scale factor to get new correct temp
                vels = vels*scale_factor
                
    #Calculates and stores the speeds for a converged simulation
    if i > speed_range_min and i < speed_range_max:
        for m in range(len(vels)):
            Speed_plot[speed_index] = np.linalg.norm(vels[m,:])
            speed_index += 1
    
    #sample every x timesteps
    if(i % vid_sample_frames == 0):
        #update plotting arrays
        video[frame_index,:,:] = pos
        KE_plot[frame_index] = KineticEnergy(vels, mass)
        PE_plot[frame_index] = pot_energy
        Temp_plot[frame_index] = 0.66 * KE_plot[frame_index] / num_particles
        Total_energy_plot[frame_index] = KE_plot[frame_index] + PE_plot[frame_index]
        time_plot[frame_index] = i
        frame_index += 1
        
        #Radial distribution functions
        bins = RadialDist(pos,bins,box_size,dr,num_particles)
        
        
#finish timing the simulation
end1 = time.time() 


#Radial distribution normalization
index_set = np.arange(0,len(bins),1)
radial_plotx = np.arange(0,maxdist,0.02)
bins = bins/(simulation_steps/vid_sample_frames) 
bins = bins/num_particles
bins = bins/(4*np.pi/3 * dr**3 * ((index_set + 1)**3 - index_set**3) )
bins = 2*bins/density
    
#calculate energy per particle
KE_plot = KE_plot/num_particles
PE_plot = PE_plot/num_particles
Total_energy_plot = Total_energy_plot/num_particles


""" PLOTTING SECTION """

"""
#plot individual graphs
plt.figure(1)
plt.plot(time_plot, KE_plot)
plt.ylabel('Kinetic Energy per Particle')
plt.xlabel('Timestep')
plt.suptitle('Kinetic Energy')
plt.ylim(0,2.3)
plt.show()

plt.figure(2)
plt.plot(time_plot, PE_plot)
plt.ylabel('Potential Energy per Particle')
plt.xlabel('Timestep')
plt.suptitle('Potential Energy')
plt.ylim(-6,-4)
plt.show()

plt.figure(3)
plt.plot(time_plot, Total_energy_plot)
plt.ylabel('Total Energy per Particle')
plt.xlabel('Timestep')
plt.suptitle('Total Energy')
plt.ylim(-2,-4)
plt.show()
"""
#Plot all on one figure
plt.figure(4)
plt.plot(time_plot, KE_plot)
plt.plot(time_plot, PE_plot)
plt.plot(time_plot, Total_energy_plot)
plt.legend('Kinetic Energy' 'Potential Energy' 'Total Energy')



#SPeed Histogram
num_bins = int(np.floor((np.max(Speed_plot) - np.min(Speed_plot))/0.2))
plt.figure(6)
def f(t):
    return (mass/(2*np.pi*kB*Temp))**(3/2) * 4 * np.pi * t**2 * np.exp(-(mass*t**2)/(2*kB*Temp) )
plt.hist(Speed_plot, num_bins)
t = np.arange(0.0,5.0, 0.02)
nnn = f(t)*125000
plt.plot(t,nnn)
print(np.average(Speed_plot))
print(np.var(Speed_plot))
plt.ylabel('# of particles')
plt.xlabel('Particle Speed')
plt.suptitle('Distribution of Particle Speeds')
plt.show()


#plt.figure(7)
#plt.plot(time_plot,Temp_plot)
#plt.show()

plt.figure(8)
plt.plot(radial_plotx,bins)
plt.ylabel('g(r)')
plt.xlabel('radial distance')
#plt.title('Radial distribution function for density = %' %density)
plt.show

avg_T = np.sum(Temp_plot)/len(Temp_plot)

print(avg_T)
print(end1-start1)










