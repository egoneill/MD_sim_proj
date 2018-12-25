!"""
Created on Sun Oct  7 12:24:29 2018
@author: Acer
"""

#Ising_MC_functions
import numpy as np
import matplotlib.pyplot as plt

def PlotLatticeConfiguration(pos, final_spins, title):
  #@rvanlehn 2016, implimented in Python by Jsheavly 2018 
  #PlotLatticeConfiguration
  #  Draws scatter plot representation of "finalstate", with spins colored
  #  by their value. Title string has to be input from the user.
  plt.close(1)
  plt.figure(1)
    # Visualize final state of system using scatter plot with filled in (large)
    # points. Uses two color color map
    
  if sum(final_spins) == -400:
      plt.scatter(pos[:, 0], pos[:, 1], 200, c='blue')
        
  elif sum(final_spins) == 400:
      plt.scatter(pos[:, 0], pos[:, 1], 200, c='red')
        
  else:
      plt.scatter(pos[:, 0], pos[:, 1], 200, c=final_spins, cmap='bwr')
    
  plt.xlabel('x position',fontsize=14)
  plt.ylabel('y position',fontsize=14)
  plt.xticks(np.arange(0,np.max(pos[:,0]+1)),fontsize=14)
  plt.yticks(np.arange(0,np.max(pos[:,1]+1)),fontsize=14)
  
  
  plt.title(title,fontsize=16)
  plt.tight_layout()
  return

def InitializeLattice( num_lattice_x, num_lattice_y ):
# @rvanlehn 2016, implimented in Python by Jsheavly 2018 
# InitializeLattice
#   Determines number of total lattice points from lattice dimensions,
#   assigns them random spins. Returns number of lattice points and array
#   of spins.
# Params:
#   num_lattice_x = number of columns in lattice 
#   num_lattice_y = number of rows in lattice

    tot_num_lattice=num_lattice_x*num_lattice_y # total number of lattice points

    # "spin" of lattice site (either +-1)
    #spins=zeros(tot_num_lattice, 1)
    # Initialize lattice by randomly assigning spins to be up or down.
    #for i in range(tot_num_lattice):
    # Generate random integer of either 1 or 2 multiply by 2 and subtract
    # 3 to get either -1 or 1.
    spins =2*np.random.randint(1,3,size=[tot_num_lattice])-3
    
    return tot_num_lattice, spins
  
def AssembleSquareLattice( num_lattice_x, num_lattice_y ):
# @rvanlehn 2016
# AssembleSquareLattice
#   Creates a square lattice according to input arguments related to
#   size of lattice in x/y dimensions.
#   Returns position of lattice points and a 2D array containing the list
#   of neighbors for each lattice point
# Params:
#   num_lattice_x = number of columns in lattice 
#   num_lattice_y = number of rows in lattice

    # define parameters
    # cartesian positions of each lattice point
    pos=np.zeros([num_lattice_x*num_lattice_y, 2])
    # indices in array of 4 neighbors for each lattice point
    neighbors=np.zeros([num_lattice_x*num_lattice_y, 4])
    # spatial increments in x/y dimension for making lattice
    incr_x = 1.0
    incr_y = 1.0
    # place all lattice points now
    #j indicates the row number(y), and i indicates the column number(x), so the ideces go down the lattice and then across
    for i in range(num_lattice_x):
        for j in range(num_lattice_y):
            bead_index = (i)*num_lattice_y + j
            pos[bead_index, 0] = i*incr_x
            pos[bead_index, 1] = j*incr_y
           
            # Generate neighbors here - check for periodicity on lattice.
            if i < num_lattice_x-1:
                neighbors[bead_index, 0] = bead_index + num_lattice_y # neighbor to right
            else:
                neighbors[bead_index, 0] = j # neighbor to right pbc,just picks on first column
            
            if i > 0:
                neighbors[bead_index, 1] = bead_index - num_lattice_y  # neighbor to left
            else:
                neighbors[bead_index, 1] = (num_lattice_x-1)*num_lattice_y + j # neighbor to left pbc, just picks on last column
            
            if j > 0:
                neighbors[bead_index, 2] = bead_index -1# neighbor to top
            else:
                neighbors[bead_index, 2] = bead_index +num_lattice_y -1# neighbor to top pbc, just picks on bottom row
            
            if j < num_lattice_y-1:
                neighbors[bead_index, 3] = bead_index +1# neighbor to bottom 
            else:
                neighbors[bead_index, 3] = bead_index - num_lattice_y+1# neighbor to bottom pbc, just picks on top row
    neighbors=np.asarray([[int(x) for x in y] for y in neighbors])
    return pos, neighbors