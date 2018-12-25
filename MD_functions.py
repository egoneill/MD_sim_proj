"""
Created on Sun Oct  7 12:07:15 2018
@author: Acer
"""
import numpy as np
def AssembleCubeLattice( lattice_diameter, density, num_lattice_x, num_lattice_y, num_lattice_z ):

#AssembleCubeLattice
#   Creates a cubic arrangement of particles for initializing simulation
#   run
# Params:
#   lattice_diameter = distance between lattice points
#   density = number of particles / lattice_diam^3
#   num_lattice_x/y/z number of particles along each dimension

  
    # determine appropriate size of box 
    num_particles = num_lattice_x*num_lattice_y*num_lattice_z
    lin_density=(density)**(1/3)#we specify that each of the linear densities (num/length) must be the same
    box_size = 1/lin_density*np.asarray([num_lattice_x, num_lattice_y, num_lattice_z])#Then we can specify the length from the number in each dimention
    # cartesian positions of each lattice point
    pos=np.zeros([num_particles, 3])
    
    # spatial increments in x/y dimension for making lattice
    incr_x = lattice_diameter
    incr_y = lattice_diameter
    incr_z = lattice_diameter
    # place all lattice points now
    bead_index = 0
    for i in range(num_lattice_z):
        for j in range(num_lattice_y):
            for k in range(num_lattice_x):
                pos[bead_index, 0] = k*incr_x
                pos[bead_index, 1] = j*incr_y
                pos[bead_index, 2] = i*incr_z
                bead_index = bead_index + 1

    return pos, box_size

def AssignInitialVelocities( temp, k_b, mass, num_particles ):
# AssignInitialVelocities
#   Randomly assigns velocities drawn from a Maxwell-Boltzmann distribution
#   at the desired temperature. Removes any overall center-of-mass motion
#   to prevent the "flying Iceberg" problem. Rescales velocities to obtain
#   correct temperature after removing COM motion.
# Params:
#   temp = temperature used to initialize velocities
#   k_b = value of boltzmann constant
#   mass = mass of each particle
#   num_particles = number of total particles in system. 

    np.random.seed(9001)
    
    # define invmass, used below
    invmass = 1.0/mass
    # Initialize array.
    vels = np.zeros([num_particles, 3])
    
    # get 3 randomly generated numbers sampled from a normal (Gaussian)
    # distribution as opposed to uniform distribution
    for i in range(num_particles):
        # Maxwell-Boltzmann distribution for velocities is a Gaussian
        # distribution with mean = 0 and standard deviation = sqrt(kT/m).
        # randn(3,1) generates 3 random numbers from a Gaussian
        # distribution with mean = 0 and standard devition = 1. A property
        # of a variable, X, that is Gaussian distributed is that
        # multiplying by a factor A does not change its mean, but
        # multiplies its standard deviation by A. Therefore we multiply the
        # results of randn() by the desired standard deviation to draw from
        # a M-B distribution.
        vels[i,:] = np.random.randn(3)*(k_b*temp*invmass)**(1/2)


    # A few tricks here want to prevent the entire system having a net
    # velocity (flying ice box), so we remove center of mass velocity from each
    # particle to set center of mass velocity to 0.
    com_vel = np.zeros([1, 3])
    for i in range(num_particles):
        com_vel = com_vel + vels[i, :]
    com_vel = com_vel / num_particles
    for i in range(num_particles):
        vels[i,:] = vels[i, :] - com_vel
    
    # Next, rescale velocities by calculating current temperature, then multiplying
    # all velocities uniformly to get new correct temperature. Calculate temperature from
    # T = 2/3k * KE = 1/3k * m v^2
    # Could have combined with above step, but keep separate to show main
    # idea.
    
    cur_ke = 0.0
    for i in range(num_particles):
       cur_ke = cur_ke + 0.5*mass*np.dot(vels[i, :], vels[i, :]) 

    cur_temp = 0.66 * cur_ke / num_particles
    # Get scaling factor
    scale_factor = np.sqrt(temp /cur_temp) 
    # multiply all velocities by scale factor to get new correct temp
    new_ke = 0.0
    for i in range(num_particles):
        vels[i,:] = vels[i,:] * scale_factor
        new_ke = new_ke + 0.5*mass*np.dot(vels[i,:], vels[i,:]) 
   
    rescale_temp = 0.66 * new_ke / num_particles
    print('Original temp =',cur_temp ,', rescaled temp =',  rescale_temp)
    return vels
def Plot_3D_configuration(pos,**kwargs):
    #pos should be a final configuration of your atoms, as an n x 3 array.
    #identitiy should be a set of ones and zeros, which will tell you which component is which for part f
    #by default, you should make identity a set of zeros
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    size=np.shape(pos)
    number=size[0]
    if ('identity' in kwargs):
      identity=np.asarray(kwargs['identity'])
    else:
      identity=np.ones(number)
    colors=np.transpose([identity,np.zeros(number),1-identity])
    if ('title' in kwargs):
      title=kwargs['title']
    else:
      title='Output Configuration'
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    x,y,z=pos[:,0],pos[:,1],pos[:,2]
    ax.scatter(x,y,z,c=colors,s=100)
    """                                                                                                                                                    
    Scaling is done from here...                                                                                                                           
    """
    x_scale=np.ceil(np.max(x))+1
    y_scale=np.ceil(np.max(y))+1
    z_scale=np.ceil(np.max(z))+1
    
    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=0.75
    
    def short_proj():
      return np.dot(Axes3D.get_proj(ax), scale)
    
    ax.get_proj=short_proj
    """                                                                                                                                                    
    to here                                                                                                                                                
    """
    '''
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')
    '''
    ax.set_xlabel('X axis',fontsize=16)
    ax.set_ylabel('Y axis',fontsize=16) 
    ax.set_zlabel('Z axis',fontsize=16)
    ax.set_xticks(np.arange(0,np.ceil(np.max(x))+1))
    ax.set_yticks(np.arange(0,np.ceil(np.max(y))+1))
    ax.set_zticks(np.arange(0,np.ceil(np.max(z))+1))
    ax.set_xticklabels(np.arange(0,int(np.ceil(np.max(x))+1)),fontsize=12)
    ax.set_yticklabels(np.arange(0,int(np.ceil(np.max(y))+1)),fontsize=12)
    ax.set_zticklabels(np.arange(0,int(np.ceil(np.max(z))+1)),fontsize=12)
    ax.set_title(title,fontsize=16)
    return

def Make_3D_animation(pos_save,**kwargs):
    #pos should be a final configuration of your atoms, as an m x n x 3 array. where m is number of steps, and n is number of particles
    #identitiy should be a set of ones and zeros, which will tell you which component is which for part f
    #by default, you should make identity a set of zeros
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d import proj3d
    import matplotlib.animation as animation
    pos_save=np.asarray(pos_save)
    size=np.shape(pos_save)
    number_particles=size[1]
    number_frames=size[0]
    if ('identity' in kwargs):
      identity=np.asarray(kwargs['identity'])
    else:
      identity=np.ones(number_particles)
    colors=np.transpose([identity,np.zeros(number_particles),1-identity])
    if ('title' in kwargs):
      title=kwargs['title']
    else:
      title='Animations at frame'
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111,projection='3d')
    
      
    x,y,z=pos_save[:,:,0],pos_save[:,:,1],pos_save[:,:,2]
    #ax.scatter(x,y,z,c=colors,s=100)
    
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')
   
    ax.set_xlabel('X axis',fontsize=16)
    ax.set_ylabel('Y axis',fontsize=16) 
    ax.set_zlabel('Z axis',fontsize=16)
    
    
    ax.set_xticks(np.arange(0,np.ceil(np.max(x))+1))
    ax.set_yticks(np.arange(0,np.ceil(np.max(y))+1))
    ax.set_zticks(np.arange(0,np.ceil(np.max(z))+1))
    ax.set_xticklabels(np.arange(0,int(np.ceil(np.max(x))+1)),fontsize=12)
    ax.set_yticklabels(np.arange(0,int(np.ceil(np.max(y))+1)),fontsize=12)
    ax.set_zticklabels(np.arange(0,int(np.ceil(np.max(z))+1)),fontsize=12)
    """                                                                                                                                                    
    Scaling is done from here...                                                                                                                           
    """
    '''
    x_scale=np.ceil(np.max(x))+1
    y_scale=np.ceil(np.max(y))+1
    z_scale=np.ceil(np.max(z))+1
    
    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=0.75
    
    def short_proj():
      return np.dot(Axes3D.get_proj(ax), scale)
    
    ax.get_proj=short_proj
    '''
    """                                                                                                                                                    
    to here                                                                                                                                                
    """
    Title=ax.set_title(title,fontsize=16)
    def update_graph(num):
      graph._offsets3d=(x[num,:], y[num,:],z[num,:] )
      Title.set_text(title+'time='+str(num))
    
    #ims.append((plt.scatter(x[i,:],y[i,:],c=colors,s=100)))
    vol=np.max([np.max(x),np.max(y),np.max(z)])
    rel_size=5000/vol
    graph=ax.scatter(x[0,:], y[0,:],z[0,:],c=colors,s=rel_size)
    im_ani = animation.FuncAnimation(fig, update_graph,number_frames, interval=5, repeat_delay=300,
                                       blit=False)
    # To save this second animation with some metadata, use the following command:
    #im_ani.save('im.mp4', metadata={'artist':'Guido'})
    plt.tight_layout()
    plt.show()
    
    return im_ani   