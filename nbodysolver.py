import numpy as np

def acc(r,m): #r position, m mass
    a = np.zeros((len(r),3)) #Create acceleration vector
    for i in range(len(r)): #Range is size of timesteps or position steps
        for j in range(len(r)): #For each particle
            ra = (((r[i,:]-r[j,:])**2).sum())**(1./2) #dot product
            if (i != j):
                a[i,:] += -(r[i,:]-r[j,:])*m[j]/(ra**3.0) #Acceleration at each time step 
    return a # return acceleration

def acctide(r,m,ms,ne): #r position, m mass, s spin, m mass of the star and the value of ne
    a = np.zeros((len(r),3)) #Create acceleration vector
    for i in range(len(r)): #Range is size of timesteps or position steps
        for j in range(len(r)): #For each particle
            ra = (((r[i,:]-r[j,:])**2).sum())**(1./2) #dot product
            if (i != j):
                a[i,:] += -(9.)*ne*((m[j]**2)/ms)*(r[i,:]-r[j,:])/(ra**8.0) #Acceleration at each time step 
    return a # return acceleration

def accspin(r,m,S,ms, ns): #r position, m mass, s spin, m mass of the star
    a = np.zeros((len(r),3)) #Create acceleration vector
    for i in range(len(r)): #Range is size of timesteps or position steps
        for j in range(len(r)): #For each particle
            ra = (((r[i,:]-r[j,:])**2).sum())**(1./2) #dot product
            if (i != j):
                a[i,:] += -(3./2)*ns*(S**2)*(m[j]/ms)*(r[i,:]-r[j,:])/(ra**5.0) #Acceleration at each time step 
    return a # return acceleration

def Jerks(r,v,m): #position, velocity, mass
    Je = np.zeros((len(r),3)) #Define the Jerk
    for i in range(len(r)):
        for j in range(len(r)):
            if (i != j):
                ra = (((r[i,:]-r[j,:])**2).sum())**(1./2) # dot product
                Je[i,:] += - ( (v[i,:]-v[j,:])/ra**3.0 - 3.*(((v[i,:]-v[j,:])*(r[i,:]-r[j,:])).sum())/(ra**5.0) *(r[i,:]-r[j,:]) )* m[j] 
    return Je;

def Jerktide(r,v,m, ms, ne): #r position, v velocity, m mass, m mass of the star, ne
    Je = np.zeros((len(r),3)) #Define the Jerk
    for i in range(len(r)):
        for j in range(len(r)):
            if (i != j):
                ra = (((r[i,:]-r[j,:])**2).sum())**(1./2) # dot product
                Je[i,:] += - (9.0)*ne*((m[j]**2)/ms)*( (v[i,:]-v[j,:])/ra**8.0 - 8.*(((v[i,:]-v[j,:])*(r[i,:]-r[j,:])).sum())*(r[i,:]-r[j,:])/(ra**10.0) ) 
    return Je;

def Jerkspin(r,v,m, S, ms, ns): #r position, v velocity, m mass, s spin, m mass of the star
    Je = np.zeros((len(r),3)) #Define the Jerk
    for i in range(len(r)):
        for j in range(len(r)):
            if (i != j):
                ra = (((r[i,:]-r[j,:])**2).sum())**(1./2) # dot product
                Je[i,:] += - (3./2)*ns*(S**2)*(m[j]/ms)*( (v[i,:]-v[j,:])/ra**5.0 - 5.*(((v[i,:]-v[j,:])*(r[i,:]-r[j,:])).sum())*(r[i,:]-r[j,:])/(ra**7.0)  ) 
    return Je;

def HermiteUpdate(dt, r, v, m): 
    a = acc(r, m)          # current acceleration
    adot = Jerks(r,v,m)     # current jerks
    rp = r + dt*v + dt**2/2 * a + dt**3/6* adot   # predict
    vp = v + dt*a + dt**2/2 * adot
    ap = acc(rp,m)          # predicted acceleration
    adotp = Jerks(rp,vp,m)  # predicted jerks 
    vp = v + dt/2*(a+ap) - dt**2/12*(adotp-adot)  # correct
    rp = r + dt/2*(v + vp) - dt**2/10 * (ap-a)

    return rp,vp,ap,adotp

#def HermiteUpdate(dt, r, v, m): 
#    a = acc(r, m)          # current acceleration
#    adot = Jerks(r,v,m)     # current jerks
#    rp = r + dt*v + dt**2/2 * a + dt**3/6* adot   # predict
#    vp = v + dt*a + dt**2/2 * adot
#    ap = acc(rp,m)          # predicted acceleration
#    adotp = Jerks(rp,vp,m)  # predicted jerks 
#    vp = v + dt/2*(a+ap) - dt**2/12*(adotp-adot)  # correct
#    rp = r + dt/2*(v + vp) - dt**2/10 * (ap-a)
 
#    return rp,vp


def HermiteUpdatespin(dt, r, v, m, S, ms, ns): #s spin, m mass of the star
    a = acc(r, m) + accspin(r, m, S, ms, ns)          # current acceleration
    adot = Jerks(r,v,m) + Jerkspin(r, v, m, S, ms, ns) # current jerks
    rp = r + dt*v + dt**2/2 * a + dt**3/6* adot   # predict
    vp = v + dt*a + dt**2/2 * adot
    ap = acc(rp, m) + accspin(rp, m, S, ms, ns)         # predicted acceleration
    adotp = Jerks(rp, vp, m) + Jerkspin(rp, vp, m, S, ms, ns) # predicted jerks 
    vp = v + dt/2*(a+ap) - dt**2/12*(adotp-adot)  # correct
    rp = r + dt/2*(v + vp) - dt**2/10 * (ap-a)
 
    return rp,vp

#def HermiteUpdatetide(dt, r, v, m, ms, ns): # m mass of the star
#    a = acc(r, m) + acctide(r, m, ms, ns)          # current acceleration
#    adot = Jerks(r,v,m) + Jerktide(r, v, m, ms, ns) # current jerks
#    rp = r + dt*v + dt**2/2 * a + dt**3/6* adot   # predict
#    vp = v + dt*a + dt**2/2 * adot
#    ap = acc(rp, m) + acctide(rp, m, ms, ns)         # predicted acceleration
#    adotp = Jerks(rp, vp, m) + Jerktide(rp, vp, m, ms, ns) # predicted jerks 
#    vp = v + dt/2*(a+ap) - dt**2/12*(adotp-adot)  # correct
#    rp = r + dt/2*(v + vp) - dt**2/10 * (ap-a)
 
#    return rp,vp

def HermiteUpdatetide(dt, r, v, m, ms, ns): # m mass of the star
    a = acc(r, m) + acctide(r, m, ms, ns)          # current acceleration
    adot = Jerks(r,v,m) + Jerktide(r, v, m, ms, ns) # current jerks
    rp = r + dt*v + dt**2/2 * a + dt**3/6* adot   # predict
    vp = v + dt*a + dt**2/2 * adot

    aNp = acc(rp, m)
    aTp = acctide(rp, m, ms, ns)
    ap = aNp + aTp         # predicted acceleration

    aNdotp = Jerks(rp, vp, m)
    aTdotp = Jerktide(rp, vp, m, ms, ns)
    adotp = aNdotp + aTdotp  # predicted jerks 
    
    vp = v + dt/2*(a+ap) - dt**2/12*(adotp-adot)  # correct
    rp = r + dt/2*(v + vp) - dt**2/10 * (ap-a)
 
    return rp,vp,ap,adotp,aNp,aNdotp


def Hermite4th(pri,sec, bina, nsteps, Dt):
    
    N=2
    m = np.ones(N)#/N #Remove the N if not necessary
    m[0]=pri.mass
    m[1]=sec.mass
    
    r_res = np.zeros((2,3,nsteps)) # 2 because of two bodies
    v_res = np.zeros((2,3,nsteps))
    a_res = np.zeros((2,3,nsteps)) # 2 because of two bodies
    adot_res = np.zeros((2,3,nsteps))
    
    time = np.zeros(nsteps)
    r_res[:,:,0] = bina.r.copy()
    v_res[:,:,0] = bina.v.copy()
    
    for i in range(1,nsteps):
        (r_res[:,:,i],v_res[:,:,i],a_res[:,:,i],adot_res[:,:,i]) = HermiteUpdate(Dt, r_res[:,:,i-1], v_res[:,:,i-1], m)
        time[i] = time[i-1] + Dt
        
    return r_res, v_res, a_res, adot_res

#def Hermite4th(pri,sec, bina, nsteps, Dt):
#    
#    N=2
#    m = np.ones(N)#/N #Remove the N if not necessary
#    m[0]=pri.mass
#    m[1]=sec.mass
#    
#    r_res = np.zeros((2,3,nsteps)) # 2 because of two bodies
#    v_res = np.zeros((2,3,nsteps))
    
#    time = np.zeros(nsteps)
#    r_res[:,:,0] = bina.r.copy()
#    v_res[:,:,0] = bina.v.copy()
    
#    for i in range(1,nsteps):
#        (r_res[:,:,i],v_res[:,:,i]) = HermiteUpdate(Dt, r_res[:,:,i-1], v_res[:,:,i-1], m)
#        time[i] = time[i-1] + Dt
        
#    return r_res, v_res


def Hermite4thspin(pri,sec, bina, nsteps, Dt):
    
    N=2
    m = np.ones(N)#/N #Remove the N if not necessary
    m[0]=pri.mass
    m[1]=sec.mass
    
    r_res = np.zeros((2,3,nsteps)) # 2 because of two bodies
    v_res = np.zeros((2,3,nsteps))
    
    time = np.zeros(nsteps)
    r_res[:,:,0] = bina.r.copy()
    v_res[:,:,0] = bina.v.copy()
    
    for i in range(1,nsteps):
        (r_res[:,:,i],v_res[:,:,i]) = HermiteUpdatespin(Dt, r_res[:,:,i-1], v_res[:,:,i-1], m, sec.spin, sec.mass, sec.nspin)
        time[i] = time[i-1] + Dt
        
    return r_res, v_res

#def Hermite4thtide(pri,sec, bina, nsteps, Dt):
#
#    N=2
#    m = np.ones(N)#/N #Remove the N if not necessary
#    m[0]=pri.mass
#    m[1]=sec.mass
#    
#    r_res = np.zeros((2,3,nsteps)) # 2 because of two bodies
#    v_res = np.zeros((2,3,nsteps))
#    
#    time = np.zeros(nsteps)
#    r_res[:,:,0] = bina.r.copy()
#    v_res[:,:,0] = bina.v.copy()
#    
#    for i in range(1,nsteps):
#        (r_res[:,:,i],v_res[:,:,i]) = HermiteUpdatetide(Dt, r_res[:,:,i-1], v_res[:,:,i-1], m, sec.mass, sec.ntide)
#        time[i] = time[i-1] + Dt
#        
#    return r_res, v_res

def Hermite4thtide(pri,sec, bina, nsteps, Dt):
    
    N=2
    m = np.ones(N)#/N #Remove the N if not necessary
    m[0]=pri.mass
    m[1]=sec.mass
    
    r_res = np.zeros((2,3,nsteps)) # 2 because of two bodies
    v_res = np.zeros((2,3,nsteps))

    aN_res = np.zeros((2,3,nsteps)) # 2 because of two bodies
    adotN_res = np.zeros((2,3,nsteps))

    a_res = np.zeros((2,3,nsteps)) # 2 because of two bodies
    adot_res = np.zeros((2,3,nsteps))

    
    time = np.zeros(nsteps)
    r_res[:,:,0] = bina.r.copy()
    v_res[:,:,0] = bina.v.copy()
    
    for i in range(1,nsteps):
        (r_res[:,:,i],v_res[:,:,i],a_res[:,:,i],adot_res[:,:,i],aN_res[:,:,i],adotN_res[:,:,i]) = HermiteUpdatetide(Dt, r_res[:,:,i-1], v_res[:,:,i-1], m, sec.mass, sec.ntide)
        time[i] = time[i-1] + Dt
        
    return r_res, v_res, a_res, adot_res, aN_res, adotN_res
