import numpy as np
import time
import math
import random
#------------------------------------------------------------------------------------------
#Units and conversions:
#------------------------------------------------------------------------------------------
#code units: Rsun, Msun, G=1, ...
c_SI       = 299792458.0        #m/s
M_sun_SI   = 1.989*(10.**30.)   #kg
R_sun_SI   = 695800000.         #m
R_bull_SI = 1000 #m 
AU_SI      = 149597871000.      #m 
G_new_SI   = 6.67*(10.**(-11.)) #m**3 kg**(-1) s**(-2)
AU_U       = AU_SI/R_sun_SI                             #from dist AU to code units (U)
kmsec_U    = 1000./np.sqrt(G_new_SI*M_sun_SI/R_bull_SI)  #from vel km/sec to code units compact objects
#kmsec_U    = 1000./np.sqrt(G_new_SI*M_sun_SI/R_sun_SI)  #from vel km/sec to code units (U)

time_U     = np.sqrt((R_bull_SI)/(G_new_SI*M_sun_SI)) #from code units(U) to time sec
c_CU = c_SI/1000*kmsec_U
Rsch_1Msun_unitRsun = ((2.*G_new_SI*(1.*M_sun_SI))/(c_SI**2.))/R_sun_SI

class Particle:
    
    
    def __init__(self, name, mass, radi, spin): #Add lane endem equation so I can find the value of the radius with just the mass

        #Obtained this numbers from Poisson's book 
        k2ms = 0.014443 #Main Sequence: Tidal love Number k2 for polytropic index n=3, k = 0.0144430  
        k2ns = 0.375966#Neutron Star: Index range n = [0.5, 1]. For n=0.5, k = 0.449154. For n=2/3, k = 0.375966. For n=1, k = 0.259909
        k2wd = 0.143279 #White Dwarf: Index n = 1.5, k=0.143279 non relativistic electrons, n = 3, k = 0.0144430    
        kdns = 1.1311 #Irvin's computation from Chakrabati's paper

        
        if name == "BH":
            self.name = "BH"
            self.mass = mass
            self.radi = (2*G_new_SI*self.mass*M_sun_SI/c_SI**2)/1000
            self.spin = spin
            self.rdis = None
            self.sdis = None
            self.ntide = 0
            self.ndtide = 0
            self.eta = (1./360)*(self.radi**6)/(c_CU) 
        
        elif name == "MS":
            self.name = name
            self.mass = mass
            self.radi = radi
            self.spin = spin
            self.spinv = np.zeros([3]) #This is redefined when choosing spin alignment 
            self.sdis = (self.mass/((self.radi)**3))**(1./2)
            self.ntide = (2./3)*k2ms*(self.radi**5)  # n = (2/3)*k*l^5
            self.nspin = (2./3)*k2ms*(self.radi**5) # Same as n_tidal in Newtonian Limit
            self.info = "Main Sequence. Polytropic Index n = 3 "
            
        elif name == "WD":
            self.name = name
            self.mass = mass
            self.radi = radi
            self.spin = spin
            self.sdis = (self.mass/((self.radi)**3))**(1./2)
            self.ntide = (2./3)*k2wd*(self.radi**5)  # n = (2/3)*k*l^5
            self.nspin = (2./3)*k2wd*(self.radi**5) # Same as n_tidal in Newtonian Limit
            self.info = "White Dwarf. Polytropic Index n = [1.5,3]"
            
        elif name == "NS":
            self.name = name
            self.mass = mass
            self.radi = radi
            self.spin = spin
            self.sdis = (self.mass/((self.radi)**3))**(1./2)
            self.ntide = (2./3)*k2ns*(self.radi**5)  # n = (2/3)*k*l^5
            self.nspin = (2./3)*k2ns*(self.radi**5) # Same as n_tidal in Newtonian Limit
            self.ndtide = (1./2)*kdns*(self.radi**7/c_CU**2)
            self.info = "Neutron Star. Polytropic Index n = [0.5,1] "
        
        else: 
            print("Introduce a valid name for the star")
#Set name as WD, NS, and given the name given the polytrope values
    
#Compute tidal radios
           
    #def newname(self, new_name):
    #    self.name = new_name 

class Binaryecc:
    
    def __init__(self, pri, sec, orbd, ecc, mean): #primary, secondary, orbital distance, eccentricity
        
        mr = pri.mass/sec.mass #Mass ratio
        mt = pri.mass + sec.mass #Total mass binary
        
        self.orbd = orbd
        
        #Add eccentricity
        ecc_bin = ecc
        kk = 0.85 #This seems like some parameter that I have chosen to use given info online
        mean_anom = mean #2.*np.pi - 0.1 #(2.*np.pi)*random.random() #mean_anomaly
        E0 = mean_anom + np.sign(np.sin(mean_anom))*kk*ecc_bin
        E1 = E0 - (E0 - ecc_bin*np.sin(E0) - mean_anom)/(1 - ecc_bin*np.cos(E0))
        ifor = 8
        Ei1 = E0 #Define E_{i-1}
        for i in range(ifor):
            Ei = Ei1 - (Ei1 - ecc_bin*np.sin(Ei1) - mean_anom)/(1 - ecc_bin*np.cos(Ei1))   
            Ei1 = Ei
        ecc_anom = Ei
        bin_f = 2.*np.arctan(((1+ecc_bin)/(1-ecc_bin))**(1./2))*np.tan(ecc_anom/2.)
        #Redefinition of mass variables
        mb1 = pri.mass
        mb2 = sec.mass
        Mb12_SI     = (mb1+mb2)*M_sun_SI
        pfac_SI     = ((orbd*R_sun_SI)*(1.-ecc_bin**2))
        hfac_SI     = (np.sqrt(G_new_SI*Mb12_SI*pfac_SI))
        rdist_SI    = (pfac_SI/(1.+ecc_bin*np.cos(bin_f)))
        v_ang_SI    = (hfac_SI/rdist_SI)
        v_rad_SI    = ((ecc_bin*np.sin(bin_f)/pfac_SI)*hfac_SI)
        #calc pos and vel of reduced-mass object:
        posx_U  = - ((rdist_SI*np.cos(bin_f))/R_sun_SI)
        posy_U  =   ((rdist_SI*np.sin(bin_f))/R_sun_SI)
        velx_U  = - ((v_ang_SI*np.sin(bin_f) - v_rad_SI*np.cos(bin_f))*(kmsec_U/1000.))
        vely_U  = - ((v_rad_SI*np.sin(bin_f) + v_ang_SI*np.cos(bin_f))*(kmsec_U/1000.))
        #define pos/vel vecs:
        posvec_U = np.array([posx_U, posy_U, 0.0])
        velvec_U = np.array([velx_U, vely_U, 0.0])

        #change to binary COM:
        b1_posxyz_binCM    = - (mb2/(mb1+mb2))*posvec_U
        b1_velxyz_binCM    = - (mb2/(mb1+mb2))*velvec_U
        b2_posxyz_binCM    =   (mb1/(mb1+mb2))*posvec_U
        b2_velxyz_binCM    =   (mb1/(mb1+mb2))*velvec_U


        N = 2
        self.r = np.zeros((N,3))
        self.v = np.zeros((N,3))
        m1 = pri.mass
        m2 = sec.mass
        redvel = np.sqrt((m1+m2)/orbd)
        self.redvel = redvel
        #self.r[0] = np.array([(m2/mt)*orbd,0,0])
        #self.r[1] = np.array([-(m1/mt)*orbd,0,0])
        #self.v[0] = np.array([0,(m2/mt)*redvel,0])
        #self.v[1] = np.array([0,-(m1/mt)*redvel,0])
        self.r[0] = b1_posxyz_binCM
        self.r[1] = b2_posxyz_binCM
        self.v[0] = b1_velxyz_binCM
        self.v[1] = b2_velxyz_binCM
        
        ##################
        #Period
        ##################
        
        periodctime = 2.0*np.pi*(orbd**(3./2))/np.sqrt(mt)
        self.periodct = periodctime
        self.period =  periodctime*time_U/(24*60*60)#Period in days
        
        ##################
        #Roche Lobe
        ##################
        
        self.rochepri = (0.49*(mr**(2./3))*orbd)/(0.6*(mr**(2./3))+ np.log(1+mr**(1./3)))
        
        self.rochesec = (0.49*(mr**(-2./3))*orbd)/(0.6*(mr**(-2./3))+ np.log(1+mr**(-1./3)))
        
        if self.rochepri <= pri.radi:
            self.rochefill = True
                
        elif self.rochesec <= sec.radi:
            self.rochefill = True
            
        else: 
            self.rochefill = False
            
        ##################
        #Collision
        ##################
            
        if (pri.radi + sec.radi) <= orbd:
            self.coll = True
        
        else:
            self.coll = False 
            
        
        ##################
        #Tidal Radius
        ##################
        
        if pri.name == "BH":
            self.tidalrpri = pri.radi #Set equal to its radius
        
        else:
            self.tidalrpri = pri.radi*(sec.mass/pri.mass)**(1./3)
        
        if sec.name == "BH":
            self.tidalrsec = sec.radi
        else:
            self.tidalrsec = sec.radi*(pri.mass/sec.mass)**(1./3)
            
        ##################
        #Time to merge
        ##################
        self.mergertime = (5/256)*(self.orbd**4)*(c_CU**5)*(1./(pri.mass*sec.mass*(pri.mass+sec.mass)))
        ##################
        #Spin Effects
        ##################
        
        if pri.name == "BH":
            self.spineffpri = 0#(3./(2*pri.mass))*(pri.nspin*(pri.spin**2))*(1./orbd**2)
        
        else:
            self.spineffpri = (3./(2*pri.mass))*(pri.nspin*(pri.spin**2))*(1./orbd**2)
            
        if sec.name == "BH":
            self.spineffsec = 0#(3./(2*sec.mass))*(sec.nspin*(sec.spin**2))*(1./orbd**2)
        
        else:
            self.spineffsec = (3./(2*sec.mass))*(sec.nspin*(sec.spin**2))*(1./orbd**2)
            
        ##################
        #Tidal Effects
        ##################
        
        #Correct/check here tidal effects
        
        if pri.name == "BH":
            self.tideeffpri = 0#(9./(1*pri.mass))*(pri.ntide*sec.mass)*(1./orbd**5)
        
        else:
            self.tideeffpri = -(9./(1*pri.mass))*(pri.ntide*sec.mass)*(1./orbd**5)
            
        if sec.name == "BH":
            self.tideeffsec = 0#(9./(1*sec.mass))*(sec.ntide*pri.mass)*(1./orbd**5)
        
        else:
            self.tideeffsec = -(9./(1*sec.mass))*(sec.ntide*pri.mass)*(1./orbd**5)
            
        ##################
        #Dynamical tides
        ##################
        
        if pri.name == "BH":
            self.dyntideeffpri = 0#(9./(1*pri.mass))*(pri.ntide*sec.mass)*(1./orbd**5)
        
        else:
            self.dyntideeffpri = -(18./(1*pri.mass))*(pri.ndtide*sec.mass)*(1./orbd**7)
            
        if sec.name == "BH":
            self.dyntideeffsec = 0#(9./(1*sec.mass))*(sec.ntide*pri.mass)*(1./orbd**5)
        
        else:
            self.dyntideeffsec = -(18./(1*sec.mass))*(sec.ndtide*pri.mass)*(1./orbd**7)
        
        ##################
        #Dissipative Effects
        ##################
        
        if pri.name == "BH":
            self.disseffpri = -(9./(1*pri.mass))*(pri.eta*sec.mass)*(1./orbd**6)
        
        else:
            self.disseffpri = 0 #(9./(1*pri.mass))*(pri.ntide*sec.mass)*(1./orbd**5)
            
        if sec.name == "BH":
            self.disseffsec = -(9./(1*sec.mass))*(sec.eta*pri.mass)*(1./orbd**6)
        
        else:
            self.disseffsec = 0 #(9./(1*sec.mass))*(sec.ntide*pri.mass)*(1./orbd**5)     
        
class Binary:
    
    def __init__(self, pri, sec, orbd, ecc): #primary, secondary, orbital distance, eccentricity
        
        mr = pri.mass/sec.mass #Mass ratio
        mt = pri.mass + sec.mass #Total mass binary
        
        self.orbd = orbd
        
        N = 2
        self.r = np.zeros((N,3))
        self.v = np.zeros((N,3))
        m1 = pri.mass
        m2 = sec.mass
        redvel = np.sqrt((m1+m2)/orbd)
        self.redvel = redvel
        self.r[0] = np.array([(m2/mt)*orbd,0,0])
        self.r[1] = np.array([-(m1/mt)*orbd,0,0])
        self.v[0] = np.array([0,(m2/mt)*redvel,0])
        self.v[1] = np.array([0,-(m1/mt)*redvel,0])
        
        ##################
        #Period
        ##################
        
        periodctime = 2.0*np.pi*(orbd**(3./2))/np.sqrt(mt)
        self.periodct = periodctime
        self.period =  periodctime*time_U/(24*60*60)#Period in days
        
        ##################
        #Roche Lobe
        ##################
        
        self.rochepri = (0.49*(mr**(2./3))*orbd)/(0.6*(mr**(2./3))+ np.log(1+mr**(1./3)))
        
        self.rochesec = (0.49*(mr**(-2./3))*orbd)/(0.6*(mr**(-2./3))+ np.log(1+mr**(-1./3)))
        
        if self.rochepri <= pri.radi:
            self.rochefill = True
                
        elif self.rochesec <= sec.radi:
            self.rochefill = True
            
        else: 
            self.rochefill = False
            
        ##################
        #Collision
        ##################
            
        if (pri.radi + sec.radi) <= orbd:
            self.coll = True
        
        else:
            self.coll = False 
            
        ##################
        #Tidal Radius
        ##################
        
        if pri.name == "BH":
            self.tidalrpri = pri.radi #Set equal to its radius
        
        else:
            self.tidalrpri = pri.radi*(sec.mass/pri.mass)**(1./3)
        
        if sec.name == "BH":
            self.tidalrsec = sec.radi
        else:
            self.tidalrsec = sec.radi*(pri.mass/sec.mass)**(1./3)
    
        ##################
        #Spin Effects
        ##################
        
        if pri.name == "BH":
            self.spineffpri = 0#(3./(2*pri.mass))*(pri.nspin*(pri.spin**2))*(1./orbd**2)
        
        else:
            self.spineffpri = (3./(2*pri.mass))*(pri.nspin*(pri.spin**2))*(1./orbd**2)
            
        if sec.name == "BH":
            self.spineffsec = 0#(3./(2*sec.mass))*(sec.nspin*(sec.spin**2))*(1./orbd**2)
        
        else:
            self.spineffsec = (3./(2*sec.mass))*(sec.nspin*(sec.spin**2))*(1./orbd**2)
            
        ##################
        #Tidal Effects
        ##################
        
        #Correct/check here tidal effects
        
        if pri.name == "BH":
            self.tideeffpri = 0#(9./(1*pri.mass))*(pri.ntide*sec.mass)*(1./orbd**5)
        
        else:
            self.tideeffpri = (9./(1*pri.mass))*(pri.ntide*sec.mass)*(1./orbd**5)
            
        if sec.name == "BH":
            self.tideeffsec = 0#(9./(1*sec.mass))*(sec.ntide*pri.mass)*(1./orbd**5)
        
        else:
            self.tideeffsec = (9./(1*sec.mass))*(sec.ntide*pri.mass)*(1./orbd**5)
            
        ##################
        #Dissipative Effects
        ##################            
    

        ##################
        #GW Effects
        ##################

        #a0_r = (G*m2/(r12**(2.)))*r12_u#Term 1/c^0 Unit radial direction

        #a2_r = (G*m2/(r12**(2.)))*(((5*G*m1)/(r12) + (4*G*m2)/(r12) + ((3./2)*(np.dot(r12_u,v12))**(2) - np.dot(v1,v1) + 4*np.dot(v1,v2) -2*np.dot(v2,v2)))*r12_u) #Term 1/c^2 Unit radial direction

        #a2_v = (G*m2/(r12**(2.)))*((4*(np.dot(r12_u,v1)) - 3*(np.dot(r12_u,v2)))*v12) #Term 1/c^2 Velocity direction

        #a4_r = (G*m2/(r12**(2.)))*( (-57.0*(G**2)*(m1**2)/(4*r12**2)) + (-69.0*(G**2)*m1*m2/(2*r12**2)) + (-9.0*(G**2)*(m2**2)/(r12**2)) + ((np.dot(r12_u, v1)**2)*( (-15.0/8*(np.dot(r12_u,v2))**2) + (3./2*np.dot(v1,v1)) + (-6.*np.dot(v1,v2)) + (9./2*np.dot(v2,v2)) ) + np.dot(v1,v2)*(-2*(np.dot(v1,v2)) + 4*np.dot(v2,v2)) -2*v2**4 ) + (G*m1/r12)*( (39./2*np.dot(r12_u, v1)**2) + (-39*np.dot(r12_u, v1)*np.dot(r12_u, v2)) + (17.0/2*np.dot(r12_u, v2)**2) + (-15./4*np.dot(v1,v1)) + (-5./2*np.dot(v1,v2)) + (5./4*np.dot(v2,v2))) + (G*m2/r12)*( (2.*np.dot(r12_u, v1)**2) + (-4*np.dot(r12_u, v1)*np.dot(r12_u, v2)) + (-6*np.dot(r12_u, v2)**2) + (-8*np.dot(v1,v2)) + (4*np.dot(v2,v2))))*r12_u #Term 1/c^4 Unit radial direction
    
        #a4_v = (G*m2/(r12**(2.)))*(( (G*m2/r12)*((-2*np.dot(r12_u, v1))  + (np.dot(r12_u,v2))) + (G*m2/(4*r12))*( (-63.*np.dot(r12_u, v1)) + (55.*np.dot(r12_u, v2))) + np.dot(r12_u,v2)*( (4*np.dot(v1,v2)) + (np.dot(v1,v1)) + (-5.*np.dot(r12_u,v2)*np.dot(v2,v2)) + (9./2*np.dot(r12_u,v2)**2 ) ) + np.dot(r12_u,v1)*( (-6.*np.dot(r12_u, v2)**2) + (-4.*np.dot(v1,v2)) + (4.*np.dot(v2,v2)) ) )*v12) #Term 1/c^4 Velocity direction

        #a5_r = ((4./5)*(G**2)*m2*m1/(r12**(3.)))*(np.dot(r12_u,v12))*((52.0*G*m2/(3.0*r12) - 6.0*G*m1/r12 + 3.0*np.dot(v12,v12))*r12_u) #Term 1/c^5 Unit radial direction

        #a5_v = ((4./5)*(G**2)*m2*m1/(r12**(3.)))*((2*G*m1/r12 - 8.0*G*m2/r12 - np.dot(v12,v12))*v12) #Term 1/c^5 Velocity direction

        #Contribuciones

        #a2 = (1./(c_u**2))*(a2_r + a2_v)

        #a4 = (1./(c_u**4))*(a4_r + a4_v)

        #a5 = (1./(c_u**5))*(a5_r + a5_v)
