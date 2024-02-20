import numpy as np
import functions as f


material='Si' #Set detector material as 'Si' or 'Ge'

########Set parameters, either as scalars or arrays of the same length#########
####Short range parameters####
c1s = 1
c3s = 0
c4s = 0
c5s = 0
c6s = 0
c7s = 0
c8s = 0
c9s = 0
c10s = 0
c11s = 0
c12s = 0
c13s = 0
c14s = 0
c15s = 0

####Long range parameters####
c1l = 0
c3l = 0
c4l = 0
c5l = 0
c6l = 0
c7l = 0
c8l = 0
c9l = 0
c10l = 0
c11l = 0
c12l = 0
c13l = 0
c14l = 0
c15l = 0


m_x = 10  #DM mass in MeV. Must be between 3 and 1000 for Ge and 4 and 1000 for Si



#Returns the number of events per kg*year for the first 4 bins.
rates=f.inference(material,c1s,c3s,c4s,c5s,c6s,c7s,c8s,c9s,c10s,c11s,c12s,c13s,c14s,c15s,
            c1l,c3l,c4l,c5l,c6l,c7l,c8l,c9l,c10l,c11l,c12l,c13l,c14l,c15l,m_x)
print(rates)
#If input parameters are single scalars, rates[j] is the is the rate of events with j+1 electron hole pairs.
#If given multiple input parameters at once, rates[i,j] is the rate of events with j+1 electron hole pairs
#produced by parameter combination i.


