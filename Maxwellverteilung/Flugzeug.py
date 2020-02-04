import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

selection = int(input("What do you want to do, 1 for raw data, 2 for describe? \n"))
#constant measurements
rho=1.225#kg/m^3
length=0.81#m pm 0.01
radius=0.08#m pm 0.005
#Magnuskraft
weightdiff_kg = [0.048, 0.9, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02, 0.015]
magnusforce_N = [round(9.81*m, 2) for m in weightdiff_kg]
#Windgeschwindigkeit
pressurediff_mbar = [3, 3, 3, 2, 3, 3, 3, 2, 3]
airspeed_mps = [round(np.sqrt(200* p/rho), 2) for p in pressurediff_mbar]
#coefficient
fcoefficient = [F/(4*1.2*np.pi * (radius)**2 * length  * v) for F, v in zip(magnusforce_N, airspeed_mps)] 
#errors
weighterr_kg = 0.1
pressureerr_rel = 0.1 
#propagation
magnuserr = 0.981
airspeederr_mps = [0.5 * np.sqrt(200/(rho * pressureerr_rel * p)) for p in pressurediff_mbar]
#aggregate
agg = list(zip(magnusforce_N, airspeed_mps, airspeederr_mps, fcoefficient))
df= pd.DataFrame(agg, columns = ['Magnusforce [N]', 'Airspeed[m/s]', 'Airspeederr[m/s]', 'f*k [Hz]'])
if selection == 1:
    print(df, '\n', magnuserr)
elif selection == 2:
    print(df.describe(), '\n', magnuserr)
