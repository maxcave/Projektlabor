import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
pd.set_option('display.max_columns', 30)
selection = int(input("Make your selection: \n-1 for raw data \n-2 for grouped averages \n-3 for grouped maximums "))
rounding = int(input("How many decimal points of precision? (Don't be a dick and just use integers)"))
#hier sind unsere essergebnisse, einmal die standard luftdichte aus dem internet als konstante und danach eine dictionary mit den Essergebnissen. Die dictionary wird in ein pandas Dataframe gepackt damit man leichter mit den Daten umgehen kann
rho = 1.225 #kg/m^3
laenge = 0.205 #m
daten =  {
        'Messreihe': ['Metall', 'Metall', 'Plastik', 'Plastik', 'Plastik', 'Plastik', 'Plastik', 'Plastik', 'Plastik', 'Plastik', 'Gitter', 'Gitter',  'Gitter', 'Gitter', 'Gitter', 'Gitter', 'Schmirgelpapier'],
        'Druckunterschied [mbar]' : [2.4, 2.8, 2, 1.7, 0.9, 1.4, 1.7, 1.1, 1.8, 2, 2.3, 1.8, 1.9, 1.5, 1.4, 1, 2.2],
        'Druckfehler [rel]' : 0.1,
        'Frequenz [Hz]' : [50, 51, 48, 53, 54, 53, 54, 54, 55, 56, 38, 38, 37, 36, 36, 39, 49],
        'Frequenzfehler [Hz]' : 1,
        'Kraft [N]' : [0.2, 0.2, 0.45, 0.5, 0.4, 0.6, 0.65, 0.6, 0.4, 0.4, 0.5, 0.5, 0.45, 0.4, 0.4, 0.35, 0.8],
        'Kraftfehler [N]': 0.05,
        'Radius [m]': [0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046],
        'Radiusfehler [m]': 0.005
        }
df = pd.DataFrame(daten, columns = ['Messreihe', 'Druckunterschied [mbar]','Druckfehler [rel]', 'Frequenz [Hz]', 'Frequenzfehler [Hz]', 'Kraft [N]', 'Kraftfehler [N]', 'Radius [m]', 'Radiusfehler [m]'])
#Dem Dataframe wird eine spalte hinzugefügt welches die luftgeschwindigkeit enthält
df['airspeed [m/s]'] = [np.sqrt(200 * p/rho) for p in df['Druckunterschied [mbar]']]
df['airspeedfehler'] = [0.5 * np.sqrt(200/(rho * daten['Druckfehler [rel]'] * p)) for p in df['Druckunterschied [mbar]']]
#dem Dataframe wird eine Spalte mit dem Reibungskoeffizienten hinzugefügt
df['Reibungskoeffizient'] = [F/(4*1.2*np.pi * (r)**2 * laenge * f * v) for F, r, f, v in zip(df['Kraft [N]'], df['Radius [m]'], df['Frequenz [Hz]'], df['airspeed [m/s]'])]

df['Koeffizientenfehler'] = [np.sqrt(
        ((daten['Kraftfehler [N]'])/(4*rho*v*2*np.pi*f*r**2 * laenge))**2 + \
        ((-F*vf)/(4*rho*v**2 *2*np.pi*f*r**2 *laenge))**2 + \
        ((-F*daten['Frequenzfehler [Hz]']*2*np.pi)/(4*rho*v*(2*np.pi*f)**2 *r**2 * laenge))**2 + \
        ((-2*F*daten['Radiusfehler [m]'])/(4*rho*v*2*np.pi*f*r**3 *laenge))**2 + \
        ((-F*0.005)/(4*rho*v*2*np.pi*f*r**2 *laenge**2))**2 \
        ) for v, f, r, F, vf in \
        zip(df['airspeed [m/s]'],  df['Frequenz [Hz]'], df['Radius [m]'], df['Kraft [N]'], df['airspeedfehler'])]

if selection == 1:
    print(df)
elif selection == 2:
    print(round(df.groupby('Messreihe').mean(), rounding))
elif selection == 3:
    print(round(df.groupby('Messreihe').max(), rounding))
