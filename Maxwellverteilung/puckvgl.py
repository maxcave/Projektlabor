#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 01:46:18 2019

@author: cave
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.optimize import curve_fit
#define fit function
def maxwell(x, A):
    return x * A * np.exp(-1 * (A*(x)**2 / 2))
def sigma(ns, wbs, popts):
    r = [np.sqrt((y - maxwell(x, popts))**2) for y, x in zip(ns, wbs)]
    return np.mean(r)
def rmse(ns, wbs, popts):
   a = [(y - maxwell(x, popts))**2 for y,x in zip(ns,wbs)]
   return np.sqrt(np.mean(a))
xspace = np.linspace(0,25,1000)
#############################
dm9gh=pd.read_csv("M9GH_Geschwindigkeiten.csv", sep="\t")
sm9gh=dm9gh["v"]
dm9gh3=pd.read_csv("M9GH3_Geschwindigkeiten.csv", sep="\t")
sm9gh3=dm9gh3["v"]

nzm9gh = [x for x in sm9gh if x > 0.48]
nzm9gh3 = [x for x in sm9gh3 if x > 0.27]

plt.subplot(1,2,1)
nm9gh, binsm9gh, patchesm9gh = plt.hist(nzm9gh, bins=1000, density=True, histtype="bar", label="Messwerte")
wbm9gh = [(binsm9gh[x] + binsm9gh[x+1])/2 for x in range(len(binsm9gh)-1)]
poptmm9gh, pcovmm9gh = curve_fit(maxwell, wbm9gh, nm9gh, absolute_sigma=True)
plt.plot(xspace, maxwell(xspace, poptmm9gh), color="orange", label="Fit A={}".format(round(poptmm9gh[0],2)))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.ylim(0,0.15)
plt.legend()
plt.title("6 Pucks")
plt.rcParams["legend.loc"]="upper right"
plt.xlabel("v [Pixel/Frame]")
plt.ylabel("Relative Häufigkeit")
plt.xlim(0, 25)
print('pcovmm9gh = ', pcovmm9gh, "A", poptmm9gh, "\n sigma", sigma(nm9gh, wbm9gh, poptmm9gh))

plt.subplot(1,2,2)
nm9gh3, binsm9gh3, patchesm9gh3 = plt.hist(nzm9gh3, bins=1000, density=True, histtype="bar", label="Messwerte")
wbm9gh3 = [(binsm9gh3[x] + binsm9gh3[x+1])/2 for x in range(len(binsm9gh3)-1)]
poptmm9gh3, pcovmm9gh3 = curve_fit(maxwell, wbm9gh3, nm9gh3, absolute_sigma=True)
plt.plot(xspace, maxwell(xspace, +poptmm9gh3), color="orange", label="Fit A={}".format(round(poptmm9gh3[0],2)))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.ylim(0,0.15)
plt.title("3 Pucks")
plt.legend()
plt.xlabel("v [Frame]")
plt.ylabel("Relative Häufigkeit")
plt.xlim(0, 25)
print("pcovmm9gh3 = ", pcovmm9gh3, "A", poptmm9gh3, "\n sigma", sigma(nm9gh3, wbm9gh3, poptmm9gh3))

plt.tight_layout()
plt.rcParams.update({'font.size': 8})
#plt.savefig("puckvgl", dpi = 500)
plt.show()
