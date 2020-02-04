#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 12:52:00 2019

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
    return np.average(r)
def rmse(ns, wbs, popts):
   a = [(y - maxwell(x, popts))**2 for y,x in zip(ns,wbs)]
   return np.sqrt(np.mean(a))

xspace = np.linspace(0,25,1000)
#############################
dm6gh=pd.read_csv("M6GH_Geschwindigkeiten.csv", sep="\t")
sm6gh=dm6gh["v"]

dm7gh=pd.read_csv("M7GH_Geschwindigkeiten.csv", sep="\t")
sm7gh=dm7gh["v"]

dm8gh=pd.read_csv("M8GH_Geschwindigkeiten.csv", sep="\t")
sm8gh=dm8gh["v"]

dm9gh=pd.read_csv("M9GH_Geschwindigkeiten.csv", sep="\t")
sm9gh=dm9gh["v"]

###########################################
nm6gh, binsm6gh, patchesm6gh = plt.hist(sm6gh, bins=300, density=True, histtype="step", label="Messwerte")
nm7gh, binsm7gh, patchesm7gh = plt.hist(sm7gh, bins=300, density=True, histtype="step", label="Messwerte")
nm8gh, binsm8gh, patchesm8gh = plt.hist(sm8gh, bins=300, density=True, histtype="step", label="Messwerte")
nm9gh, binsm9gh, patchesm9gh = plt.hist(sm9gh, bins=300, density=True, histtype="step", label="Messwerte")
#plt.show()

wbm6gh = [(binsm6gh[x] + binsm6gh[x+1])/2 for x in range(len(binsm6gh)-1)]
wbm7gh = [(binsm7gh[x] + binsm7gh[x+1])/2 for x in range(len(binsm7gh)-1)]
wbm8gh = [(binsm8gh[x] + binsm8gh[x+1])/2 for x in range(len(binsm8gh)-1)]
wbm9gh = [(binsm9gh[x] + binsm9gh[x+1])/2 for x in range(len(binsm9gh)-1)]

poptmm6gh, pcovmm6gh = curve_fit(maxwell, wbm6gh, nm6gh, absolute_sigma=True)
poptmm7gh, pcovmm7gh = curve_fit(maxwell, wbm7gh, nm7gh, absolute_sigma=True)
poptmm8gh, pcovmm8gh = curve_fit(maxwell, wbm8gh, nm8gh, absolute_sigma=True)
poptmm9gh, pcovmm9gh = curve_fit(maxwell, wbm9gh, nm9gh, absolute_sigma=True)

a6=round(poptmm6gh[0],3)
a7=round(poptmm7gh[0],3)
a8=round(poptmm8gh[0],3)
a9=round(poptmm9gh[0],3)

plt.plot(xspace, maxwell(xspace, +poptmm6gh), color="green", label="Fit M6 A={}".format(a6))
plt.plot(xspace, maxwell(xspace, +poptmm7gh), color="red", label="Fit M7 A={}".format(a7))
plt.plot(xspace, maxwell(xspace, +poptmm8gh), color="orange", label="Fit M8 A={}".format(a8))
plt.plot(xspace, maxwell(xspace, +poptmm9gh), color="blue", label="Fit M9 A={}".format(a9))

print("pcovmm6gh", pcovmm6gh)
"""
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlim(0,)
plt.ylim(0,)
plt.xlabel("v in [Px/Frame]")
plt.ylabel("Relative Häufigkeit")
plt.legend()
plt.tight_layout()
plt.rcParams.update({'font.size': 8})
#plt.savefig("geschwindigkeitsvgl", dpi = 500)
plt.show()
"""
