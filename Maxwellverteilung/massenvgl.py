#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:16:32 2019

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
dm9gg=pd.read_csv("M9GG_Geschwindigkeiten.csv", sep="\t")
sm9gg=dm9gg["v"]

nzm9gh = [x for x in sm9gh if x > 0.48]
nzm9gg = [x for x in sm9gg if x > 0.27]

plt.subplot(1,2,1)
nm9gh, binsm9gh, patchesm9gh = plt.hist(nzm9gh, bins=100, density=True, histtype="bar")
wbm9gh = [(binsm9gh[x] + binsm9gh[x+1])/2 for x in range(len(binsm9gh)-1)]
poptmm9gh, pcovmm9gh = curve_fit(maxwell, wbm9gh, nm9gh, absolute_sigma=True)
plt.plot(xspace, maxwell(xspace, poptmm9gh), color="orange", label="Fit A={}".format(round(poptmm9gh[0],2)))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.ylim(0,0.15)
plt.legend()
plt.title("m\u2081 = 32g")
plt.rcParams["legend.loc"]="upper right"
plt.xlabel("v [Pixel/Frame")
plt.ylabel("Häufigkeit")
plt.xlim(0, 25)
print("pcovmm9gh = ", pcovmm9gh, "A", poptmm9gh)

plt.subplot(1,2,2)
nm9gg, binsm9gg, patchesm9gg = plt.hist(nzm9gg, bins=binsm9gh, density=True, histtype="bar")
wbm9gg = [(binsm9gg[x] + binsm9gg[x+1])/2 for x in range(len(binsm9gg)-1)]
poptmm9gg, pcovmm9gg = curve_fit(maxwell, wbm9gg, nm9gg, absolute_sigma=True)
plt.plot(xspace, maxwell(xspace, poptmm9gg), color="orange", label="Fit A={}".format(round(poptmm9gg[0],2)))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.ylim(0,0.15)
plt.title("m\u2082 = 19g")
plt.legend()
plt.xlabel("v [Pixel/Frame]")
plt.ylabel("Häufigkeit")
plt.xlim(0, 25)
print("pcovmm9gg = ", pcovmm9gg, "A", poptmm9gg)

plt.tight_layout()
plt.rcParams.update({'font.size': 8})
plt.savefig("../grafiken/massenvgl", dpi = 500)
plt.show()
