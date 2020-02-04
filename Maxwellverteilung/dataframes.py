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
    return np.average(r)

xspace = np.linspace(0,25,1000)
#############################
dm6gh=pd.read_csv("M6GH_Geschwindigkeiten.csv", sep="\t")
sm6gh=dm6gh["v"]

plt.subplot(1,1,1)
nm6gh, binsm6gh, patchesm6gh = plt.hist(sm6gh, bins=300, density=True, histtype="step", label="Messwerte")
plt.show(block=True)
#plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

wbm6gh = [(binsm6gh[x] + binsm6gh[x+1])/2 for x in range(len(binsm6gh)-1)]
plt.subplot(1,1,1)
poptmm6gh, pcovmm6gh = curve_fit(maxwell, wbm6gh, nm6gh)
plt.plot(xspace, maxwell(xspace, +poptmm6gh), color="green", label="Fit")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()

###############################
dm7gh=pd.read_csv("M7GH_Geschwindigkeiten.csv", sep="\t")
sm7gh=dm7gh["v"]

#plt.subplot(1,1,1)
nm7gh, binsm7gh, patchesm7gh = plt.hist(sm7gh, bins=1000, density=True, histtype="step", label="Messwerte")
plt.show(block=True)

wbm7gh = [(binsm7gh[x] + binsm7gh[x+1])/2 for x in range(len(binsm7gh)-1)]

plt.subplot(1,1,1)
poptmm7gh, pcovmm7gh = curve_fit(maxwell, wbm7gh, nm7gh)
plt.plot(xspace, maxwell(xspace, +poptmm7gh), color="red", label="Fit")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()
#plt.ylim(0,)
#plt.legend()
#plt.xlabel("")
#plt.title("Motorstärke 7")
#plt.xlim(0, 25)
##############################
dm8gh=pd.read_csv("M8GH_Geschwindigkeiten.csv", sep="\t")
sm8gh=dm8gh["v"]

#plt.subplot(1,1,1)
nm8gh, binsm8gh, patchesm8gh = plt.hist(sm8gh, bins=1000, density=True, histtype="step", label="Messwerte")
plt.show(block=False)

wbm8gh = [(binsm8gh[x] + binsm8gh[x+1])/2 for x in range(len(binsm8gh)-1)]

plt.subplot(1,1,1)
poptmm8gh, pcovmm8gh = curve_fit(maxwell, wbm8gh, nm8gh)
plt.plot(xspace, maxwell(xspace, +poptmm8gh), color="orange", label="Fit")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()
#plt.ylim(0,)
#plt.legend()
#plt.xlabel("v")
#plt.title("Motorstärke 8")
#plt.xlim(0, 25)
###########################
dm9gh=pd.read_csv("M9GH_Geschwindigkeiten.csv", sep="\t")
sm9gh=dm9gh["v"]
nzm9gh = [x for x in sm9gh if x > 0.0635]

plt.subplot(1,1,1)
nm9gh, binsm9gh, patchesm9gh = plt.hist(nzm9gh, bins=1000, density=True, histtype="step", label="Messwerte")
plt.show(block=True)

wbm9gh = [(binsm9gh[x] + binsm9gh[x+1])/2 for x in range(len(binsm9gh)-1)]
plt.subplot(1,1,1)
poptmm9gh, pcovmm9gh = curve_fit(maxwell, wbm9gh, nm9gh)
plt.plot(xspace, maxwell(xspace, +poptmm9gh), color="blue", label="Fit")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()
#plt.ylim(0,)
#plt.legend()
#plt.xlabel("v")
#plt.title("Motorstärke 9")
#plt.xlim(0, 25)
#############################

dm9gg=pd.read_csv("M9GG_Geschwindigkeiten.csv", sep="\t")
sm9gg=dm9gg["v"]
#nzm9gg = [x for x in sm9gg if x > 0.0475]
#
#plt.subplot(1,2,1)
#nm9gg, binsm9gg, patchesm9gg = plt.hist(nzm9gg, bins=1000, density=True, histtype="step", label="m2 = 19g")
#plt.gca().yaxis.set_major_formatter(PercentFormatter(1))



#wbm9gg = [(binsm9gg[x] + binsm9gg[x+1])/2 for x in range(len(binsm9gg)-1)]
#plt.subplot(1,2,2)
#poptmm9gg, pcovmm9gg = curve_fit(maxwell, wbm9gg, nm9gg)
#plt.plot(xspace, maxwell(xspace, +poptmm9gg), color="orange", label="Fit m2")

###############################
dm9gh3=pd.read_csv("M9GH3_Geschwindigkeiten.csv", sep="\t")
sm9gh3=dm9gh3["v"]
#
#plt.subplot(1,2,2)
#nm9gh3, binsm9gh3, patchesm9gh3 = plt.hist(sm9gh3, bins=100, density=True, histtype="bar", label="Messwerte")
#plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
#
#wbm9gh3 = [(binsm9gh3[x] + binsm9gh3[x+1])/2 for x in range(len(binsm9gh3)-1)]
#
#poptmm9gh3, pcovmm9gh3 = curve_fit(maxwell, wbm9gh3, nm9gh3)
#plt.plot(xspace, maxwell(xspace, +poptmm9gh3), color="orange", label="Maxwell Fit")

#plt.subplots_adjust(wspace = 0.3, hspace=0.4)
#plt.rcParams.update({'font.size': 11}) #isn't interpreted by console
#plt.savefig("motorsubvgl", dpi = 500)

