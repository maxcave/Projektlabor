#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 19:23:50 2019

@author: cave
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.optimize import curve_fit

dm6gh=pd.read_csv("M6GH_Geschwindigkeiten.csv", sep="\t")
sm6gh=dm6gh["v"]
dm7gh=pd.read_csv("M7GH_Geschwindigkeiten.csv", sep="\t")
sm7gh=dm7gh["v"]
dm8gh=pd.read_csv("M8GH_Geschwindigkeiten.csv", sep="\t")
sm8gh=dm8gh["v"]
dm9gh=pd.read_csv("M9GH_Geschwindigkeiten.csv", sep="\t")
sm9gh=dm9gh["v"]

nzm6gh = [x for x in sm6gh if x > 0.065]
nzm7gh = [x for x in sm7gh if x > 0.039]
nzm8gh = [x for x in sm8gh if x > 0.045]
nzm9gh = [x for x in sm9gh if x > 0.063]


def maxwell(x, A):
    return x * A * np.exp(-1 * (A*(x)**2 / 2))

xspace = np.linspace(0,25,1000)
plt.subplot(2,2,1)
nm6gh, binsm6gh, patchesm6gh = plt.hist(nzm6gh, bins=100, density=True, histtype="bar")
wbm6gh = [(binsm6gh[x] + binsm6gh[x+1])/2 for x in range(len(binsm6gh)-1)]
poptmm6gh, pcovmm6gh = curve_fit(maxwell, wbm6gh, nm6gh, absolute_sigma=True)
plt.plot(xspace, maxwell(xspace, +poptmm6gh), color="orange", label="Fit A={}".format(round(poptmm6gh[0],2)))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlim(0,25)
plt.ylim(0,0.2)
plt.legend()
plt.title("Motorstärke 6")
plt.xlabel("v [Pixel/Frame]")
plt.ylabel("Häufigkeit")


plt.subplot(2,2,2)
nm7gh, binsm7gh, patchesm7gh = plt.hist(nzm7gh, bins=100, density=True, histtype="bar")
wbm7gh = [(binsm7gh[x] + binsm7gh[x+1])/2 for x in range(len(binsm7gh)-1)]
poptmm7gh, pcovmm7gh = curve_fit(maxwell, wbm7gh, nm7gh, absolute_sigma=True)
plt.plot(xspace, maxwell(xspace, +poptmm7gh), color="orange", label="Fit A={}".format(round(poptmm7gh[0],2)))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlim(0,25)
plt.ylim(0,0.2)
plt.title("Motorstärke 7")
plt.legend()
plt.xlabel("v [Pixel/Frame]")
plt.ylabel("Häufigkeit")

plt.subplot(2,2,3)
nm8gh, binsm8gh, patchesm8gh = plt.hist(nzm8gh, bins=100, density=True, histtype="bar")
wbm8gh = [(binsm8gh[x] + binsm8gh[x+1])/2 for x in range(len(binsm8gh)-1)]
poptmm8gh, pcovmm8gh = curve_fit(maxwell, wbm8gh, nm8gh, absolute_sigma=True)
plt.plot(xspace, maxwell(xspace, +poptmm8gh), color="orange", label="Fit A={}".format(round(poptmm8gh[0],2)))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlim(0,25)
plt.ylim(0,0.2)
plt.title("Motorstärke 8")
plt.legend()
plt.xlabel("v [Pixel/Frame]")
plt.ylabel("Häufigkeit")

plt.subplot(2,2,4)
nm9gh, binsm9gh, patchesm9gh = plt.hist(nzm9gh, bins=100, density=True, histtype="bar")
wbm9gh = [(binsm9gh[x] + binsm9gh[x+1])/2 for x in range(len(binsm9gh)-1)]
poptmm9gh, pcovmm9gh = curve_fit(maxwell, wbm9gh, nm9gh, absolute_sigma=True)
plt.plot(xspace, maxwell(xspace, +poptmm9gh), color="orange", label="Fit A={}".format(round(poptmm9gh[0],2)))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlim(0,25)
plt.ylim(0,0.2)
plt.legend()
plt.title("Motorstärke 9")
plt.xlabel("v [Pixel/Frame]")
plt.ylabel("Häufigkeit")

print("M6cov ", pcovmm6gh, "A", poptmm6gh) 
print("M7cov ", pcovmm7gh, "A", poptmm7gh) 
print("M8cov ", pcovmm8gh, "A", poptmm8gh)
print("M9cov ", pcovmm9gh, "A", poptmm9gh) 
print(binsm6gh[1], binsm7gh[1], binsm8gh[1],binsm9gh[1])
plt.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.rcParams.update({'font.size': 9})
plt.savefig("../grafiken/motorsubvgl", dpi = 500)
plt.show()
