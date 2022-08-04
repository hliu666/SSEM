# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:18:52 2022

@author: 16072
"""
import numpy as np
import pandas as pd 

class SSEM_Model:
    def __init__(self, driving):
        self.PAR = np.array(driving['PAR'])
        self.TA  = np.array(driving['Tair'])
        
        self.Bwood = 14500 * 1e-6 * 10000 ## convert g/m2 -> Mg/ha
        self.Bleaf = 2950 * 0.01  ## generates a LAI that is too high ***
        self.SOM = np.mean((1.575 + 0.94 + 2.32) * 1e-3 * 10000) ## sum up litter, CWD, and soil; change units (US-ME2)
        self.X = [self.Bleaf, self.Bwood, self.SOM]

        self.alpha = 0.02
        self.Q10 = 2.0
        self.Rbasal = 0.15
        self.falloc_1 = 0.5
        self.falloc_2 = 0.4
        self.falloc_3 = 0.1
        self.sigma_leaf = 0.012
        self.sigma_stem = 0.002
        self.sigma_soil = 0.001
        self.SLA = 4.5
        self.litterfall = 1.25e-5
        self.mortality = 4.0e-7
        
        self.params = [self.alpha, self.Q10, self.Rbasal, self.falloc_1, self.falloc_2, self.falloc_3,\
                    self.sigma_leaf, self.sigma_stem, self.sigma_soil, self.SLA, self.litterfall, self.mortality]
      
    def mod_list(self):
        GPP_list, LAI_list = [], []
        for i in range(len(self.PAR)):
            inputs = [self.PAR[i], self.TA[i]]
            _, _, self.outputs = self.run_SSEM(self.X, self.params, inputs, timestep = 1800)
            LAI_list.append(self.outputs[0])
            GPP_list.append(self.outputs[1])

        return GPP_list, LAI_list
    
    def run_SSEM(self, X, params, inputs, timestep = 1800):
          
          ##Unit Converstion: umol/m2/sec to Mg/ha/timestep
          k = 1e-6 * 12 * 1e-6 * 10000 * timestep #mol/umol*gC/mol*Mg/g*m2/ha*sec/timestep
          
          PAR, temp = inputs
          
          ## photosynthesis
          LAI = X[0] * self.SLA * 0.1  #0.1 is conversion from Mg/ha to kg/m2
          if(PAR > 1e-20):
            GPP = max(0, self.alpha * (1 - np.exp(-0.5 * LAI)) * PAR)
          else:
            GPP = 0
              
          ## respiration & allocation
          alloc = [GPP* self.falloc_1, GPP* self.falloc_2, GPP* self.falloc_3] ## Ra, NPPwood, NPPleaf
          Rh = max(self.Rbasal * X[2] * self.Q10 ** (temp / 10), 0) ## pmax ensures SOM never goes negative
          
          ## turnover
          self.litterfall = X[0] * self.litterfall
          self.mortality = X[1] * self.mortality
          ## update states
          X1 = max(X[0] + alloc[2] * k - self.litterfall, self.sigma_leaf, 0)
          X2 = max(X[1] + alloc[1] * k - self.mortality, self.sigma_stem, 0)
          X3 = max(X[2] + self.litterfall + self.mortality - Rh * k, self.sigma_soil, 0)
          
          
          #return parameters 
          LAI = X1 * self.SLA * 0.1
          NEP = GPP - alloc[0] - Rh
          Ra = alloc[0]
          NPPw = alloc[1]
          NPPl = alloc[2]
      
          X = [X1, X2, X3]
          params = [self.alpha, self.Q10, self.Rbasal, self.falloc_1, self.falloc_2, self.falloc_3,\
                      self.sigma_leaf, self.sigma_stem, self.sigma_soil, self.SLA, self.litterfall, self.mortality]
          outputs = [LAI, GPP, NEP, Ra, NPPw, NPPl, Rh]   
          return X, params, outputs


df = pd.read_csv("C:/Users/16072/Desktop/FluxcourseUSB/FluxCourseForecast/data/US-xHA.csv")
df = df[(df['year']>=2019)&(df['year']<2020)]
driving = df[['DoY','Hour','Tair', 'Rg', 'GPP_uStar_f']]
driving['PAR'] = driving['Rg']/0.486
m = SSEM_Model(driving)
gpp, lai = m.mod_list()
driving['GPP_Sim'] = gpp
driving['LAI_Sim'] = lai

driving = driving[driving['Rg']>=10]
daily = driving.groupby(['DoY']).mean()

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(daily['LAI_Sim'])
fig = plt.figure()
plt.plot(daily['GPP_Sim'])  
plt.plot(daily['GPP_uStar_f'])  
        