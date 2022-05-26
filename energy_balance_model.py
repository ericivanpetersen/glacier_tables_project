#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"))))

import numpy as np 
import pandas as pd 
from Rounce_EBM.meltmodel_global import CrankNicholson, calc_surface_fluxes, calc_surface_fluxes_cleanice
import json
from aws_tools import calc_temp_blackbody
from matplotlib import pyplot as plt 
import matplotlib.dates as mdates

class energy_balance_model(object):

    """
    Class to contain all parameters and variables
    relevant to a glacier surface energy balance 
    model, and routines to act on them.
    """

    def __init__(self, proj_filepath):

        # Read in project file:
        proj_file = open(proj_filepath)
        proj = json.load(proj_file)
        
        # Define values from project file:
        # Debris Properties:
        self.k = proj["k"] # Thermal conductivity debris layer, W/(m K)
        self.albedo = proj["albedo"] # Albedo of debris
        self.a_bulk = proj["a_bulk"] # Bulk transfer coefficient for turbulent fluxes
        self.z_meas = proj["z_meas"] # Height of meteorological measurements (T, RH, u)
        self.emissivity = proj["emissivity"] # debris layer emissivity
        self.rho_d = proj["rho_d"] # Bulk density of debris layer, kg/m^3
        self.c_d = proj["c_d"] # Specific heat capacity of debris, J/(kg K)

        #self.dt = proj["delta_t"] # length of timestep in seconds.
        # AWS data filepath:
        self.aws_file = proj["aws_file"]

        # Load in AWS data:
        self.aws_data = pd.read_csv(self.aws_file, parse_dates=True, index_col=0)
        self.timestamp = self.aws_data.index
        self.nsteps = len(self.timestamp)

        self.dt = (self.timestamp[1] - self.timestamp[0]).total_seconds() # timestep in seconds
        print('Timestep = {}'.format(self.dt))

        # Define met variables from AWS data:
        self.SW_sky = self.aws_data[proj["SW_sky"]]
        #self.SW_ground = self.aws_data[proj["SW_ground"]]
        self.LW_sky = self.aws_data[proj["LW_sky"]]
        self.LW_ground = self.aws_data[proj["LW_ground"]]
        self.T_air = self.aws_data[proj["T_air"]] + 273.15 # Temperature in K
        self.P = self.aws_data[proj["Pressure"]] * 100
        self.RH_air = self.aws_data[proj["RH_air"]] / 100
        self.u = self.aws_data[proj["u"]]

        # Derived values:
        self.T_srf = calc_temp_blackbody(self.LW_ground, self.LW_sky, eps=self.emissivity) + 273.15

        return
    
    def calc_energy_flux_melt(self, debris_thickness, stake_file=None, time_zero=None, n_iter_max=1000, plot=True, plot_period=None, save_dir=None):
        """
        Use Crank-Nicholson scheme and calculation of 
        surface fluxes to determin sub-debris melt 
        for a given debris thickness (in m).
        """

        # Enhancement factors for heat fluxes based on geometry of the supraglacial rock
        EF = 1 # General enhancement factor for long wave radiation and sensible heat fluxes
        EF_SW = 1 # Enhancement factor for shortwave radiation

        nsteps = self.nsteps
        if debris_thickness>0:
            h = debris_thickness/10
            N = int(debris_thickness/h + 1)
            # ===== DEBRIS-COVERED GLACIER ENERGY BALANCE MODEL =====
            # Constant defined by Reid and Brock (2010) for Crank-Nicholson Scheme
            C = self.k * self.dt / (2 * self.rho_d * self.c_d * h**2)
            # "Crank Nicholson Newton Raphson" Method for LE Rain
            # Compute Ts from surface energy balance model using Newton-Raphson Method at each time step and Td
            # at all points in debris layer
            Td = np.zeros((N, nsteps))
            a_Crank = np.zeros((N,nsteps))
            b_Crank = np.zeros((N,nsteps))
            c_Crank = np.zeros((N,nsteps))
            d_Crank = np.zeros((N,nsteps))
            A_Crank = np.zeros((N,nsteps))
            S_Crank = np.zeros((N,nsteps))
        else:
            Td = np.zeros((nsteps))

        n_iterations = np.zeros((nsteps))
        Ts_past = np.zeros((nsteps))
        LE = np.zeros((nsteps))
        Rn = np.zeros((nsteps))
        H_flux = np.zeros((nsteps))
        Qc = np.zeros((nsteps))
        P_flux = np.zeros((nsteps))
        dLE = np.zeros((nsteps))
        dRn = np.zeros((nsteps))
        dH_flux = np.zeros((nsteps))
        dQc = np.zeros((nsteps))
        dP_flux = np.zeros((nsteps))
        F_Ts = np.zeros((nsteps))
        dF_Ts = np.zeros((nsteps))
        Qc_ice = np.zeros((nsteps))
        Melt = np.zeros((nsteps))
        dsnow = np.zeros((nsteps))      # snow depth [mwe]
        tsnow = np.zeros((nsteps))      # snow temperature [K]
        snow_tau = np.zeros((nsteps))   # non-dimensional snow age

        if debris_thickness > 0:
            for i in np.arange(self.nsteps):

                Td[N-1,i] = 273.15 # pin the ice surface at the freezing/melting point

                # Initially assume Ts = Prediction from LW, for all other time steps assume it's equal to previous Ts
                if i == 0:
                    Td[0,i] = self.T_srf[i]
                else:
                    Td[0,i] = Td[0,i-1]

                # Calculate debris temperature profile for timestep i
                Td = CrankNicholson(Td, self.T_air, i, debris_thickness, N, h, C, a_Crank, b_Crank, c_Crank,
                                    d_Crank, A_Crank, S_Crank)

                # Surface energy fluxes
                (F_Ts[i], Rn[i], LE[i], H_flux[i], P_flux[i], Qc[i], dF_Ts[i], dRn[i], dLE[i], dH_flux[i],
                    dP_flux[i], dQc[i], dsnow[i], tsnow[i], snow_tau[i]) = (
                        calc_surface_fluxes(Td[:,i], self.T_air[i], self.RH_air[i], self.u[i], self.SW_sky[i], self.LW_sky[i],
                                            0, 0, self.P[i], self.albedo, self.k, self.a_bulk,
                                            h, 0, 0, 0, 0,
                                            0, debris_thickness,
                                            option_snow=0,
                                            option_snow_fromAWS=0, i_step=i, EF=EF, EF_SW=EF_SW))
                
                # Newton-Raphson method to solve for surface temperature
                while abs(Td[0,i] - Ts_past[i]) > 0.01 and n_iterations[i] < n_iter_max:

                    n_iterations[i] = n_iterations[i] + 1
                    Ts_past[i] = Td[0,i]
                    # max step size is 1 degree C
                    Td[0,i] = Ts_past[i] - F_Ts[i] /dF_Ts[i]
                    if (Td[0,i] - Ts_past[i]) > 1:
                        Td[0,i] = Ts_past[i] + 1
                    elif (Td[0,i] - Ts_past[i]) < -1:
                        Td[0,i] = Ts_past[i] - 1

                    # Debris temperature profile for timestep i
                    Td = CrankNicholson(Td, self.T_air, i, debris_thickness, N, h, C, a_Crank, b_Crank, c_Crank,
                                        d_Crank, A_Crank, S_Crank)

                    # Surface energy fluxes
                    (F_Ts[i], Rn[i], LE[i], H_flux[i], P_flux[i], Qc[i], dF_Ts[i], dRn[i], dLE[i], dH_flux[i],
                        dP_flux[i], dQc[i], dsnow[i], tsnow[i], snow_tau[i]) = (
                            calc_surface_fluxes(Td[:,i], self.T_air[i], self.RH_air[i], self.u[i], self.SW_sky[i], self.LW_sky[i],
                                                0, 0, self.P[i], self.albedo, self.k, self.a_bulk,
                                                h, 0, 0, 0, 0,
                                                0, debris_thickness,
                                                option_snow=0,
                                                option_snow_fromAWS=0, i_step=i, EF=EF, EF_SW=EF_SW))

                    if n_iterations[i] == n_iter_max:
                        Td[0,i] = (Td[0,i] + Ts_past[i]) / 2

                Qc_ice[i] = self.k * (Td[N-2,i] - Td[N-1,i]) / h
                if Qc_ice[i] < 0:
                    Qc_ice[i] = 0
                # Melt [m ice]
                Melt[i] = Qc_ice[i] * self.dt / (900 * 334000) # m of melt in each time step
        else: # Clean ice energy balance model 
            # Surface energy fluxes
            ice_albedo = 0.4
            a_neutral_ice = self.a_bulk
            for i in np.arange(self.nsteps):
                (F_Ts[i], Rn[i], LE[i], H_flux[i], P_flux[i], Qc[i], dsnow[i], tsnow[i], snow_tau[i]) = (
                        calc_surface_fluxes_cleanice(self.T_air[i], self.RH_air[i], self.u[i], self.SW_sky[i], self.LW_sky[i],
                                            0, 0, self.P[i], ice_albedo, a_neutral_ice, 0, 0, 0,
                                            0, 0))
                Melt[i] = F_Ts[i] * self.dt / (900 * 334000) # m of melt in each time step
                
        Melt[Melt<0] = 0 # No negative melt allowed
        Cum_Melt = Melt.cumsum() # cumulative m of melt
        Melt_Rate = Melt * 100 * 24 * 3600 / self.dt # Melt expressed as cm/day
        mean_melt_rate = np.mean(Melt_Rate)

        if stake_file is not None:
            stake = pd.read_csv(stake_file,parse_dates=True,index_col=0)
            stake['Cum_Melt'] = np.cumsum(stake['Melt'])
            time_zero = stake.index[0]
        
        if time_zero is not None:
            i = self.timestamp >= time_zero
            time_zero = self.timestamp[i][0] # set time zero to conform with data timestamps
            tzi = np.argwhere(self.timestamp >= time_zero)
            print('Time zero = {}'.format(time_zero))
            Cum_Melt = Cum_Melt - Cum_Melt[tzi[0]] # Cumulative Melt starts at time zero
            # Set plotting period to start at time zero:
            if plot_period is None:
                plot_period[0] = time_zero
                plot_period = [time_zero, self.timestamp[-1]]

        if save_dir is not None:
            dtstr = str(debris_thickness*100)
            outpath_eb = save_dir + '/' + dtstr + 'cm_eb_melt.csv'
            outpath_dt = save_dir + '/' + dtstr + 'cm_debris_temp.csv'

            # Construct pandas dataframe for energy balance output:
            eb = pd.DataFrame()
            eb['Rad'] = Rn
            eb['H'] = H_flux
            eb['LE'] = LE
            eb['G'] = Qc
            eb['Melt'] = Melt
            eb['Cum_Melt'] = Cum_Melt
            eb['Melt_Rate'] = Melt_Rate
            eb.index = self.timestamp
            eb.to_csv(outpath_eb)
            self.eb = eb

            # Construct pandas dataframe for debris temperature:
            Td_out = pd.DataFrame()
            depth_str = []
            for n in range(N):
                depth = h * n
                Td_out[str(depth)] = Td[n,:]
            Td_out.index = self.timestamp
            Td_out.to_csv(outpath_dt)

        if plot:
            dt_format = mdates.DateFormatter('%m-%d')
            plt.figure()
            ax1=plt.subplot(311)
            plt.plot(self.timestamp, Rn)
            plt.plot(self.timestamp, LE)
            plt.plot(self.timestamp, H_flux)
            plt.plot(self.timestamp, Qc)
            plt.legend(['Rad','LE', 'H', 'G'], loc = 'upper right')
            plt.ylabel(r'$W/m^2$')
            ax1.xaxis.set_major_formatter(dt_format)
            plt.grid(True)

            if plot_period is not None:
                plt.xlim(plot_period)

            ax2=plt.subplot(312)
            plt.plot(self.timestamp, Melt * 100 * 24 * 3600 / self.dt) # Melt expressed as cm/day
            plt.ylabel('Melt Rate,\n cm/day')
            ax2.xaxis.set_major_formatter(dt_format)

            if plot_period is not None:
                plt.xlim(plot_period)

            if debris_thickness > 0:
                ax3=plt.subplot(313)
                plt.plot(self.timestamp, Td[0,:]-273.15)
                plt.ylabel('Debris Surf\n Temp, '+r'$^{\circ}$'+'C')
                ax3.xaxis.set_major_formatter(dt_format)

                if plot_period is not None:
                    plt.xlim(plot_period)
                    i = np.where( (self.timestamp > plot_period[0]) & (self.timestamp < plot_period[1]))
                    if stake_file is not None:
                        plt.ylim( Cum_Melt[i].min()*100-Cum_Melt[i].max()*3, max(Cum_Melt[i].max()*105, max(stake['Cum_Melt'])*1.05))

            #plt.show()

        return mean_melt_rate
    

def main():

    # Change directory to location of this python :
    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    ebm_test = energy_balance_model('./Model_Setup/kennicott_example.json')

    clean_ice_melt_rate = ebm_test.calc_energy_flux_melt(0.00, plot_period=['2021-07-01','2021-07-15']) # Debris Thickness = 0
    print("Avg. clean ice melt rate = {:.1f} cm/day".format(clean_ice_melt_rate))

    clean_ice_melt_rate = ebm_test.calc_energy_flux_melt(.10, plot_period=['2021-07-01','2021-07-15']) # Debris Thickness = 10 cm
    print("Avg. melt rate, under 10 cm debris = {:.1f} cm/day".format(clean_ice_melt_rate))

    plt.show()

if __name__ == "__main__":
    main()