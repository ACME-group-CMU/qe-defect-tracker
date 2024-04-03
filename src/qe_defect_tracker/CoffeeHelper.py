#Import Packages
import os
import subprocess
import importlib
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mp_api.client import MPRester as MPRester_new

from qe_defect_tracker import Utility
importlib.reload(Utility)

class CoffeeHelper(object):

    def __init__(self,defect_object,cell_matrix,supercell_dim,mpid,ecut,debug_obj=None,correction_details = False):

        self.debug_obj = debug_obj
        self.util_obj = Utility.Utility(self.debug_obj)

        self.mpid = mpid
        self.defect_object = defect_object

        #cell matrix only used to create normalized cell vectors
        self.cell_matrix = cell_matrix

        #make a list of three supercells of increasing size (+1 in each dimension)
        supercell_dim_1 = np.array(supercell_dim)
        supercell_dim_2 = copy.deepcopy(supercell_dim_1) + 1
        supercell_dim_3 = copy.deepcopy(supercell_dim_1) + 2

        self.list_of_supercell_dims = [supercell_dim_1,supercell_dim_2,supercell_dim_3]
        self.list_of_supercell_energies = [0,0,0]

        self.super_structure = self.defect_object.generate_defect_structure()

        self.ecut = ecut
        self.sigma = 0
        self.sigma_by_charge = False
        self.correction_details = correction_details

        try:
            with MPRester_new(self.util_obj.checkEnvironmentalVars("MPI_API_KEY")) as mp_new:
                dielectric_info = mp_new.dielectric
            self.epsilon = dielectric_info.get_data_by_id(self.mpid).e_total
        except:
            self.debug_obj.debug_log(f"Epsilon value cannot be fetched from materials project.")

        if (self.correction_details != False):
            try:
                self.sigma_by_charge = self.correction_details['sigma_by_charge']
            except:
                self.debug_obj.debug_log("No manual sigma provided")
                self.debug_obj.debug_log(f"Correction details: {self.correction_details}")

            try:
                manual_epsilon = self.correction_details['epsilon']
                self.epsilon = manual_epsilon
                self.debug_obj.debug_log(f"Epsilon manually set to {self.epsilon}.")
            except:
                self.debug_obj.debug_log("No manual epsilon provided")

        self.input_name = "coffee_#.in"
        self.output_name = "coffee_#.out"

        self.defect_loc = defect_object.site.as_dict()["abc"]
        self.debug_obj.debug_log(f"Defect Loc: {self.defect_loc}")

        self.supercell_dimensions = np.array(self.super_structure.lattice.abc)
        self.debug_obj.debug_log(f"supercell_dimensions: {self.supercell_dimensions}")

        self.unit_cell_dimensions = self.supercell_dimensions/supercell_dim_1

        self.energyCorrection = 0
        self.E_pot_corr = 0
        self.E_lattice = 0

        self.COFFEE_DIR = self.util_obj.checkEnvironmentalVars("COFFEE_DIR")
        self.debug_obj.debug_log(f"self.COFFEE_DIR: {self.COFFEE_DIR}")


        self.coffee_g_fit = f"{self.COFFEE_DIR}/GaussianFit/g_fit.py"
        self.coffee_lattice_energy = f"{self.COFFEE_DIR}/coffee.py"
        self.coffee_dV_0p = f"{self.COFFEE_DIR}/PotentialAlignment/dV_0p.py"
        self.coffee_dV_mD = f"{self.COFFEE_DIR}/PotentialAlignment/dV_mD.py"
        self.coffee_plavg = f"{self.COFFEE_DIR}/PotentialAlignment/Utilities/plavg.py"
        #plot_fit.py is simple and has been pulled into a function in this class


    def _calculateSigma(self,charge_path,charge):

        sigmaFound = False

        #check if sigma provided manually
        if self.sigma_by_charge != False:
            try:
                self.sigma = self.sigma_by_charge[charge]
                self.debug_obj.debug_log(f"Manual Sigma: {self.sigma}")
                sigmaFound = True
                return sigmaFound
            except:
                self.debug_obj.debug_log(f"No manual sigma provided for charge {charge}")

        gaussian_fit_input = "coffee_gfit.in"
        gaussian_fit_output = "coffee_gfit.out"

        gaussian_fit_input = charge_path + "/" + gaussian_fit_input

        # find the relevant cube file
        cube_filename = "espresso_plot_num_7.cube"
        cube_path = charge_path + f"/{cube_filename}"

        # create a gaussian fit input file
        data = f"file_name = {cube_path}\nfile_type = cube\nplt_dir = a3"

        with open(gaussian_fit_input, 'w') as f:
            f.write(data)
            f.close()

        self.debug_obj.debug_log(f"self.coffee_g_fit: {self.coffee_g_fit}")
        self.debug_obj.debug_log(f"gaussian_fit_input: {gaussian_fit_input}")
        #Run gaussian fit and save the output
        args = ["python", self.coffee_g_fit,gaussian_fit_input,">",gaussian_fit_output]
        result = subprocess.run(args, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = result.stdout.decode('utf-8')

        with open(gaussian_fit_output,"w") as f:
            f.write(output)
            f.close()

        for line in output.split("\n"):
            if 'Sigma fit' in line:
                self.sigma = float(line.split()[-2])
                self.debug_obj.debug_log(f"Sigma: {self.sigma }")
                sigmaFound = True

        return sigmaFound

    # Purpose | Manage the process of calculating the FNV correction
    def calculateFNV(self,charge,charge_path,no_charge_path,pristine_path):
        #generates file, but doesn't make call to coffee
        energyFound = False

        sigmaFound = self._calculateSigma(charge_path,charge)
        self.debug_obj.debug_log(f"Accepted sigma: {self.sigma} bohr")
        energyFoundCount = 0

        lattice_correction_found = False
        potential_correction_found = False
        all_corrections_found = False

        if sigmaFound:

            for iter, supercell_dim in enumerate(self.list_of_supercell_dims):

                #make a unique coffee input for each supercell
                supercell_str = f"{str(supercell_dim[0])}x{str(supercell_dim[1])}x{str(supercell_dim[2])}"
                input_name_with_supercell = self.input_name.replace("#",supercell_str)
                specific_input_name = charge_path + "/" + input_name_with_supercell

                self._generateCoffeeInput(charge,supercell_dim,specific_input_name)

                #Run the lattice correction program coffee.py
                args = ["python", self.coffee_lattice_energy,specific_input_name,">",self.output_name]
                result = subprocess.run(args, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                output = result.stdout.decode('utf-8')

                output_name_with_supercell = self.output_name.replace("#",supercell_str)
                with open(output_name_with_supercell,"w") as f:
                    f.write(output)
                    f.close()

                for line in output.split("\n"):
                    if '!  Total Energy (eV)' in line:
                        uncorrected_energy = float(line.split(" ")[-1])
                        self.debug_obj.debug_log(f"Coffee.py energy: {uncorrected_energy} eV")
                        self.list_of_supercell_energies[iter] = uncorrected_energy
                        energyFoundCount += 1
                        break

            if energyFoundCount == len(self.list_of_supercell_dims):
                self._calculateLatticeEnergy() #this is difficult to validate
                lattice_correction_found = True

        #Calculate the potential corrections if the lattice correction was successful
        if lattice_correction_found:

            self.debug_obj.debug_log("Calculating potential alignment correction")

            #Run dV_0p.py
            dV0p_filename = "coffee_dV0p.in"
            self._generate_in_dV_0p(charge,no_charge_path,pristine_path,dV0p_filename)
            args = ["python", self.coffee_dV_0p,dV0p_filename]
            result = subprocess.run(args, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            output = result.stdout.decode('utf-8')
            self.debug_obj.debug_log(f"dV_0p.py output: {output}")

            #Plot dv_0p
            dV0p = pd.read_table('coffee_dV0p_voltage_diff_a1.plot', sep=' ')
            dV0p.columns = ['dist','volt']

            defect_coord = self.supercell_dimensions[0]* self.defect_loc[0]
            #defect_coord = 7.762
            self.debug_obj.debug_log(f"Defect Coords: {defect_coord}")
            farthest_point_from_defect = self.determine_farthest_point_in_cell(defect_coord,dV0p['dist'])
            self.E_vop = np.interp(farthest_point_from_defect,dV0p['dist'],dV0p['volt'])
            self.debug_obj.debug_log(f"dV0p @ far from defect = {self.E_vop}")
            self.debug_obj.debug_log(f"dV0p @ far from defect = {self.E_vop}")
            plt.clf()
            plt.plot(dV0p['dist'],dV0p['volt'],color='k',label=f"$-q(V_0-V_p)$")
            plt.axvline(x=defect_coord,color='red',linestyle='--',label='defect coordinate')
            plt.axvline(x=farthest_point_from_defect,color='blue',linestyle='--',label='coordinate farthest from defect')
            plt.axhline(y=self.E_vop,color='blue',linestyle='--')
            #plt.text(x=farthest_point_from_defect,y=0,s=f"Voltage diff: {voltage_value:.3f}")
            plt.title(f"$-q(V_0-V_p)$ Difference Correction = {self.E_vop:.3f}")
            plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
            plt.gca()
            plt.savefig('coffee_dv0p_correction.png',bbox_inches='tight',pad_inches=1)
            plt.clf()

            #Run dV_mD.py
            dVmD_filename = "coffee_dVmD.in"
            self._generate_in_dV_mD(charge_path,no_charge_path,dVmD_filename)
            args = ["python", self.coffee_dV_mD,dVmD_filename]
            result = subprocess.run(args, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            output = result.stdout.decode('utf-8')
            self.debug_obj.debug_log(f"dV_mD.py output: {output}")

            #Plot dv_md
            dVmD_model = pd.read_table('coffee_dVmD_model_a1.plot', sep=' ')
            dVmD_model.columns = ['dist','volt']
            dVmD_model['volt'] = dVmD_model['volt'] * -1
            dVmD_vdiff = pd.read_table('coffee_dVmD_voltage_diff_a1.plot', sep=' ')
            dVmD_vdiff.columns = ['dist','volt']

            #dVmD_vdiff['volt'] = dVmD_vdiff['volt']*self.get_sign_of_charge(charge)
            voltage_model = np.interp(farthest_point_from_defect,dVmD_model['dist'],dVmD_model['volt'])
            self.debug_obj.debug_log(f"dVmD_model @ far from defect = {voltage_model}")
            voltage_vdiff = np.interp(farthest_point_from_defect,dVmD_vdiff['dist'],dVmD_vdiff['volt'])
            self.debug_obj.debug_log(f"dVmD_vdiff @ far from defect = {voltage_vdiff}")
            self.debug_obj.debug_log(f"dVmD_vdiff @ far from defect = {voltage_vdiff}")
            self.E_vq0m = voltage_vdiff - voltage_model
            plt.plot(dVmD_vdiff['dist'],dVmD_vdiff['volt'],color='k',label=f"$(V_0-V_p)$")
            plt.plot(dVmD_model['dist'],dVmD_model['volt'],color='orange',label="$V_{model}$")
            plt.axvline(x=defect_coord,color='red',linestyle='--',label='defect coordinate')
            plt.axvline(x=farthest_point_from_defect,color='blue',linestyle='--',label='coordinate farthest from defect')
            plt.axhline(y=voltage_model,color='orange',linestyle='--',label='model voltage farthest from defect')
            plt.axhline(y=voltage_vdiff,color='k',linestyle='--',label=f"$(V_0-V_p)$ farthest from defect")
            plt.plot([farthest_point_from_defect,farthest_point_from_defect],
                     [voltage_model,voltage_vdiff],marker = 's',color='green',label="correction bounds")
            plt.title(f"$[(V_0-V_p)-V_m]$ Correction = {self.E_vq0m:.3f}")
            plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
            #plt.gca()
            plt.savefig('coffee_dvmD_correction.png',bbox_inches='tight',pad_inches=1)
            plt.clf()

            total_FNV_correction = (self.E_lattice + self.E_vq0m + self.E_vop)
            self.debug_obj.debug_log(f"#################################################")
            self.debug_obj.debug_log(f"Charge: {charge}")
            self.debug_obj.debug_log(f"total_FNV_correction: {total_FNV_correction}")
            self.debug_obj.debug_log(f"#################################################")

        #Now calculate the potential correction energy
        potential_correction_found = True

        if lattice_correction_found and potential_correction_found:
            all_corrections_found = True

        return all_corrections_found

    def getFNVCorrection(self):
        list_of_corrections = {
            "E_lattice": self.E_lattice,
            "E_vq0m": self.E_vq0m,
            "E_vop": self.E_vop,
            "total": (self.E_lattice + self.E_vq0m + self.E_vop)
        }
        return list_of_corrections

    #This code is an implementation of plot_fit.py as created by the developers of CoFFEE
    # Naik, M. H. & Jain, M. CoFFEE: Corrections For Formation Energy and ...
    # Eigenvalues for charged defect simulations. Computer Physics Communications 226, 114–126 (2018).
    def _calculateLatticeEnergy(self):

        self.debug_obj.debug_log("Attmepting to calculate lattice energy correction ")
        # Computes the fitting polynomial.
        # C_u takes the 3 model energies
        # L2, L3 and L4 are 1/\Omega^{1/3} for the correspoding cells.
        def compute_fit(C_u,L2,L3,L4):
            self.debug_obj.debug_log(C_u)
            self.debug_obj.debug_log(f"L2: {L2}")
            self.debug_obj.debug_log(f"L3: {L3}")
            self.debug_obj.debug_log(f"L4: {L4}")
            A = np.array([[ L2, L2**3, 1.], [  L3, L3**3, 1.] , [  L4, L4**3, 1. ] ],dtype=float)
            A_inv =  np.linalg.inv(A)
            X_u = np.dot(A_inv,C_u)
            return X_u

        alat = self.unit_cell_dimensions[0]

        # 1/\Omega^{1/3} for 4x4x4, 5x5x5 and 6x6x6 cells.
        supercell_volumes = [np.prod(self.list_of_supercell_dims[0]),
                             np.prod(self.list_of_supercell_dims[1]),
                             np.prod(self.list_of_supercell_dims[2])]

        one_by_V = np.array([1,1,1])/supercell_volumes
        one_by_A = (one_by_V)**(1/3.)*(1/alat)

        # Compute the fit: p(\Omega) = f_1 + f_2/(\Omega^(1/3)) + f_3/(\Omega)
        X = compute_fit(self.list_of_supercell_energies,one_by_A[0],one_by_A[1],one_by_A[2])

        self.E_m_iso = X[2]
        self.debug_obj.debug_log(f"E_m_iso = {self.E_m_iso}")
        self.E_lattice = self.E_m_iso - self.list_of_supercell_energies[0]
        self.debug_obj.debug_log(f"E_lattice = {self.E_lattice}")

        return self.E_lattice

    def _generatePlanarAverageInput(self,cube_filename,planar_avg_filename):
        #Format is
        """
        &plavg
        file_name = ../MoS2.cube
        file_type = cube
        plt_dir = a1
        factor = 13.605698
        /
        """
        input = f"&plavg\nfile_name = {cube_filename}\nfile_type = cube\nplt_dir = a1\nfactor = 13.605698\n/"

        with open(planar_avg_filename, 'w') as f:
            f.write(input)
            f.close()

    # Purpose | Generate the coffee input file. Function made with GPT!
    def _generateCoffeeInput(self,charge,supercell_dim,specific_input_name):

        angstroms_to_bohr = 1.8897259886

        celldm1 = self.cell_matrix[0][0]
        normed_matrix = copy.deepcopy(self.cell_matrix)/celldm1
        # Cell Parameters
        lattice_vectors = f"{normed_matrix[0][0]}   {normed_matrix[0][1]}   {normed_matrix[0][2]}\n" \
                          f"{normed_matrix[1][0]}   {normed_matrix[1][1]}   {normed_matrix[1][2]}\n" \
                          f"{normed_matrix[2][0]}   {normed_matrix[2][1]}   {normed_matrix[2][2]}"

        cell_dimensions = str(supercell_dim[0]*self.unit_cell_dimensions[0]*angstroms_to_bohr) + " " + \
                          str(supercell_dim[1]*self.unit_cell_dimensions[1]*angstroms_to_bohr) + " " + \
                          str(supercell_dim[2]*self.unit_cell_dimensions[2]*angstroms_to_bohr)
        #cell_dimensions = "1.0 1.0 1.0"

        cell_parameters = f"&CELL_PARAMETERS\n\nLattice_Vectors(normalized):\n{lattice_vectors}\n\nCell_dimensions bohr\n{cell_dimensions}\n\nEcut={self.ecut} Hartree\n/\n"

        # Dielectric Parameters
        epsilon1_a1 = self.epsilon
        epsilon1_a2 = self.epsilon
        epsilon1_a3 = self.epsilon

        dielectric_parameters = f"&DIELECTRIC_PARAMETERS Bulk\nEpsilon1_a1 = {epsilon1_a1}\nEpsilon1_a2 = {epsilon1_a2}\nEpsilon1_a3 = {epsilon1_a3}\n/\n"

        # Gaussian Parameters
        centre_a1 = self.defect_loc[0]
        centre_a2 = self.defect_loc[1]
        centre_a3 = self.defect_loc[2]

        gaussian_parameters = f"&GAUSSIAN_PARAMETERS:\nTotal_charge = {charge}\nSigma = {self.sigma}\nCentre_a1 = {centre_a1}\nCentre_a2 = {centre_a2}\nCentre_a3 = {centre_a3}\n/\n"

        document = cell_parameters + dielectric_parameters + gaussian_parameters

        with open(specific_input_name, 'w') as f:
            f.write(document)

    def _generate_in_dV_0p(self,charge,no_charge_path,pristine_path,filename):
        data = f"&dV_0p\n"
        data += f"file_type = xsf\n"
        data += f"file_pristine = {pristine_path}/espresso_plot_num_1.xsf\n"
        data += f"file_neutral = {no_charge_path}/espresso_plot_num_1.xsf\n"
        data += f"charge = {charge}\n"
        data += f"plt_dir = a1\n"
        data += f"factor = Ryd\n"
        data += f"/\n"

        with open(filename, 'w') as f:
            f.write(data)
            f.close()

    def _generate_in_dV_mD(self,charge_path,no_charge_path,filename):
        data = f"&dV_mD\n"
        data += f"file_type = xsf\n"
        data += f"file_model = {charge_path}/V_r.npy\n"
        data += f"file_charged = {charge_path}/espresso_plot_num_1.xsf\n"
        data += f"file_neutral = {no_charge_path}/espresso_plot_num_1.xsf\n"
        data += f"plt_dir = a1\n"
        data += f"factor = Ryd\n"
        data += f"/\n"

        with open(filename, 'w') as f:
            f.write(data)
            f.close()

    def determine_farthest_point_in_cell(self,defect_coordinate,cell_distances):
        cell_length = cell_distances.max()
        max_seperation = cell_length/2

        ideal_1 = defect_coordinate - max_seperation
        if(ideal_1 > cell_distances.min() and ideal_1 < cell_length):
            self.debug_obj.debug_log(f"Farthest coordinate from defect: {ideal_1}")
            return ideal_1
        else:
            ideal_2 = defect_coordinate + max_seperation
            self.debug_obj.debug_log(f"Farthest coordinate from defect: {ideal_2}")
            return ideal_2

    def get_sign_of_charge(self,charge):
        if charge != 0:
            return abs(charge)/charge
        else:
            return 0




#%%
