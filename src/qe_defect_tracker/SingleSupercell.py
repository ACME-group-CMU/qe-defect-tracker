#Import Packages

#Materials Project
from mp_api.client import MPRester as MPRester_new

#ASE
from ase import Atoms
from ase.calculators.espresso import Espresso
import ase.io.espresso as ase_qe_support
from ase.io import read as ase_read

#Pymatgen
from pymatgen.analysis.defects.core import Defect
from pymatgen.io.ase import AseAtomsAdaptor

#Other
import copy
import os
import subprocess
import importlib
import math
import numpy as np
import re

#Defect tracking files

#Other Defect Trackin Items
from qe_defect_tracker import CoffeeHelper
importlib.reload(CoffeeHelper)

from qe_defect_tracker import Utility
importlib.reload(Utility)


#Uses the parameters for the pristine supercell
#If the Pristine params are changed all the defect energies are cleared out (check if this is necessary)
class SingleSupercell(object):

    # Purpose | Create an instance for every unique type of supercell object. Every defect has a unique SS
    # arg | type | "PRISTINE","VAC","SUB","INT"
    # arg | type_num | int | unique identifier, e.g. 0,1,2
    # arg | supercell | pymatgen supercell object
    # arg | defect_object | pymatgen defect object | pristine cell won't have this
    # arg | mpid | string | unique materials project identifier
    def __init__(self,type,type_num,unitcell=None,supercell=None,supercell_dim = None,defect_object = None,mpid = None,debug_obj=None,delete_wfc_files=False):

        self.debug_obj = debug_obj
        self.util_obj = Utility.Utility(self.debug_obj)

        self.type = type #"PRISTINE","VAC","SUB","INT"
        self.type_num = type_num #unique identifier, e.g. 0,1,2

        self.supercell = supercell #pymatgen object
        self.unitcell = unitcell
        self.supercell_dim = supercell_dim
        if self.supercell == None and self.unitcell == None:
            self.debug_obj.debug_log("Cell not defined. Either supercell or unitcell must be defined.",type='Error')

        self.defect_object = defect_object
        self.raw_energy_dict = {}
        self.pymatgen_defect_objects = []
        self.calc_successful = False
        self.chargeList = [0]
        self.mpid = mpid
        self.correction_type = "FNV"
        self.fs_correction_details = False

        self.error = False

        self.ESPRESSO_PP_EXE = self.util_obj.checkEnvironmentalVars("ESPRESSO_PP_EXE")

        self.delete_wfc_files = delete_wfc_files

        #naming convetions
        self.generic_cube_filename = "espresso_plot_num_"

    # Purpose | Determine what defect charges to check
    def selectDefectCharges(self,chargeList):

        #A method to rearrange charge list to run 0 first
        def reorder_list(items, early):
            moved = [item for item in early if item in items]
            remain = [item for item in items if item not in moved]
            return moved + remain

        if self.type == "PRISTINE":
            self.chargeList == [0]
        else:
            #hacky hardcode. Ultimately will implement a function to determine
            # appropriate charges for each defect type
            if chargeList == False:
                self.chargeList = [-2,-1,0,1,2]
            else:
                #This will re-order w.r.t. absolute value.
                move_early = [0,1,-1,2,-2,3,-3,4,-4,5,-5,6,-6]
                self.chargeList = reorder_list(chargeList,move_early)

    # Purpose | Return the list of raw energies for each defect w.r.t. charge
    def getRawEnergies(self):
        return self.raw_energy_dict

    # Purpose | Clear the raw energy dictionary. Will be called when QE params are updated
    def clearDefectEnergies(self):
        self.raw_energy_dict = {}

    # Purpose | Run QE for each selected charge of the defects
    # arg | qe_parameters | dictionary object of QE arguments
    # arg | pseudopotentials | dictionary of pseudopotentials
    # arg | kpts | list of kpoints | e.g. [2,2,2]
    # return True is successful, false if not
    def runDFTCalculation(self,qe_parameters,pseudopotentials,kpts=None,min_kpt_density=None,
                          correction_params = None,force_recalc = False):

        self.pseudopotentials = pseudopotentials
        self.qe_parameters = qe_parameters

        if kpts is None and min_kpt_density is None:
            self.debug_obj.debug_log(f"Kpt info is not defined!",type="ERROR")

        if kpts is not None and min_kpt_density is not None:
            self.debug_obj.debug_log(f"Kpt info is over-defined. Directly select kpts or set the kpt density.",type="ERROR")

        try:
            self.debug_obj.debug_log(f"Unitcell Matrix: \n{self.unitcell.lattice.matrix}")
        except:
            print("Unitcell not currently defined")

        if min_kpt_density is not None:
            unitcell_rec_lengths = np.array(self.unitcell.lattice.reciprocal_lattice.abc)
            self.debug_obj.debug_log(f"unitcell_rec_lengths: {unitcell_rec_lengths}")
            kpt_unitcell_exact = min_kpt_density * unitcell_rec_lengths
            self.debug_obj.debug_log(f"kpt_unitcell_exact: {kpt_unitcell_exact}")
            self.unitcell_kpts = [math.ceil(x) for x in kpt_unitcell_exact]
            self.debug_obj.debug_log(f"unitcell kpts: {self.unitcell_kpts}")
            #CANNOT yet set the supercell kpts

        if kpts is not None:
            try:
                if type(kpts[0]) is list and type(kpts[1]) is list:
                    self.unitcell_kpts = kpts[0]
                    self.supercell_kpts = kpts[1]
                else:
                    self.unitcell_kpts = kpts
                    self.supercell_kpts = kpts
            except:
                self.unitcell_kpts = kpts
                self.supercell_kpts = kpts

        if correction_params is not None:
            self.fs_correction_type = correction_params['type']
            self.fs_correction_details = correction_params[self.correction_type]
            self.debug_obj.debug_log(f"correction_params: {correction_params}")

        global_path = os.getcwd()

        self.supercell_path = global_path + "/supercells"
        self.util_obj.createNewDirectory(self.supercell_path)

        self.defect_path = self.supercell_path + "/" + self.type + "_" + str(self.type_num)
        self.util_obj.createNewDirectory(self.defect_path)

        calc_successful_count = 0
        successCriteria = len(self.chargeList)

        for charge in self.chargeList:

            calc_successful_update = 0
            prefix = self.qe_parameters['control']['prefix']

            self.debug_obj.debug_log(f"Running DFT for defect: {self.type}, num: {self.type_num}, charge: {charge}")

            #Set a naming convention
            charge_path = self.defect_path + '/' + str(charge)
            self.util_obj.createNewDirectory(charge_path)
            self.util_obj.changeDirectory(charge_path)

            valid_file_found__unitcell_output = False
            valid_file_found__supercell_output = False
            valid_file_found__wfcs = False
            valid_file_found__pp_requiring_wfcs = False
            valid_file_found__pp_not_requiring_wfcs = False

            #if pristine, search for an exitsing espresso.pwo file for unit cell
            if self.type == "PRISTINE":
                self.util_obj.changeDirectory("./unitcell")
                unitcell_filename = 'espresso.pwo'
                search_result = self.search_qe_output(unitcell_filename)

                if search_result['job_done'] == True:
                    #Now grab the relaxed cell info of the unit cell
                    new_atoms_loc = ase_read(unitcell_filename)
                    pymat_pristine_object = AseAtomsAdaptor.get_structure(atoms = new_atoms_loc)
                    pymat_pristine_object.make_supercell(self.supercell_dim)
                    self.supercell = pymat_pristine_object
                    valid_file_found__unitcell_output = True
                else:
                    self.debug_obj.debug_log(f"A valid espresso.pwo for the unitcell has not been found.")

            # search for an existing supercell espresso.pwo output file
            self.util_obj.changeDirectory(charge_path)
            output_filename = "espresso.pwo"
            search_result = self.search_qe_output(output_filename)

            if search_result['job_done'] == True:
                valid_file_found__supercell_output = True
                self.raw_energy_dict[charge] = search_result['total_energy']
                calc_successful_update = 1
            else:
                self.debug_obj.debug_log(f"A valid espresso.pwo for the supercell has not been found.")

            #Now check if any cube files (which might need wfc files) are already present
            if(self.correction_type != "None"):

                if(self.correction_type == "FNV"):

                    cube_info = 1
                    cube_filename = f"{self.generic_cube_filename}{cube_info}.xsf"
                    valid_file_found__pp_not_requiring_wfcs = self.util_obj.checkIfFileExists(cube_filename) #returns True/False

                    #set to true for pristine and q=0 cases
                    valid_file_found__pp_requiring_wfcs = True

                    if charge != 0:

                        #Look for the plot_num = 7 cube
                        cube_info = 7
                        cube_filename = f"{self.generic_cube_filename}{cube_info}.cube"
                        valid_file_found__pp_requiring_wfcs = self.util_obj.checkIfFileExists(cube_filename) #returns True/False


            #Now Check if wfc files exist
            try:
                self.util_obj.changeDirectory(charge_path)
                self.util_obj.changeDirectory(f"./{prefix}.save")
                if self.util_obj.checkIfFileExists("wfc") == True:
                    valid_file_found__wfcs = True
                self.util_obj.changeDirectory(charge_path)
            except:
                self.debug_obj.debug_log(f"An error likely occurred from changing directories.")

            run_pw_unitcell = False
            run_pw_supercell = False
            run_pp_that_requires_wfc = False
            run_pp_that_doesnt_require_wfc = False

            #Item of interest #1 - pp not from wfcs
            if valid_file_found__pp_not_requiring_wfcs == False:
                run_pp_that_doesnt_require_wfc = True

            #Item of interest #2 - pp from wfcs
            if valid_file_found__pp_requiring_wfcs == False:
                run_pp_that_requires_wfc = True
                if valid_file_found__wfcs == False:
                    run_pw_supercell = True

            #Item of interest #3 - pw supercell output
            if valid_file_found__supercell_output == False:
                run_pw_supercell = True

            if run_pw_supercell == True:
                if self.type == "PRISTINE" and valid_file_found__unitcell_output == False:
                    run_pw_unitcell = True

            if force_recalc == True:
                if self.type == "PRISTINE":
                    run_pw_unitcell = True
                run_pw_supercell = True
                run_pp_that_requires_wfc = True
                run_pp_that_doesnt_require_wfc = True

            self.debug_obj.debug_log(f"valid_file_found__supercell_output: {valid_file_found__supercell_output}")
            self.debug_obj.debug_log(f"valid_file_found__wfcs: {valid_file_found__wfcs}")
            self.debug_obj.debug_log(f"valid_file_found__pp_requiring_wfcs: {valid_file_found__pp_requiring_wfcs}")
            self.debug_obj.debug_log(f"valid_file_found__pp_not_requiring_wfcs: {valid_file_found__pp_not_requiring_wfcs}")

            self.debug_obj.debug_log(f"run_pw_unitcell: {run_pw_unitcell}")
            self.debug_obj.debug_log(f"run_pw_supercell: {run_pw_supercell}")
            self.debug_obj.debug_log(f"run_pp_that_requires_wfc: {run_pp_that_requires_wfc}")
            self.debug_obj.debug_log(f"run_pp_that_doesnt_require_wfc: {run_pp_that_doesnt_require_wfc}")


            if run_pw_unitcell == True:

                #make a subfolder so as not to mess up supercell calcs
                pristine_unitcell_path = charge_path + '/unitcell'
                self.util_obj.createNewDirectory(pristine_unitcell_path)
                self.util_obj.changeDirectory(pristine_unitcell_path)

                #copy parameters so as not make any undesired change to passed-in params
                qe_parameters_temp = copy.deepcopy(self.qe_parameters)

                #create an ase unit cell
                #system = copy.deepcopy(qe_parameters_temp['system'])
                self.debug_obj.debug_log(f"Unitcell Matrix: \n{self.unitcell.lattice.matrix}")

                #Be careful about the molecule and num. The num is for EACH species listed
                ASE_atom_obj = AseAtomsAdaptor.get_atoms(self.unitcell)

                #Generate cube file of the atom (can remove after debug finished)
                from ase.io.cube import write_cube
                cube_name = "pre_dft_unitcell.cube"
                with open(cube_name,'w') as file:
                    write_cube(file, atoms=ASE_atom_obj)

                #vc-relax the single unit cell
                qe_parameters_temp['control']['calculation'] = "vc-relax"

                #create QE calc with input data
                qe_calc = Espresso(pseudopotentials=self.pseudopotentials,
                                    input_data = qe_parameters_temp,
                                    kpts=(self.unitcell_kpts[0],self.unitcell_kpts[1],self.unitcell_kpts[2]),
                                    tstress=True,
                                    tprnfor=True)

                ASE_atom_obj.calc = qe_calc
                ASE_atom_obj.get_total_energy()

                #Now grab the newly relaxed cell info
                output_file = "espresso.pwo"
                new_atoms_loc = ase_read(output_file)
                pymat_pristine_object = AseAtomsAdaptor.get_structure(atoms = new_atoms_loc)
                pymat_pristine_object.make_supercell(self.supercell_dim)
                self.supercell = pymat_pristine_object

            if run_pw_supercell == True:

                self.debug_obj.debug_log(f"Starting DFT calculation")

                self.util_obj.changeDirectory(charge_path)

                #Set the kpts here if defined with a density
                if min_kpt_density is not None:
                    supercell_rec_lengths = np.array(self.supercell.lattice.reciprocal_lattice.abc)
                    self.debug_obj.debug_log(f"supercell_rec_lengths: {supercell_rec_lengths}")
                    kpt_supercell_exact = min_kpt_density * supercell_rec_lengths
                    self.debug_obj.debug_log(f"kpt_supercell_exact: {kpt_supercell_exact}")
                    self.supercell_kpts = [math.ceil(x) for x in kpt_supercell_exact]
                    self.debug_obj.debug_log(f"Supercell kpts: {self.supercell_kpts}")

                #copy the params of the generic defect and update for the charge
                qe_parameters_temp = copy.deepcopy(self.qe_parameters)
                qe_parameters_temp['system']['tot_charge'] = charge

                self.debug_obj.debug_log(f"Supercell Matrix: \n{self.supercell.lattice.matrix}")

                ASE_atom_obj = AseAtomsAdaptor.get_atoms(self.supercell)

                #Generate cube file of the atom (can remove after debug finished)
                from ase.io.cube import write_cube
                cube_name = "pre_dft_supercell.cube"
                with open(cube_name,'w') as file:
                    write_cube(file, atoms=ASE_atom_obj)

                # Add the parameters which must be passed via ASE rather than directly through the QE input
                #Designed to directly pass through info without anything fancy
                try:
                    self.debug_obj.debug_log(f"Attempting to set extra ASE parameters")
                    ase_params = qe_parameters_temp['ase_parameters']

                    #Magnetic moments
                    try:
                        magmoms = ase_params["magmoms"]
                        if magmoms is not None:
                            self.debug_obj.debug_log(f"Attempting to set magnetic moments")
                            ASE_atom_obj.set_initial_magnetic_moments(magmoms)
                    except:
                        self.debug_obj.debug_log(f"Something is wrong with the provided magmom list.")

                except:
                    self.debug_obj.debug_log(f"No ASE parameters provided.")

                self.debug_obj.debug_log("Created the atom object!")

                #create QE calc with input data
                qe_calc = Espresso(pseudopotentials=self.pseudopotentials,
                                   input_data = qe_parameters_temp,
                                   kpts=(self.supercell_kpts[0],self.supercell_kpts[1],self.supercell_kpts[2]),
                                   tstress=True,
                                   tprnfor=True)

                #Add the QE calculator to the Atom object
                ASE_atom_obj.calc = qe_calc
                
                #Get total energy (this initiates a self-consistent calculation)
                try:
                    self.raw_energy_dict[charge] = ASE_atom_obj.get_total_energy()
                    calc_successful_update = 1
                except:
                    self.debug_obj.debug_log("AHH! The DFT calculation could not complete!")
                    calc_successful_update = 0

            #Check correction type and make an required files from DFT results
            if run_pp_that_doesnt_require_wfc == True or run_pp_that_requires_wfc == True:

                self.util_obj.changeDirectory(charge_path)

                if(self.correction_type == "FNV"):

                    if run_pp_that_doesnt_require_wfc:
                        plot_num = 1
                        grid_size = self.fs_correction_details['grid_size']
                        self.debug_obj.debug_log(f"grid_size: {grid_size}")
                        self.generate_pp_file(prefix, charge_path,plot_num,grid_size=grid_size)

                    if run_pp_that_requires_wfc == True and charge !=0:

                        plot_num = 7
                        band_num = self.selectCubeBand(self.fs_correction_details,charge)
                        self.generate_pp_file(prefix, charge_path,plot_num,band_num)

            #Now delete the wfc files (because they are very large)
            if self.delete_wfc_files == True:
                if self.type == "PRISTINE":
                    wfc_loc = f"{charge_path}/unitcell/{prefix}.save/"
                    self.util_obj.changeDirectory(wfc_loc)
                    self.util_obj.deleteFiles("wfc",precise = False)
                #for the supercell
                wfc_loc = f"{charge_path}/{prefix}.save/"
                self.util_obj.changeDirectory(wfc_loc)
                self.util_obj.deleteFiles("wfc",precise = False)

            self.util_obj.changeDirectory(global_path)

            calc_successful_count = calc_successful_count + calc_successful_update

        if successCriteria == calc_successful_count:
            self.calc_successful = True

        return self.calc_successful

    # Purpose | Generate full pymatgen defect objects for each charged defect
    # arg | energy_pristine | number | the energy of the pristine supercell
    # arg | pymatgen_parameters | dict of params, e.g. "vbm", "band_gap", "scaling_matrix"
    def ComputeDefectFormationEnergy(self,energy_pristine,pymatgen_parameters):


        #todo list
        #Calculate the chemical potentials of relevant species
        #Calculate the chemical potential of atoms added/removed
        self.debug_obj.debug_log(f"Calculating Defect Formation Energy from {self.type}_{self.type_num}")

        #info from parameters
        ecut = self.qe_parameters['system']['ecutwfc']

        coffeeObject = CoffeeHelper.CoffeeHelper(self.defect_object,
                                                 self.supercell.lattice.matrix,
                                                 self.supercell_dim,
                                                 self.mpid,
                                                 ecut,
                                                 debug_obj=self.debug_obj,
                                                 correction_details = self.fs_correction_details)

        for charge in self.chargeList:

            self.debug_obj.debug_log(f"Calculating DFE for {self.type}_{self.type_num}, charge = {charge}")

            pristine_path = self.supercell_path + '/PRISTINE_0/0/'
            no_charge_path = self.defect_path + '/0/'
            charge_path = self.defect_path + '/' + str(charge) + '/'

            self.util_obj.changeDirectory(charge_path)

            #Uncorrect energy (defect energy - pristine energy)
            energy_charge_defect = self.raw_energy_dict[charge]
            uncorrected_energy = energy_charge_defect - energy_pristine

            #FNV correction
            corrections = {'FNV':0}
            if charge != 0:
                FNV_calc_successful = coffeeObject.calculateFNV(charge,charge_path,no_charge_path,pristine_path)

                if(FNV_calc_successful == False):
                    self.debug_obj.debug_log("FNV Correction failed")
                    return False

                #Provided as dictionary
                FNV_energies = coffeeObject.getFNVCorrection()
                self.debug_obj.debug_log("FNV_energy: " + str(FNV_energies))

                corrections.update(FNV_energies)

            #self.defect_object.set_charge(charge)
            self.defect_object.user_charges = [charge]

            """ #Hold over from old Pymatgen
            defect_entry = Defect(self.defect_object,
                                       uncorrected_energy,
                                       corrections=corrections,
                                       parameters=pymatgen_parameters)
            """

            #defect_entry = Defect(self.defect_object.defect_structure)

            self.pymatgen_defect_objects.append(copy.deepcopy(self.defect_object))

        #global_path = self.qe_parameters['control']['outdir']
        #self.util_obj.changeDirectory(global_path)

    # Purpose | Get the pymatgen defect objects
    def getPymatgenDefectObjects(self):
        return self.pymatgen_defect_objects

    # will check if 1) file exists, 2) job completed,
    # will return dict with 1) job done boolean, 2) energy [eV]
    # filename can be a path
    def search_qe_output(self,filename):

        result = {
            'job_done':False,
            'total_energy': False
        }

        self.debug_obj.debug_log(f"Searching for {filename}")
        file_found = self.util_obj.checkIfFileExists(filename)

        if file_found == True:

            self.debug_obj.debug_log(f"Current directory: {os.getcwd()}")
            self.debug_obj.debug_log(f"The file '{filename}' has been found! Huzzah!")

            job_complete = False
            total_energy_found = False
            total_energy = 0

            with open(filename) as file:

                #Loop through every line in the output file
                for line in file:

                    #multiple lines match. Let run to the end of file
                    if '!    total energy' in line:
                        #self.debug_obj.debug_log(f"Total energy found!")
                        total_energy_found = True

                        unit = line.split(' ')[-1]
                        #self.debug_obj.debug_log(f"Energy unit: {unit}")

                        total_energy = float(line.split(" ")[-2])
                        #self.debug_obj.debug_log(f"Initial energy value: {total_energy}")

                        if "Ry" in unit:
                            #self.debug_obj.debug_log(f"Energy in Ry!")
                            ry_to_eV = 13.6057039763
                            total_energy = total_energy*ry_to_eV

                    if 'JOB DONE.' in line:
                        job_complete = True
                        self.debug_obj.debug_log(f"Job complete found!")

                if total_energy_found and job_complete:
                    result['job_done'] = True
                    result['total_energy'] = total_energy
                    self.debug_obj.debug_log(f"Total energy = {total_energy} eV!")

        return result

    # Attempt to auto select the wfc band (rarely works)
    # If explicitly set, will put the value from correction_details
    def selectCubeBand(self,correction_details,charge):

        band_num = 0
        auto_select_bands = correction_details['auto_select_bands']

        if auto_select_bands == True:
            electrons_at_neutral = False
            defect_level = False

            try:
                electrons_at_neutral = correction_details['electrons_at_neutral']
                defect_level = correction_details['defect_level']
            except:
                self.debug_obj.debug_log(f"To use auto_select_bands, must specify electrons_at_neutral and defect_level.")

            #will not be filled
            if defect_level == 'deep':
                electrons_at_charge = electrons_at_neutral + (-1*charge)
                band_num = math.ceil(electrons_at_charge/2) + 1
            #will be filled because near valence edge
            if defect_level == 'shallow':
                electrons_at_charge = electrons_at_neutral + (-1*charge)
                band_num = math.ceil(electrons_at_charge/2)

        if auto_select_bands == False:
            band_by_charge = False
            try:
                band_by_charge = self.fs_correction_details['band_by_charge']
            except:
                self.debug_obj.debug_log(f"If auto_select_bands FALSE, must specify band_by_charge dictionary for each charge.")
            band_num = band_by_charge[charge]

        return band_num


    def generate_pp_file(self,prefix, defect_directory,plot_num,band_num = None,grid_size = None):

        self.debug_obj.debug_log(f"Generating post processing (either cube or xsf) files.")

        self.util_obj.changeDirectory(defect_directory)

        input_name = f"espresso_pn_{plot_num}.ppi"
        output_name = f"espresso_pn_{plot_num}.ppo"
        pp_filename = f"{self.generic_cube_filename}{plot_num}"
        data = ""

        #this creates a cube file
        if plot_num == 7:
            data = f"&INPUTPP\n    prefix = '{prefix}'\n    outdir = './'\n    filplot = 'temp.data'\n    plot_num = {plot_num}\n    kpoint(1) = 1\n    kband(1) = {band_num}\n/ \n&PLOT\n    nfile = 1\n    filepp(1) = 'temp.data'\n    output_format = 6\n    fileout = '{pp_filename}.cube'\n    iflag = 3\n/ \n"
        #this creates a xsf file (because the cube code didn't allow control over voxel number) (NB - in theory pp.x should allow this, so either my error or QE code error)
        if plot_num == 1:
            if grid_size == None:
                data = f"&INPUTPP\n    prefix = '{prefix}'\n    outdir = './'\n    filplot = 'temp.data'\n    plot_num = {plot_num}\n/ \n&PLOT\n    nfile = 1\n    filepp(1) = 'temp.data'\n    output_format = 3\n    fileout = '{pp_filename}.xsf'\n    iflag = 3\n/ \n"
            else:
                data = f"&INPUTPP\n    prefix = '{prefix}'\n    outdir = './'\n    filplot = 'temp.data'\n    plot_num = {plot_num}\n/ \n&PLOT\n    nfile = 1\n    filepp(1) = 'temp.data'\n    output_format = 3\n    fileout = '{pp_filename}.xsf'\n    iflag = 3\n    interpolation='bspline'\n    nx = {grid_size[0]},ny = {grid_size[1]},nz = {grid_size[2]}\n/\n"


        with open(input_name, 'w',encoding='utf-8',newline='\n') as f:
            f.write(data)
            f.close()

        job_complete = False

        #try without any specification
        args = ["mpirun",self.ESPRESSO_PP_EXE,"-in",input_name,">",output_name]
        self.debug_obj.debug_log(f"pp.x argument: {args}")
        result = subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = result.stdout.decode('utf-8')

        #save the file (will want to see any errors)
        with open(output_name,"w") as file:
            file.write(output)
            file.close()

        # Look through file for job done.
        with open(output_name,"r") as file:
            for line in file:
                if 'JOB DONE.' in line:
                    job_complete = True

        #If no job complete, try again with specification of -np 1
        if job_complete == False:

            self.debug_obj.debug_log(f"Initial espresso.ppi job failed.")
            self.debug_obj.debug_log(f"Re-attempting espresso.ppi job.")
            args = ["mpirun","-np","1",self.ESPRESSO_PP_EXE,"-in",input_name,">",output_name]
            self.debug_obj.debug_log(f"pp.x argument: {args}")
            result = subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            output = result.stdout.decode('utf-8')

            with open(output_name,"w") as file:
                file.write(output)
                file.close()

            with open(output_name,"r") as file:
                for line in file:
                    if 'JOB DONE.' in line:
                        job_complete = True

        if job_complete == False:
            self.debug_obj.debug_log(f"Espresso.ppi job failed.")

        with open(output_name,"w") as f:
            f.write(output)
            f.close()


    #create an ase cell. Set up this way to make sure that ALL digits are used (other methods truncated digits)
    def generate_ase_cell(self,system_dict,pymatgen_object):

        angstroms_to_bohr = 1.8897259886
        system = copy.deepcopy(system_dict)

        lattice_a = angstroms_to_bohr * pymatgen_object.lattice.a #bohr
        lattice_b = angstroms_to_bohr * pymatgen_object.lattice.b #bohr
        lattice_c = angstroms_to_bohr * pymatgen_object.lattice.c #bohr
        self.debug_obj.debug_log(f"lattice_a: {lattice_a}")
        self.debug_obj.debug_log(f"lattice_b: {lattice_b}")
        self.debug_obj.debug_log(f"lattice_c: {lattice_c}")

        system['celldm(1)'] = lattice_a
        system['celldm(2)'] = lattice_b/lattice_a
        system['celldm(3)'] = lattice_c/lattice_a
        system['celldm(4)'] = math.cos(pymatgen_object.lattice.angles[2]*math.pi/180.0)
        system['celldm(5)'] = math.cos(pymatgen_object.lattice.angles[1]*math.pi/180.0)
        system['celldm(6)'] = math.cos(pymatgen_object.lattice.angles[0]*math.pi/180.0)

        ase_cell = ase_qe_support.ibrav_to_cell(system)
        ase_cell = ase_cell[1]

        return ase_cell

#%%
