
#Pymatgen
from pymatgen.analysis.defects.generators import VacancyGenerator
from pymatgen.analysis.defects.generators import SubstitutionGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.ext.matproj import MPRester
from dotenv import load_dotenv

#other
import os
import importlib
from dotenv import load_dotenv

#Other Defect Trackin Items
from qe_defect_tracker import SingleSupercell
importlib.reload(SingleSupercell)

from qe_defect_tracker import Debug
importlib.reload(Debug)

from qe_defect_tracker import Utility
importlib.reload(Utility)

class Tracker:

    # arg | pristine_supercell | Pymatgen Object of Pristine Supercell
    #pymatgen_unitcell - A self-defined unit cell. The supercell_dim will be applied to make supercell
    def __init__(self,unit_cell_type,
                 pymatgen_unitcell = None,
                 supercell_dim = None,
                 molecule_species_mpid = None,
                 delete_wfc_files=False):

        self.debug = True
        self.debug_obj = Debug.Debug(self.debug)
        self.util_obj = Utility.Utility(self.debug_obj)

        #load in config info from the .env
        load_dotenv(override=True)

        self.ESPRESSO_PW_EXE = self.util_obj.checkEnvironmentalVars("ESPRESSO_PW_EXE")
        print(f"self.ESPRESSO_PW_EXE: {self.ESPRESSO_PW_EXE}")
        self.util_obj.checkEnvironmentalVars("ESPRESSO_PP_EXE") #just check, don't need to save
        coffee_dir = self.util_obj.checkEnvironmentalVars("COFFEE_DIR") #just check, don't need to save
        self.util_obj.checkEnvironmentalVars("MPI_API_KEY")
        self.debug_obj.kill_if_errors()

        self.debug_obj.debug_log(f"coffee_dir: {coffee_dir}")

        self.unit_cell_type = unit_cell_type
        self.mpid = molecule_species_mpid
        self.supercell_dim = supercell_dim

        self.qe_parameters = None
        self.pseudopotentials = None
        self.kpts = None
        self.defect_dict = {} #key is composition_position
        self.calc_successful = False
        self.delete_wfc_files = delete_wfc_files

        self.type_count = {
            "VAC": 0,
            "SUB": 0,
            "INT": 0,
        }

        supercell_pristine = None
        unitcell_pristine = None
        if self.unit_cell_type == 'primitive' or self.unit_cell_type == 'conventional':

            #Create a pristine supercell
            with MPRester(self.util_obj.checkEnvironmentalVars("MPI_API_KEY")) as mp:
                #This will be the primitive structure
                unit_cell_structure = mp.get_structure_by_material_id(molecule_species_mpid,) #Au

            if self.unit_cell_type == 'primitive':
                unitcell_pristine = unit_cell_structure

            if self.unit_cell_type == 'conventional':
                unitcell_pristine = SpacegroupAnalyzer(unit_cell_structure).get_conventional_standard_structure()

        if self.unit_cell_type == 'self-defined':
            unitcell_pristine = pymatgen_unitcell

        #now generate the pristine object that will be copied to make all defects
        self.pristine_object = SingleSupercell.SingleSupercell(type ="PRISTINE",
                                                               type_num=0,
                                                               unitcell = unitcell_pristine,
                                                               supercell_dim = self.supercell_dim,
                                                               debug_obj = self.debug_obj,
                                                               delete_wfc_files = self.delete_wfc_files)


    #Enables control of the parallelization parameters np,nk,nb,nt,nd
    #Optional. Without parameters, parallelization parameters auto determined by QE
    def setEspressoCommand(self,compute_parameters):
        
        if compute_parameters == {} or compute_parameters == False or compute_parameters == None:
            espresso_command = f"mpirun {self.ESPRESSO_PW_EXE} -in PREFIX.pwi > PREFIX.pwo"
            os.environ["ASE_ESPRESSO_COMMAND"] = espresso_command
            self.debug_obj.debug_log(f"Espresso Command: {espresso_command}")
            return
        
        np_str = ''
        nk_str = ''
        nb_str = ''
        nt_str = ''
        nd_str = ''

        if 'np' in compute_parameters.keys():
            np_str = f"-np {compute_parameters['np']} "
        
        if 'nk' in compute_parameters.keys():
            nk_str = f"-nk {compute_parameters['nk']} "
            
        if 'nb' in compute_parameters.keys():
            nb_str = f"-nb {compute_parameters['nb']} "

        if 'nt' in compute_parameters.keys():
            nt_str = f"-nt {compute_parameters['nt']} "

        if 'nd' in compute_parameters.keys():
            nd_str = f"-nd {compute_parameters['nd']} "
            
        espresso_command = f"mpirun {np_str}{self.ESPRESSO_PW_EXE} {nk_str}{nb_str}{nt_str}{nd_str}-in PREFIX.pwi > PREFIX.pwo"
        os.environ["ASE_ESPRESSO_COMMAND"] = espresso_command        
        self.debug_obj.debug_log(f"Espresso Command: {espresso_command}")

    # Purpose | Sets relevant parameters and initiates DFT calc for pristine supercell
    # arg | qe_parameters | dictionary object of QE arguments
    # arg | pseudopotentials | dictionary of pseudopotentials
    # arg | kpts | list of kpoints | e.g. [2,2,2]
    # Note: This function does two things (sets params and starts DFT calc) because
    #       I don't want a mismatch between QE params and DFT-calculated energy.
    # Note: If NEW params/pseudos/kpts are set, the old defect energies are cleared out
    def calculatePristineEnergy(self,qe_parameters,pseudopotentials,kpts=None,min_kpt_density=None,
                                compute_parameters=None,correction_params = None,force_recalc = False):

        self.setEspressoCommand(compute_parameters)

        self.debug_obj.debug_log("Calculate Pristine Energy")

        #Automatically clear out all defect energies
        #A mismatch in parameters/pseudos/kpts would lead to incorrect results
        if self.qe_parameters is not None and self.qe_parameters != qe_parameters:
            self.clearDefectEnergies()

        self.qe_parameters = qe_parameters
        self.pseudopotentials = pseudopotentials
        self.kpts = kpts
        self.min_kpt_density = min_kpt_density

        #Check that any info used in code is present
        if(self.check_qe_parameters() == False):
            self.debug_obj.debug_log("Necessary values not provided!",type="ERROR")
            return False

        global_path = os.getcwd()

        result = self.pristine_object.runDFTCalculation(self.qe_parameters,
                                                        self.pseudopotentials,
                                                        kpts = self.kpts,
                                                        min_kpt_density = self.min_kpt_density,
                                                        correction_params = correction_params,
                                                        force_recalc = force_recalc)

        self.util_obj.changeDirectory(global_path)

        self.debug_obj.debug_log(f"DFT Calculation of Pristine Supercell: {str(result)}" ,type="LOG")

    # Purpose | Calculates the DFT energy for all (or all empty) defect instances
    # arg | forceRecalculation | boolean, added to allow full re-calc of all energies
    def calculateDefectEnergy(self,force_recalc = False,correction_params = None):

        self.debug_obj.debug_log(f"Calculating Defect Energy")
        
        pristine_energy = self.getPristineEnergy()
        pymatgen_parameters = {}

        correction_type = correction_params['type']

        if not self.defect_dict:
            self.debug_obj.debug_log("No defects added!" ,type="LOG")
            return

        global_path = os.getcwd()

        for defect in self.defect_dict:

            self.util_obj.changeDirectory(global_path) #make sure in the global folder
            
            self.debug_obj.debug_log(f"Calculating Defect Energy of defect: {defect}")

            if(self.calc_successful == False or force_recalc == True):
                self.defect_dict[defect].runDFTCalculation(self.qe_parameters,
                                                           self.pseudopotentials,
                                                           self.kpts,
                                                           correction_params = correction_params,
                                                           force_recalc = force_recalc)

            self.defect_dict[defect].ComputeDefectFormationEnergy(pristine_energy,
                                                                  pymatgen_parameters)

    # Purpose | Add a single defect to the pristine supercell (will check uniqueness)
    # arg | defectType | string of defect type | i.e. "VAC","SUB", or "INT"
    # arg | defectComp | dictionary of defect update to composition | e.g. {'Al':-1} for an aluminum vacacny
    # arg | relevantSpecies | list of strings of species (necessary for chemical potential)
    # Note: Creates a defect instance, checks if unique, if so add to object
    def addDefects(self,defectType,defectComp,chargeList = False):
        
        self.debug_obj.debug_log(f"\nAdding Defect: {defectType}, {defectComp}")

        defectTypeFound = False
        defects_to_add = {}

        def check_if_matching_composition(comp1,comp2):
            for item in comp1:
                if comp1[item] != comp2[item]:
                    return False
            return True

        allowed_defect_composition = (self.pristine_object.supercell.composition.as_dict()).copy()
        self.debug_obj.debug_log(f"allowed_defect_composition: {allowed_defect_composition}")

        #Update the allowed defect composition to reflect the provided defectComp
        for itemToChange in defectComp:
            if itemToChange in allowed_defect_composition:
                pristine_quantity = allowed_defect_composition[itemToChange]
                change_to_pristine = defectComp[itemToChange]
                allowed_defect_composition[itemToChange] = pristine_quantity + change_to_pristine
            else:
                allowed_defect_composition[itemToChange] = defectComp[itemToChange]

        if defectType == "VAC":
            defectTypeFound = True
            vac_generator = VacancyGenerator()
            for vac_defect in vac_generator.generate(self.pristine_object.supercell):
                #define a unique defect name (will be used to block non-unique additons)
                comp = str(vac_defect.defect_structure.composition.element_composition).replace(" ","_")
                defect_name = "VAC_" + str(comp) + "_" + str(vac_defect.site)
                self.debug_obj.debug_log(defect_name)
                #create an addition to the dictionary of defects to adds
                defects_to_add[defect_name] = vac_defect

        if defectType == "SUB":
            defectTypeFound = True

            sub_element = ""
            for element in defectComp:
                if defectComp[element] == 1:
                    sub_element = element
                    break

            for sub_defect in SubstitutionGenerator(self.pristine_object.supercell,element=sub_element):
                self.debug_obj.debug_log(sub_defect.defect_composition)
                #define a unique defect name (will be used to block non-unique additons)
                comp = str(sub_defect.defect_structure.composition.element_composition).replace(" ","_")
                defect_name = "SUB_" + str(comp) + "_" + str(sub_defect.site)
                #create an addition to the dictionary of defects to adds
                defects_to_add[defect_name] = sub_defect
            #fill out later

        if defectType == "INT":
            defectTypeFound = False
            #fill out later

        if defectTypeFound == False:
            self.debug_obj.debug_log(f"No defect found of type: {str(defectType)}",type = "WARNING")
            return False

        #Now iterate through the possible defects to add to the system
        for newKey in defects_to_add:

            #Check that the defect is the right composition as specified in arguments
            new_comp_dict = defects_to_add[newKey].defect_structure.composition.element_composition.as_dict()
            if check_if_matching_composition(allowed_defect_composition,new_comp_dict):

                #Check that the defect has not already been added (no dupes!)
                if newKey not in self.defect_dict.keys():
                    self.debug_obj.debug_log(f"New unique defect: {newKey}")
                    if defectType == "SUB":
                        self.type_count["SUB"] += 1
                    if defectType == "VAC":
                        self.type_count["VAC"] += 1
                    if defectType == "INT":
                        self.type_count["INT"] += 1

                    supercell_structure = defects_to_add[newKey].defect_structure
                    
                    self.defect_dict[newKey] = SingleSupercell.SingleSupercell(
                        type = defectType,
                        type_num = self.type_count[defectType],
                        supercell = supercell_structure,
                        supercell_dim = self.supercell_dim,
                        defect_object = defects_to_add[newKey],
                        mpid = self.mpid,
                        debug_obj = self.debug_obj,
                        delete_wfc_files = self.delete_wfc_files)

                    self.defect_dict[newKey].selectDefectCharges(chargeList)

    # Purpose | clear out all defects
    def clearDefectDict(self):
        self.defect_dict = {}

    # Purpose | Clear all defect energies
    # Note: This will automatically happen if the QE params have been altered
    def clearDefectEnergies(self):
        for defect in self.defect_dict:
            self.defect_dict[defect].clearDefectEnergies()

    # Purpose | Get the energy of the pristine supercell
    def getPristineEnergy(self):
        energy_dict = self.pristine_object.getRawEnergies()
        return energy_dict[0]

    # Purpose | Get the energy of every defect (provided as dictionary w.r.t the defect charge)
    def getDefectEnergies(self):

        if self.defect_dict is None:
            self.debug_obj.debug_log("No defects exist yet!",type="LOG")
            return False

        dict_temp = {}
        for defect in self.defect_dict:
            dict_temp[defect] = self.defect_dict[defect].getRawEnergies()

        return dict_temp

    # Purpose | Check that any values used in code are present in the parameters
    # Note: This is hacky right now. If more terms need to be validated,
    # make a dictionary of regexes and check against that
    def check_qe_parameters(self):

        #ecutwfc is used. Checked here to make sure no quiet errors

        if 'ecutwfc' not in self.qe_parameters['system'].keys():
            return False

        """"
        ecut = self.qe_parameters['system']['ecutwfc']
        isNumber = isinstance(ecut, (int, float))

        if isNumber == False:
            return False
        """
        return True

    def getPymatgenDefectObjects(self):
        all_defects = []
        for defect in self.defect_dict:
            defect_entries = self.defect_dict[defect].getPymatgenDefectObjects()
            all_defects.append(defect_entries.copy())
        return all_defects

#%%
