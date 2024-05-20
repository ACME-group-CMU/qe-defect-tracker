import sympy as sym
import numpy as np
import math
from scipy.optimize import minimize
from pymatgen.transformations.standard_transformations import RotationTransformation
from pymatgen.ext.matproj import MPRester
import importlib

from qe_defect_tracker import Utility
importlib.reload(Utility)

class Optimize_Structure_Rotation():

    def __init__(self,starting_struct,predicted_struct):
        self.starting_struct = starting_struct
        self.predicted_struct_w_variables = predicted_struct
        self.first_obj_value_stored = False
        self.first_obj_value = 100000

    def rotate_structure(self,rotate_a, rotate_b,rotate_c):
        axis = [1,0,0]
        rotate_about_a = RotationTransformation(axis,rotate_a,angle_in_radians=False)
        final_struct = rotate_about_a.apply_transformation(self.starting_struct)

        axis = [0,1,0]
        rotate_about_b = RotationTransformation(axis,rotate_b,angle_in_radians=False)
        final_struct = rotate_about_b.apply_transformation(final_struct)

        axis = [0,0,1]
        rotate_about_c = RotationTransformation(axis,rotate_c,angle_in_radians=False)
        final_struct = rotate_about_c.apply_transformation(final_struct)

        return final_struct

    def get_fibronius_norm(self,potential_struct):

        struct_dict = {
            "a": potential_struct.lattice.a,
            "b": potential_struct.lattice.b,
            "c": potential_struct.lattice.c,
            #"alpha": potential_struct.lattice.angles[0],
            #"beta": potential_struct.lattice.angles[1],
            #"gamma": potential_struct.lattice.angles[2],
            "sin_alpha": math.sin(potential_struct.lattice.angles[0]*math.pi/180.0),
            "sin_beta": math.sin(potential_struct.lattice.angles[1]*math.pi/180.0),
            "sin_gamma": math.sin(potential_struct.lattice.angles[2]*math.pi/180.0),
            "cos_alpha": math.cos(potential_struct.lattice.angles[0]*math.pi/180.0),
            "cos_beta": math.cos(potential_struct.lattice.angles[1]*math.pi/180.0),
            "cos_gamma": math.cos(potential_struct.lattice.angles[2]*math.pi/180.0),
        }

        predicted_struct  = self.predicted_struct_w_variables
        for lattice_key in struct_dict:
            try:
                test_def_variable_value = predicted_struct.subs({lattice_key:struct_dict[lattice_key]})
                predicted_struct = test_def_variable_value
            except:
                print("Variable not in potential ibrav")

        predicted_struct = np.array(predicted_struct)
        abs_diff = np.abs(potential_struct.lattice.matrix - predicted_struct)
        abs_diff = abs_diff.astype(float)
        abs_norm = np.linalg.norm(abs_diff,'fro')
        return abs_norm

    def obj(self,arguments):
        """objective function, to be solved."""

        rotate_a, rotate_b, rotate_c = arguments[0], arguments[1], arguments[2]
        #print(f"rotate_a: {rotate_a}, rotate_b: {rotate_b}, rotate_c: {rotate_c}")
        #take starting structure and rotate
        self.final_struct = self.rotate_structure(rotate_a, rotate_b,rotate_c)
        #print(final_struct.lattice)
        f_norm = self.get_fibronius_norm(self.final_struct)

        if(self.first_obj_value_stored == False):
            self.first_obj_value = f_norm
            self.first_obj_value_stored = True

        #print(f_norm)
        return f_norm

    def run_optimize(self,initial_guess=[0,0,0]):

        bnds = [(0,360),(0,360),(0,360)]
        tol = 1*10**(-7)
        result = minimize(self.obj,initial_guess,method='SLSQP',bounds=bnds,tol=tol)
        #from scipy.optimize import direct
        #result = direct(self.obj,bounds=bnds)
        result.first_val = self.first_obj_value
        return result,self.final_struct

class MP_Helper(object):

    def __init__(self,debug_obj=None):
        self.debug_obj = debug_obj
        self.util_obj = Utility.Utility(self.debug_obj)

        a =  sym.symbols('a')
        b =  sym.symbols('b')
        c =  sym.symbols('c')
        sin_alpha =  sym.symbols('sin_alpha')
        sin_beta =  sym.symbols('sin_beta')
        sin_gamma =  sym.symbols('sin_gamma')
        cos_alpha =  sym.symbols('cos_alpha')
        cos_beta =  sym.symbols('cos_beta')
        cos_gamma =  sym.symbols('cos_gamma')

        self.ibrav_definitions = {
            "1": {'sym_keywords':['cubic'],
                  'sym_level': 1,
                  'matrix':sym.Matrix([[a*1,0,0],
                                       [0,a*1,0],
                                       [0,0,a*1]])
                  },
            "2": {'sym_keywords':['cubic'],
                  'sym_level': 2,
                  'matrix':sym.Matrix([[(a/2)*-1,(a/2)*0,(a/2)*1],
                                       [(a/2)*0,(a/2)*1,(a/2)*1],
                                       [(a/2)*-1,(a/2)*1,(a/2)*0]])
                  },
            "3": {'sym_keywords':['cubic'],
                  'sym_level': 3,
                  'matrix':sym.Matrix([[(a/2)*1,(a/2)*1,(a/2)*1],
                                       [(a/2)*-1,(a/2)*1,(a/2)*1],
                                       [(a/2)*-1,(a/2)*-1,(a/2)*1]])
                  },
            "-3": {'sym_keywords':['cubic'],
                   'sym_level': 3,
                   'matrix':sym.Matrix([[(a/2)*-1,(a/2)*1,(a/2)*1],
                                        [(a/2)*1,(a/2)* -1,(a/2)*1],
                                        [(a/2)*1,(a/2)*1,(a/2)*-1]])
                   },
            "4": {'sym_keywords':['hexagonal','trigonal'],
                  'sym_level': 4,
                  'matrix':sym.Matrix([[a,0,0],
                                       [a*(-1/2),a*math.sqrt(3)/2,0],
                                       [0,0,c]])
                  },
            "5": {'sym_keywords':['trigonal'],
                  'sym_level': 5,
                  'matrix':sym.Matrix([[a*((1-c)/2)**(0.5),-a*((1-c)/6)**(0.5),a*((1+2*c)/3)**(0.5)],
                                       [0,a*2*((1-c)/6)**(0.5),a*((1+2*c)/3)**(0.5)],
                                       [-a*((1-c)/2)**(0.5),-a*((1-c)/6)**(0.5),a*((1+2*c)/3)**(0.5)]])
                  },
            "-5": {'sym_keywords':['trigonal'],
                   'sym_level': 5,
                   'matrix':sym.Matrix([[((((1+2*c)/3)**(0.5)) - 2*(2)**(0.5)*(((1-c)/6)**(0.5)))/(3)**(0.5),
                                         ((((1+2*c)/3)**(0.5)) + 1*(2)**(0.5)*(((1-c)/6)**(0.5)))/(3)**(0.5),
                                         ((((1+2*c)/3)**(0.5)) + 1*(2)**(0.5)*(((1-c)/6)**(0.5)))/(3)**(0.5)],
                                        [((((1+2*c)/3)**(0.5)) + 1*(2)**(0.5)*(((1-c)/6)**(0.5)))/(3)**(0.5),
                                         ((((1+2*c)/3)**(0.5)) - 2*(2)**(0.5)*(((1-c)/6)**(0.5)))/(3)**(0.5),
                                         ((((1+2*c)/3)**(0.5)) + 1*(2)**(0.5)*(((1-c)/6)**(0.5)))/(3)**(0.5)],
                                        [((((1+2*c)/3)**(0.5)) + 1*(2)**(0.5)*(((1-c)/6)**(0.5)))/(3)**(0.5),
                                         ((((1+2*c)/3)**(0.5)) + 1*(2)**(0.5)*(((1-c)/6)**(0.5)))/(3)**(0.5),
                                         ((((1+2*c)/3)**(0.5)) - 2*(2)**(0.5)*(((1-c)/6)**(0.5)))/(3)**(0.5)]])
                   },
            "6":{'sym_keywords':['tetragonal'],
                 'sym_level': 6,
                 'matrix':sym.Matrix([[a,0,0],
                                      [0,a,0],
                                      [0,0,c]])
                 },
            "7":{'sym_keywords':['tetragonal'],
                 'sym_level': 7,
                 'matrix':sym.Matrix([[(a/2)*1,(a/2)*-1,(a/2)*(c/a)],
                                      [(a/2)*1,(a/2)*1,(a/2)*(c/a)],
                                      [(a/2)*-1,(a/2)*-1,(a/2)*(c/a)]])
                 },
            "8":{'sym_keywords':['orthorhombic'],
                 'sym_level': 8,
                 'matrix':sym.Matrix([[a,0,0],
                                      [0,b,0],
                                      [0,0,c]])
                 },
            "9":{'sym_keywords':['orthorhombic'],
                 'sym_level': 9,
                 'matrix':sym.Matrix([[(a/2),(b/2),0],
                                      [(-a/2),(b/2),0],
                                      [0,0,c]])
                 },
            "-9":{'sym_keywords':['orthorhombic'],
                  'sym_level': 9,
                  'matrix':sym.Matrix([[(a/2),(-b/2),0],
                                       [(a/2),(b/2),0],
                                       [0,0,c]])
                  },
            "91":{'sym_keywords':['orthorhombic'],
                  'sym_level': 9,
                  'matrix':sym.Matrix([[a,0,0],
                                       [0,(b/2),(-c/2)],
                                       [0,(b/2),(c/2)]])
                  },
            "10":{'sym_keywords':['orthorhombic'],
                  'sym_level': 10,
                  'matrix':sym.Matrix([[a/2,0,c/2],
                                       [(a/2),(b/2),0],
                                       [0,(b/2),(c/2)]])
                  },
            "11":{'sym_keywords':['orthorhombic'],
                  'sym_level': 11,
                  'matrix':sym.Matrix([[(a/2),(b/2),(c/2)],
                                       [(-a/2),(b/2),(c/2)],
                                       [(-a/2),(-b/2),(c/2)]])
                  },
            "12":{'sym_keywords':['monoclinic'],
                  'sym_level': 12,
                  'matrix':sym.Matrix([[a,0,0],
                                       [b*cos_gamma,b*sin_gamma,0],
                                       [0,0,c]])
                  },
            "-12":{'sym_keywords':['monoclinic'],
                   'sym_level': 12,
                   'matrix':sym.Matrix([[a,0,0],
                                        [0,b,0],
                                        [c*cos_beta,0,c*sin_beta]])
                   },
            "13":{'sym_keywords':['monoclinic'],
                  'sym_level': 13,
                  'matrix':sym.Matrix([[(a/2),0,(-c/2)],
                                       [b*cos_gamma, b*sin_gamma, 0],
                                       [(a/2),0,(c/2)]])
                  },
            "-13":{'sym_keywords':['monoclinic'],
                   'sym_level': 13,
                   'matrix':sym.Matrix([[(a/2),(b/2),0],
                                        [(-a/2),(b/2),0],
                                        [c*cos_beta,0,c*sin_beta]])
                   },
            "14":{'sym_keywords':['triclinic'],
                  'sym_level': 14,
                  'matrix':sym.Matrix([[a, 0, 0],
                                       [b*cos_gamma, b*sin_gamma, 0],
                                       [c*cos_beta,
                                        c*(cos_alpha-cos_beta*cos_gamma)/sin_gamma,
                                        c*(1 + 2*cos_alpha*cos_beta*cos_gamma - cos_alpha**2-cos_beta**2-cos_gamma**2)**(0.5)/sin_gamma]])
                  },
        }


    def find_best_qe_ibrav(self,molecule_species_mpid,only_test_ibravs = False,print_info=False,test_all_ibrav = False):

        selection_threshold = 1*10**-4
        valid_result = False
        symmetry_found = False
        lowest_abs_diff = 1000
        best_structure = {
            'struct': False,
            'ibrav': '14',
            'atom_num': 100000,
        }

        def get_mp_structure(molecule_species_mpid):
            docs = mpr.materials.search(material_ids=[molecule_species_mpid], fields=["structure"]);
            return docs[0].structure;


        with MPRester(self.util_obj.checkEnvironmentalVars("MPI_API_KEY")) as mpr:
            #docs = mpr.materials.search(material_ids=[molecule_species_mpid], fields=["structure"]);
            primitive_structure = get_mp_structure(molecule_species_mpid);
            try:
                docs_sym = mpr.materials.summary.search(material_ids=[molecule_species_mpid], fields=["symmetry"]);
                primitive_structure_sym = docs_sym[0].symmetry.crystal_system.__dict__['_value_'];
                symmetry_found = True
            except:
                print(f"No symmetry found for material {molecule_species_mpid}")

        #fetch conventional structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(primitive_structure)
        conventional_structure = sga.get_conventional_standard_structure()

        #fetch another basic structure type
        primitive_standard = sga.get_primitive_standard_structure()

        structures = {
            "pymatgen_primitive":primitive_structure,
            "primitive_standard": primitive_standard,
            "pymatgen_conv":conventional_structure,
        }

        for structure in structures:

            potential_struct = structures[structure]
            print(f"\nTesting Structure: {structure} (atoms: {potential_struct.num_sites})")

            for ibrav in self.ibrav_definitions:

                #If enforced, check these ibravs
                if test_all_ibrav == False:
                    if only_test_ibravs is not False:
                        if ibrav not in only_test_ibravs:
                            continue
                    else: #else, just use MP symmetry info
                        if symmetry_found:
                            if primitive_structure_sym.lower() not in self.ibrav_definitions[ibrav]['sym_keywords']:
                                continue

                initial_guesses = [[0,0,0]]

                for angle_guess in initial_guesses:

                    print(f"Testing ibrav: {ibrav}")
                    if(print_info):
                        print(f"############################")
                        print(f"Testing ibrav: {ibrav}")
                        print(f"Ibrav: {ibrav}")
                        print(f"MP Lattice:")

                    potential_ibrav = self.ibrav_definitions[ibrav]['matrix']

                    osr = Optimize_Structure_Rotation(potential_struct,potential_ibrav)
                    result,final_struct = osr.run_optimize(initial_guess=angle_guess)
                    first_val  = result.first_val
                    abs_norm = result.fun
                    print(f"Objective Function: {(result.fun)}")

                    if(print_info):
                        print(result)
                        print(f"Objective Function: {(result.fun)}")
                        print(f"Angle #1: {round(result.x[0])}")
                        print(f"Angle #2: {round(result.x[1])}")
                        print(f"Angle #3: {round(result.x[2])}")
                        for lst in final_struct.lattice.matrix:
                            print(*lst)

                    #As long as it meets threshold, can be selected as best struct
                    if(abs_norm < selection_threshold):

                        #current
                        best_struct_atom_num = best_structure['atom_num']
                        best_struct_ibrav = best_structure['ibrav']
                        best_struct_symmetry_level = self.ibrav_definitions[best_struct_ibrav]['sym_level']

                        #new final struct
                        new_struct_atom_num = final_struct.num_sites
                        new_symmetry_level = self.ibrav_definitions[ibrav]['sym_level']

                        update_best = False
                        if(best_structure['struct'] == False):
                            update_best = True
                        elif(new_struct_atom_num < best_struct_atom_num):
                            update_best = True
                        elif(new_struct_atom_num == best_struct_atom_num):
                            if(new_symmetry_level < best_struct_symmetry_level):
                                update_best = True

                        if(update_best):
                            valid_result = True
                            best_structure['struct'] = final_struct
                            best_structure['ibrav'] = ibrav
                            best_structure['atom_num'] = final_struct.num_sites

            #Will stop if result found. Penny wise, pound foolish?
            #if(valid_result):
            #    break

        if(print_info):
            if(valid_result == True):
                print(f"MPID: {molecule_species_mpid} has ibrav: {best_structure['ibrav_num']}")
            else:
                print(f"MPID: {molecule_species_mpid} has no ibrav fit!")


        #print(f"Best Ibrav {best_ibrav}")
        #print(f"Best Structure {best_structure.lattice}")

        return valid_result,best_structure








