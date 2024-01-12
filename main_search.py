# -*- coding: utf-8 -*-
r"""
Created on Fri Jan 31 14:51:43 2020

@author: Paul M. Neves


OVERVIEW:
Automatically calculates tight binding models

V2:
Updated version reworks how save file is made (now produces one line for
each FB sublattice found, instead of one line per material).


IMPORTANT VARIABLES:
 - param contains all the parameters and settings of the automatic calculation
     (see class definition for more details)
 - mat_ID is the string that contains the MP ID that is currently being calculated
 - print_str is the string that will be printed at the end of the current calculation
 - write_str is the string that will be saved to the output file at the end of the
     current calculation


GLOSSARY:
TB = tight binding
MP = Materials Project
FB = Flat Band


NOTE: Make sure that you have run "get_all_mp_IDs.py" and "get_all_structs_from_MP.py"
in the same directory as this program before running this script. These two auxiliary
scripts download the entire list of materials and their structures from the materials
project to files (all_mp_IDs.txt, all_mp_names.txt, all_mp_nsites.txt, and
mp_structs/mp-XXX.json) so that this program needs only to read these from
your filetree instead of having to pull them from the MP website each time you run.
This speeds up the whole calculation dramatically.
"""


# IMPORT LIBRARIES
# pymatgen functions
from pymatgen.ext.matproj import MPRester
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
from pymatgen.symmetry.kpath import KPathSetyawanCurtarolo, KPathLatimerMunro
from pymatgen.core import IStructure
from pymatgen.core.composition import Composition

# other materials science-ey libraries
from mendeleev import element
from pythtb import tb_model

# science-y and data-ey libraries
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
try:
    import addcopyfighandler
except Exception:
    pass

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# some standard python libraries
import os
import sys
import shutil
from time import perf_counter
import time
import traceback
import warnings
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from datetime import datetime
import json
import pickle
import itertools


class paramObj():
    r"""
    Class that stores/defines all user controllable parameters in a calculation.

    Values:
        General i/o settings
            verbose (bool): Whether to print results to screen or not.
            save_results (bool): Whether to save the reults or not.
            run_parallel (bool): Whether or not to parallelize the calculation.
            MP_KEY (string): Can get MP key (called the "API key") from MP dashboard
                at https://www.materialsproject.org/dashboard

        Tight binding settings:
            max_dist (float): Max bond distance to consider hopping of - units
                specified by dist_units.
            decay_rate (float): Rate at which hopping decays in same units as max_dist. If assigned
                a value of None, then the hopping amplitude is constant, regardless of neighbor
                distance.
            dist_units (string): Units of max_dist, either 'nn': in multiples of the nearest
                neighbor distance on this sublattice, or 'AA': in angstroms.
            max_fb_width (float): Max energy bandwidth to consider a flat band, in units of fraction
                of energy range across all bands .
            flatness_tol (float): Fraction of all calculated k points that need to be flat to be
                counted.
            full_flat (bool): Whether to check that the flat bands are across the whole BZ or just
                use the DOS values.
            check_to_N_hops (int): Checks up to N hops away to see if a flatband is trivially localized
                or not. Increased N will find more non-trivial flat-bands, but will take longer.
            high_symm_only (bool): Look for flat bands only along high symm. only or across entire BZ.

        What calculations to do:
            reduce_structure (bool): Reduces cell before calculating TB.
            print_structure_data (bool): Output structure info to terminal.
            plot_structure_3D (bool): Plot unit cell of crystal in 3D.
            simple_graphics (bool): Use circles or spheres for atoms (spheres take a LOT of graphics
                power).
            plot_TB (bool): Calculate TB and plot results.
            just_save_plots (bool): True if you just want to save the TB/crystal structure figure but
                not plot it.
            plot_TBDOS (bool): Plot DOS along with bandstructure of TB model.
            print_NN_dists (bool): Outputs near neighbor distances used to terminal.
            print_TB_params (bool): Output TB model information to the terminal.
            check_TB_flat (bool): Check for flat bands in TB model.
            plot_BS (bool): Plot BS of structure, if on MP.
            plot_DOS (bool): Plot DOS of structure, if on MP.
            plot_DOS_and_BS (bool):  Plot BS and DOS of structure, if on MP.
            plot_BZ (bool): Plot 1BZ in 3D.
            save_cgd (bool): Whether to save a .cgd file for Gavrog/Systre (see Gavrog.org)
                of all the flat band crystal nets found. Useful for model classification!
            script_path (path object): Path to the parent directory of this file. Make sure
                that this directory also contains mp_structs, all_mp_IDs.txt,
                all_mp_names.txt, and all_mp_nsites.txt.

        The list of materials to calculate:
            material_nums (list of ints, or "all"): Defines which material(s) from list of materials
                in all_mp_names.txt to calculate.
    """
    verbose = True
    save_results = True
    run_parallel = True
    MP_KEY = "YOUR KEY HERE"

    max_dist = 1.2
    decay_rate = None
    dist_units = "NN"
    max_fb_width = 0.001
    flatness_tol = 1.0
    full_flat = True
    check_to_N_hops = 5
    high_symm_only = False

    reduce_structure = False
    print_structure_data = False
    plot_structure_3D = False
    simple_graphics = True
    plot_TB = False
    just_save_plots = True
    plot_TBDOS = False
    print_NN_dists = False
    print_TB_params = False
    check_TB_flat = True
    plot_BS = False
    plot_DOS = False
    plot_DOS_and_BS = False
    plot_BZ = False
    save_cgd = True

    script_path = Path(os.path.dirname(os.path.realpath(__file__)))

    def __init__(self, mat_list=None):
        r"""
        Defines which material(s) from list of materials in all_mp_names.txt to calculate.
        Also now permits lists of strings of mp ids for mat_list.

        Options for material_nums are 'all', which calculates all, or just a list of numbers
        corresponding to lines in the all_mp_names.txt file.
        """

        # get lists of materials info:
        try:
            self.material_IDs, self.names_list, self.nsites_list = load_all_MP_names(
                info_path=self.script_path)
            if mat_list is not None:
                if isinstance(mat_list[0], int) or isinstance(mat_list[0], np.int64):
                    self.material_nums = mat_list
                elif isinstance(mat_list[0], str):
                    self.material_nums = []
                    for i in mat_list:
                        self.material_nums.append(
                            int(np.where(self.material_IDs == i)[0][0]))
                else:
                    self.material_nums = range(len(self.material_IDs))
            else:
                self.material_nums = range(len(self.material_IDs))
        except Exception:
            pass

    def as_dict(self):
        r"""Converts class to a dictionary so it is json serializeable."""
        if isinstance(self.material_nums, np.ndarray):
            mat_nums_ser = self.material_nums.tolist()
        elif isinstance(self.material_nums, range):
            mat_nums_ser = list(self.material_nums)
        else:
            mat_nums_ser = self.material_nums
        this_dict = {
            "verbose": self.verbose,
            "save_results": self.save_results,
            "run_parallel": self.run_parallel,
            "MP_KEY": self.MP_KEY,
            "max_dist": self.max_dist,
            "decay_rate": self.decay_rate,
            "dist_units": self.dist_units,
            "max_fb_width": self.max_fb_width,
            "flatness_tol": self.flatness_tol,
            "full_flat": self.full_flat,
            "check_to_N_hops": self.check_to_N_hops,
            "high_symm_only": self.high_symm_only,
            "reduce_structure": self.reduce_structure,
            "print_structure_data": self.print_structure_data,
            "plot_structure_3D": self.plot_structure_3D,
            "simple_graphics": self.simple_graphics,
            "plot_TB": self.plot_TB,
            "just_save_plots": self.just_save_plots,
            "plot_TBDOS": self.plot_TBDOS,
            "print_NN_dists": self.print_NN_dists,
            "print_TB_params": self.print_TB_params,
            "check_TB_flat": self.check_TB_flat,
            "plot_BS": self.plot_BS,
            "plot_DOS": self.plot_DOS,
            "plot_DOS_and_BS": self.plot_DOS_and_BS,
            "plot_BZ": self.plot_BZ,
            "save_cgd": self.save_cgd,
            "script_path": str(self.script_path),
            "material_nums": mat_nums_ser,
            "material_IDs": self.material_IDs.tolist(),
            "names_list": self.names_list.tolist(),
            "nsites_list": self.nsites_list.tolist()
        }
        return this_dict


class TBoutputs():
    r"""
    A class that can store the outputs from a TB output datafile.

    Must supply with a valid file name string to the datafile you want
    to get the outputs of. Then the data is stored into the following variables:

    Properties:
        header (int): The number of header lines.
        nFB_lats (int): The number of materials in the output file.
        mp_id (str list): The nMatsx1 list of materials project IDs.
        chem_name (str list): The nMatsx1 list of materials chemical names.
        specie_names (str mat.): A nMatsx10 array of names of sublattice species for each material.
        nFBs (int mat.): The corresponding number of flat bands for each specie.
        nsites (int mat): The corresponding number of sites for each specie.
        NN_dists (float mat): The corresponding nearest neighbor distance in angstroms.
        NN_dists_max (float mat): The corresponding largest nearest neighbor distance included in the model.
        NNN_dists (float mat): The first neighbor distance not included in angstroms.
    """

    def __init__(self, file_name, rmv_nonflat=True):
        r"""Loads all lists of results from file."""
        # get number of header lines
        with open(file_name, "r") as f_txt:
            for line_ind, line in enumerate(f_txt.readlines()):
                if line.startswith("mp_ID, "):
                    self.header = line_ind + 1
                    break

        # get all the date from the file
        self.mp_id = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                   usecols=0, dtype=str, invalid_raise=False)
        self.chem_name = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                       usecols=1, dtype=str, invalid_raise=False)
        self.specie_names = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                          usecols=2, dtype=str,
                                          missing_values='', invalid_raise=False)
        self.nFBs = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                  usecols=3, dtype=np.int16,
                                  filling_values=0, invalid_raise=False)
        self.nsites = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                    usecols=4, dtype=np.int16,
                                    filling_values=0, invalid_raise=False)
        self.NN_dists = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                      usecols=5, dtype=np.double,
                                      filling_values=0, invalid_raise=False)
        self.NN_dists_max = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                          usecols=6, dtype=np.double,
                                          filling_values=0, invalid_raise=False)
        self.NNN_dists = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                       usecols=7, dtype=np.double,
                                       filling_values=0, invalid_raise=False)

        # filter to just lattices containing flat bands
        FB_inds = np.nonzero(self.nFBs > 0)[0]
        if rmv_nonflat:
            self.mp_id = self.mp_id[FB_inds]
            self.chem_name = self.chem_name[FB_inds]
            self.specie_names = self.specie_names[FB_inds]
            self.nFBs = self.nFBs[FB_inds]
            self.nsites = self.nsites[FB_inds]
            self.NN_dists = self.NN_dists[FB_inds]
            self.NN_dists_max = self.NN_dists_max[FB_inds]
            self.NNN_dists = self.NNN_dists[FB_inds]
        self.nFB_lats = len(self.mp_id)


def init_file(param, save_path, time_str):
    r"""Creates and begins a TB output file."""
    if param.check_TB_flat:
        f_txt = open(
            save_path / (f"{time_str}_TB_search_results.txt"), "a", buffering=1)
        f_txt.write(f"File created on: {time_str}\nFile output from automated tight binding model "
                    "generator written by Paul M. Neves.\nSearches for (not trivially localized) flat "
                    "band hosting tight binding models using an s-orbital\ntight binding model for each "
                    "species' sublattice.\n\nParameters of search:\n")
        if param.dist_units == 'AA':
            f_txt.write(f"Maximum considered hopping distance: {param.max_dist} angstroms\n"
                        f"Decay rate of hopping strength: {param.decay_rate} (in units of angstroms)\n")
        else:
            f_txt.write(f"Maximum considered hopping distance: {param.max_dist} times the N.N. "
                        f"distance\nDecay rate of hopping strength: {param.decay_rate} (in units of "
                        f"N.N. distance)\n")
        f_txt.write(f"Maximum width of flat band (as a fraction of total band energy spread: "
                    f"{param.max_fb_width}\nFraction of all calculated k-points that need to be flat "
                    f"to count: {param.flatness_tol}\nNumber of hops considered when checking if the "
                    f"flat band is trivial: {param.check_to_N_hops}\n")
        if param.full_flat:
            f_txt.write("Required band to exist in all kpoints samples\n")
        if param.high_symm_only:
            f_txt.write(
                "Only checked band flatness along high symmetry directions\n\n")
        else:
            f_txt.write("Sampled entire 1BZ for band flatness\n\n")
        f_txt.write(
            "mp_ID, formula, El, Nfbs, nsites, nn_dist, nn_dist_max, nnn_dist\n")

        # also save param object
        with open(save_path / (f"{time_str}_TB_params.json"), 'w') as f_json:
            json.dump(param.as_dict(), f_json, indent=4)
        with open(save_path / (f"{time_str}_TB_params.pickle"), 'wb') as f_pickle:
            pickle.dump({"param": param}, f_pickle)
    return f_txt


def load_structure(param, mat_ID):
    r"""
    Loads a chemical structure given a materials project ID.

    First attempts to load a pre-saved structure from a json file. If this
    attempts to get the structure from the materials project database. If
    this fails, raises an error.
    """
    file_path = param.script_path / Path(f"mp_structs/{mat_ID}.cif")
    has_struct = False
    if not os.path.exists(file_path):
        structure = get_struct_from_MP(param, mat_ID)
        if structure is None:
            has_struct = False
        else:
            has_struct = True
    else:
        try:
            structure = IStructure.from_file(file_path)
            has_struct = True
        except Exception:
            has_struct = False
            pass

    # reduces to 'niggli' cell or 'LLL' cell, if desired
    if param.reduce_structure and has_struct:
        structure = structure.get_reduced_structure(reduction_algo='niggli')

    if not has_struct:
        raise ValueError(f"Unable to find a structure for material {mat_ID}!")

    return structure


def get_struct_from_MP(param, mat_ID):
    r"""Loads structure object from the MP."""
    structure = None
    with MPRester(param.MP_KEY, notify_db_version=False) as m:
        structure = m.get_structure_by_material_id(mat_ID)
    # except Exception:
    #    print(f"Unable to download structure from {mat_ID}")
    #    print(traceback.format_exc())
    return structure


def get_bs_from_MP(param, mat_ID):
    r"""Loads bs object from the MP."""
    bs = None
    with MPRester(param.MP_KEY, notify_db_version=False) as m:
        try:
            bs = m.get_bandstructure_by_material_id(mat_ID)
        except Exception:
            print(traceback.format_exc())
            pass
    return bs


def get_dos_from_MP(param, mat_ID):
    r"""Loads dos object from the MP."""
    dos = None
    with MPRester(param.MP_KEY, notify_db_version=False) as m:
        dos = m.get_dos_by_material_id(mat_ID)
    return dos


def save_struct_file(param, save_path, mat_ID):
    r"""Saves the structure of the desired material as a cif file."""
    structure = None
    try:
        structure = get_struct_from_MP(param, mat_ID)
    except Exception:
        print(traceback.format_exc())
        pass
    if structure is not None and structure != []:
        # save to file
        print(f"Now saving struct of: {mat_ID}")
        newfile_path = save_path / (f"{mat_ID}.cif")
        serialized_struct = structure.to(fmt="cif", filename=newfile_path)
    return


def save_bs_file(param, save_path, mat_ID):
    r"""Saves the bs of the desired material as a cif file."""
    bs = None
    try:
        bs = get_bs_from_MP(param, mat_ID)
    except Exception:
        print(traceback.format_exc())
        pass
    print(bs is None)
    if bs is not None:
        # save to file
        print(f"Now saving bs of: {mat_ID}")
        newfile_path = save_path / (f"{mat_ID}.json")
        with open(newfile_path, 'wb') as f_bs:
            json.dump({"bs": bs}, f_bs)
            size = os.path.getsize(str(f_bs))
            print(f"Saved bs with a file size of {size} bytes")
    return


def save_dos_file(param, save_path, mat_ID):
    r"""Saves the dos of the desired material as a cif file."""
    dos = None
    try:
        dos = get_dos_from_MP(param, mat_ID)
    except Exception:
        print(traceback.format_exc())
        pass
    if dos is not None:
        # save to file
        print(f"Now saving dos of: {mat_ID}")
        newfile_path = save_path / (f"{mat_ID}.json")
        with open(newfile_path, 'wb') as f_dos:
            json.dump({"dos": dos}, f_dos)
    return


def make_one_structs_json(script_path, mat_IDs):
    r"""Save all structs to one json file."""
    structures = {}
    for mat_ID in mat_IDs:
        print(f"Now collecting: {mat_ID}")
        file_path = script_path / (f"mp_structs/{mat_ID}.json")
        with open(file_path, 'r') as f_json:
            jsonObjs = json.load(f_json)
        structure = jsonObjs["structure"]
        structures.update({mat_ID: structure})

    file_path = script_path / 'all_mp_structs.json'
    with open(file_path, 'w') as f_json:
        json.dump({"structures": structures}, f_json)
    return


def refresh_MP_data(param, save_directory):
    r"""Downloads and saves all IDs, chemical names, site numbers, and structures from the MP."""
    # creates a list of all elements
    all_elements = [0]*118
    for elem_ind in range(118):
        all_elements[elem_ind] = element(elem_ind+1).symbol

    # gets... EVERY materials project ID that has ANY element in it (THIS WILL TAKE A LITTLE WHILE)
    compounds = []
    print("Obtaining all MP materials...")
    with MPRester(param.MP_KEY, notify_db_version=False) as m:
        compounds.extend(m.query(criteria={"elements": {"$in": all_elements}},
                                 properties=['material_id', 'pretty_formula', 'full_formula', 'elements',
                                             'nsites', 'unit_cell_formula'],
                                 chunk_size=0))

    # extracts the names, IDs, and nsites of each material
    print(f"{len(compounds)} compounds found")
    compounds_list = ['']*len(compounds)
    best_names = ['']*len(compounds)
    num_sites = ['']*len(compounds)
    for comp_ind in range(len(compounds)):
        # collects mp IDs
        compounds_list[comp_ind] = compounds[comp_ind]['material_id']

        # collects best names
        pretty_formula = compounds[comp_ind]['pretty_formula']
        full_formula = compounds[comp_ind]['full_formula']
        element_list = compounds[comp_ind]['elements']
        if not not pretty_formula:
            best_name = pretty_formula
        elif not not full_formula:
            best_name = full_formula
        elif not not element_list:
            best_name = ', '.join(element_list)
        else:
            best_name = ''
        best_names[comp_ind] = best_name

        # collects number of sites
        num_sites[comp_ind] = compounds[comp_ind]['nsites']

    # saves all these lists
    save_directory = Path(save_directory)
    with open(save_directory / 'all_mp_IDs.txt', 'w') as f_id:
        f_id.writelines("%s\n" % mp_id for mp_id in compounds_list)

    with open(save_directory / 'all_mp_names.txt', 'w') as f_nm:
        f_nm.writelines("%s\n" % name for name in best_names)

    with open(save_directory / 'all_mp_nsites.txt', 'w') as f_ns:
        f_ns.writelines("%s\n" % nsites for nsites in num_sites)

    # downloads all the structures
    structs_directory = save_directory / 'mp_structs'
    save_all_struct_files = partial(save_struct_file, param, structs_directory)
    if not os.path.isdir(structs_directory):
        os.makedirs(structs_directory)
    with ThreadPoolExecutor() as executor:
        executor.map(save_all_struct_files, compounds_list)

    return


def name_from_cif(file_path):
    r"""Given the path to a cif file, returns the chemical formula of the structure in that file."""
    structure = IStructure.from_file(file_path)
    species_list = structure.species
    species_dict = {}
    for specie in species_list:
        if specie in species_dict:
            species_dict[specie] = species_dict[specie] + 1
        else:
            species_dict.update({specie: 1})
    # can get spaces between elements with property "formula" instead
    this_formula = Composition(species_dict).reduced_formula
    return this_formula


def load_all_MP_names(info_path="."):
    r"""Loads in all names, IDs, and nsites info generated by refresh_MP_data."""
    info_path = Path(info_path)
    material_IDs = np.genfromtxt(info_path / 'all_mp_IDs.txt', dtype='U25',
                                 autostrip=True)
    names_list = np.genfromtxt(
        info_path / 'all_mp_names.txt', dtype='U25', autostrip=True)
    nsites_list = np.genfromtxt(
        info_path / 'all_mp_nsites.txt', dtype='u4', autostrip=True)
    return material_IDs, names_list, nsites_list


def sphere_array(center=[0, 0, 0], radius=1, fineness=20):
    r"""Create an array used to plot wireframe spheres."""
    u = np.linspace(0, 2 * np.pi, fineness)
    v = np.linspace(0, np.pi, fineness)
    x = float(radius) * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = float(radius) * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = float(radius) * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    return x, y, z


def set_axes_equal(ax):
    r'''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.25*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_unit_cell(ax, lat_vecs, origin):
    r"""Plots the unit cell outline starting from origin with lattice vectors lat_vecs"""
    draw_mat = np.zeros([16, 3])
    draw_mat[0] = origin
    draw_mat[1] = draw_mat[0] + lat_vecs[0]
    draw_mat[2] = draw_mat[1] + lat_vecs[1]
    draw_mat[3] = draw_mat[2] - lat_vecs[0]
    draw_mat[4] = draw_mat[3] - lat_vecs[1]
    draw_mat[5] = draw_mat[4] + lat_vecs[2]
    draw_mat[6] = draw_mat[5] + lat_vecs[0]
    draw_mat[7] = draw_mat[6] - lat_vecs[2]
    draw_mat[8] = draw_mat[7] + lat_vecs[2]
    draw_mat[9] = draw_mat[8] + lat_vecs[1]
    draw_mat[10] = draw_mat[9] - lat_vecs[2]
    draw_mat[11] = draw_mat[10] + lat_vecs[2]
    draw_mat[12] = draw_mat[11] - lat_vecs[0]
    draw_mat[13] = draw_mat[12] - lat_vecs[2]
    draw_mat[14] = draw_mat[13] + lat_vecs[2]
    draw_mat[15] = draw_mat[14] - lat_vecs[1]
    ax.plot(draw_mat[:, 0], draw_mat[:, 1],
            draw_mat[:, 2], color='black', linewidth=0.25)


def plot_structure(param, structure, best_name, mat_ID, plot_save_path=None, specie=None,
                   plot_unit_cell_outline=True, num_cells_to_plot=[1, 1, 1], max_bond_dist=None,
                   fig=None):
    r"""
    Plots crystal structure in 3D.

    specie (str): Optional string that, when provided, selects just one sublattice to plot.
    plot_unit_cell_outline: True plots one unit cell. "all" plots all unit cells. False plots none.
    num_cells_to_plot (int list): A 3 element int list of the number of repeated unit cells to plot
        in the x, y, and z directions respectively.
    max_bond_dist (float): The longest bond length to plot. None if you don't want any bonds plotted
    """
    if not param.just_save_plots:
        plt.ion()
    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)
        plt.clf()
    ax = fig.add_subplot(111, projection='3d', position=[
                         0.05, 0.1, 0.75, 0.85])

    # get origin of each unit cell you will plot
    lat_vecs = structure.lattice.matrix
    norigins = num_cells_to_plot[0]*num_cells_to_plot[1]*num_cells_to_plot[2]
    origin_list = np.zeros([norigins, 3])
    orig_ind = 0
    for x_ind in range(num_cells_to_plot[0]):
        for y_ind in range(num_cells_to_plot[1]):
            for z_ind in range(num_cells_to_plot[2]):
                origin_list[orig_ind] = x_ind*lat_vecs[0] + \
                    y_ind*lat_vecs[1] + z_ind*lat_vecs[2]
                orig_ind = orig_ind + 1

    # for each unit cell that needs to be plot
    atom_list = []
    atom_list_coords = None
    for orig_ind in range(norigins):
        this_origin = origin_list[orig_ind]
        # plot unit cell outline once or for each
        if plot_unit_cell_outline is True and (this_origin == origin_list[0]).all():
            plot_unit_cell(ax, lat_vecs, this_origin)
        elif plot_unit_cell_outline == 'all':
            plot_unit_cell(ax, lat_vecs, this_origin)

        # plot atoms
        param.simple_graphics = True  # whether to plot atoms in 3D or not
        legend_elements = []
        elem_list = []
        plotted_sites_list = []
        for site_ind in range(len(structure.sites)):
            # only plot sites that are the desired specie, if given
            if (structure.sites[site_ind].species_string == specie) or (specie is None):
                # options are average_ionic_radius, atomic_radius, metallic_radius,
                # van_der_waals_radius, and atomic_radius_calculated
                rad = structure.sites[site_ind].specie.average_ionic_radius
                # only first atom in species listed in cases of fractional occupancy
                nm = list(structure.sites[site_ind].species.as_dict())[0]
                # color options are cpk_color, jmol_color, and molcas_gv_color
                colr = element(nm).jmol_color
                if nm not in elem_list:  # constructs custom element legend
                    elem_list.append(nm)
                    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=nm,
                                                  markerfacecolor=colr, markersize=12*rad))

                abc = structure.sites[site_ind].frac_coords
                perm_a, perm_b, perm_c = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1],
                                                     indexing='xy')
                # all equivalent sites within +/- 1 of given site
                all_abc = np.tile(abc, (27, 1)) + np.stack((perm_a.flatten(), perm_b.flatten(),
                                                            perm_c.flatten()), axis=1)

                for trans_ind in range(27):
                    this_abc = all_abc[trans_ind, :]
                    if (this_abc >= -0.01).all() and (this_abc <= 1.01).all():
                        xyz = this_origin + (this_abc[0]*lat_vecs[0] + this_abc[1]*lat_vecs[1]
                                             + this_abc[2]*lat_vecs[2])
                        atom_list.append(site_ind)
                        if atom_list_coords is None:
                            atom_list_coords = np.array(xyz, ndmin=2)
                        else:
                            atom_list_coords = np.vstack(
                                (atom_list_coords, xyz))

                        if param.simple_graphics:
                            ax.scatter(xyz[0], xyz[1], xyz[2],
                                       c=colr, s=100*rad)
                        else:
                            [x, y, z] = sphere_array(
                                center=xyz, radius=rad, fineness=10)
                            ax.plot_surface(x, y, z, color=colr,
                                            linewidth=0, antialiased=False)

    # now plot bonds
    if max_bond_dist is not None:
        site_pairs = list(itertools.combinations(atom_list_coords, r=2))
        for site_pair in site_pairs:
            this_bond_dist = np.sqrt((site_pair[0][0] - site_pair[1][0])**2
                                     + (site_pair[0][1] - site_pair[1][1])**2
                                     + (site_pair[0][2] - site_pair[1][2])**2)
            if this_bond_dist <= max_bond_dist:
                ax.plot([site_pair[0][0], site_pair[1][0]],
                        [site_pair[0][1], site_pair[1][1]],
                        [site_pair[0][2], site_pair[1][2]],
                        color=colr, linewidth=2)

    # plot legend
    ax.legend(handles=legend_elements,
              loc='center left', bbox_to_anchor=(1, 0.5))

    # Setting the axes properties
    # ax.set_aspect("equal") # this is broken currently (08/11/20)
    set_axes_equal(ax)
    ax.axis("off")

    # ax.set_xlabel('X (Å)')
    # ax.set_ylabel('Y (Å)')
    # ax.set_zlabel('Z (Å)')
    ax.set_title(best_name + ' (' + mat_ID + ')')

    if param.just_save_plots:
        plt.ioff()
    else:
        plt.ion()
        plt.show()

    if param.save_results and plot_save_path is not None:
        fig.savefig(plot_save_path /
                    (f"{mat_ID}_{best_name}_crystal_struct.png"))

    return fig


def plotting_bs_stuff(param, mat_ID, print_str):
    r"""Function that can plot BS, DOS, and 1BZ (if available from the Materials Project)."""
    bs = None
    dos = None
    with MPRester(param.MP_KEY, notify_db_version=False) as m:
        if param.plot_BS or param.plot_DOS_and_BS or param.plot_BZ:
            bs = m.get_bandstructure_by_material_id(mat_ID)
        if param.plot_DOS or param.plot_DOS_and_BS:
            dos = m.get_dos_by_material_id(mat_ID)

    # plots BS, if requested
    if (bs is not None) and param.plot_BS:
        # is the material a metal (i.e., the fermi level cross a band)
        if bs.is_metal():
            print_str = f"{print_str} - Should be a metal...\n"
        else:
            print_str = f"{print_str} - Probably non-metallic...\n"
            print_str = f"{print_str} - Bandgap: {bs.get_band_gap()['energy']} eV\n"

        # plot band structure and BZ
        plotter = BSPlotter(bs)
        plotter.get_plot().show()

        # plot BZ, if requested
        if param.plot_BZ:
            plotter.plot_brillouin()

    # plots DOS, if requested
    if (dos is not None) and param.plot_DOS:
        plotterDOS = DosPlotter()
        plotterDOS.add_dos("Total DOS", dos)
        plotterDOS.get_plot().show()

    # plot DOS and BS combined, if requested
    if (dos is not None and bs is not None) and param.plot_DOS_and_BS:
        # bs_projection can be 'elements' or None
        # dos_projection can be 'elements', 'orbitals' or None
        plotterBSDOS = plotterBSDOSter(
            bs_projection=None, dos_projection='orbitals')
        plotterBSDOS.get_plot(bs, dos=dos)
    return print_str


def make_cgd_string(mat_ID, species, comp_ind, fullhop_array, comp_with_FBs):
    r"""Turns the inputs (generated by do_TB_model) into a gcd file for Gavrog"""
    cgd_str = ""
    cgd_str = cgd_str + "PERIODIC_GRAPH\n"
    cgd_str = cgd_str + f"ID {mat_ID}_{species}_{comp_ind}\n"
    cgd_str = cgd_str + "EDGES\n"
    # write edge list
    n_edges = len(fullhop_array)
    for edge_ind in range(n_edges):
        this_edge = fullhop_array[edge_ind]
        if (this_edge[0] in comp_with_FBs) or (this_edge[1] in comp_with_FBs):
            cgd_str = (cgd_str
                       + f"  {int(this_edge[0])} "
                       + f"{int(this_edge[1])} {int(this_edge[2])} "
                       + f"{int(this_edge[3])} {int(this_edge[4])}\n")
    cgd_str = cgd_str + "END\n\n"
    return cgd_str


def parse_systre(fname):
    r"""Gets key, if it exists, in the Systre output saved to a file.

    If no key found (ie, in case of some error), returns None.
    If more than one key in file, returns last one found.

    Can call systre and save to a file with:
    java -cp Systre-19.6.0.jar org.gavrog.apps.systre.SystreCmdline foo.cgd --systreKey > bar.txt
    """
    systre_key = None
    dimension = 3
    systre_sg = None
    rcsr_nm = None
    fb_vec = ""
    collision_str = "!!! ERROR (STRUCTURE) - Structure has collisions between next-nearest neighbors."
    ladder_str = "!!! ERROR (STRUCTURE) - Structure is non-crystallographic (a 'ladder')."
    scd_str = "!!! ERROR (STRUCTURE) - Structure has second-order collisions."

    op_str_1 = "!!! ERROR (INTERNAL) - Unexpected java.lang.RuntimeException: Spacegroup finder messed up operators.\n"
    op_str_2 = "!!!       \tat org.gavrog.apps.systre.SystreCmdline.showSpaceGroup(SystreCmdline.java:525)\n"
    op_maybe = True
    inv_str_1 = "!!! ERROR (INTERNAL) - Unexpected java.lang.ArithmeticException: matrix has no inverse\n"
    inv_str_2 = "!!!       \tat org.gavrog.jane.compounds.Matrix.inverse(Matrix.java:748)\n"
    inv_maybe = True
    nul_str_1 = "!!! ERROR (INTERNAL) - Unexpected java.lang.NullPointerException: null\n"
    nul_str_2 = "!!!       \tat org.gavrog.jane.compounds.Matrix.setSubMatrix(Matrix.java:226)\n"
    nul_maybe = True

    try:
        with open(fname, "r") as f_systre:
            for line in f_systre:
                # what we want to see
                if line[:16] == '   Systre key: "':
                    systre_key = line[16:-2]
                if line[:18] == "      dimension = ":
                    dimension = int(line[18])
                if line.startswith("   Ideal space group is "):
                    systre_sg = line[24:-2]
                if line.startswith("       Name:		"):
                    rcsr_nm = line[14:-1]

                # things Systre can't handle
                if line[:80] == collision_str:
                    systre_key = "NN_COLLISION"
                if line[:71] == ladder_str:
                    systre_key = "LADDER"
                if line[:62] == scd_str:
                    systre_key = "SEC_ORD_COLLISION"

                # error in Systre 1
                if line == op_str_1:
                    op_maybe = True
                if line == op_str_2 and op_maybe:
                    systre_key = "OP_MESS"

                # error in Systre 2
                if line == inv_str_1:
                    inv_maybe = True
                if line == inv_str_2 and inv_maybe:
                    systre_key = "INV_ERR"

                # error in Systre 3
                if line == nul_str_1:
                    nul_maybe = True
                if line == nul_str_2 and nul_maybe:
                    systre_key = "NULL_ERR"
    except Exception:
        pass

    try:
        cgd_name = str(fname)[:-4] + ".cgd"
        with open(cgd_name, "r") as f_systre:
            start_recording = False
            for line in f_systre:
                if line.startswith("EDGES"):
                    start_recording = True
                elif start_recording:
                    if line.startswith("END"):
                        start_recording = False
                    else:
                        fb_vec = fb_vec + line.strip() + " \\newline "
    except Exception:
        fb_vec = ""
        raise
    # if systre_key is None:
    #    os.system(f"cat {fname}")
    return systre_key, dimension, systre_sg, rcsr_nm, fb_vec


def is_cyclic_hopping(hop_array, start_site=0, numCycles=5, verbose=False):
    r"""
    Checks if this hopping array is trivially localized.

    Trivially localized means that the hopping cannot extent to infinity,
    in other words, whether it is possible to take a path along hoppings out
    to arbitrary distances.

    The condition that is checked is actually whether there is any hopping path
    of numCycles hops or fewer that will take you from the start_site in the
    first unit cell to the same start_site in any other unit cell.

    This is not rigorously flawless, in the case of many sites in the first
    unit cell or pathological cases, but it is a compromise between ease of
    coding and perfection. Note that this code is not currently very efficient,
    but it is beyond my capabilities to make it more efficient.

    hop_array is an Mx5 matrix that describes the allowed hoppings, where M is
    the number of allowed hoppings. The first and second columns identify the
    two sites involved, and the last three columns describe the change in unit
    cell. The order of rows does not matter.
    For example, a hopping that connects atom 3 in one cell to atom 7 in
    the cell that is one forward in x and one back in y, but on the same z
    would have a row in hop_array that looks like
    (3 7 1 -1 0)
    """

    # initialize values
    connected_sites = np.array([start_site, 0, 0, 0], ndmin=2)
    if verbose:
        print("\nHopping list (site 1, site 2, dx, dy, dz):")
        print(hop_array)
        print(f"\nStarting site: {start_site}")
        print("\nInitially connected sites:")
        print(connected_sites)
    num_hops = np.shape(hop_array)[0]
    is_cyclic = False
    keep_going = True
    loop_num = 0

    # iterate hopping out from start_site
    while keep_going:
        if verbose:
            print(f"\nStarting cycle {loop_num+1} of {numCycles}")

        # hop out from each site included so far
        new_connected_sites = connected_sites
        for site_ind in range(np.shape(connected_sites)[0]):
            this_site = connected_sites[site_ind, 0]
            for hop_ind in range(num_hops):
                if hop_array[hop_ind, 0] == this_site:
                    dx = connected_sites[site_ind, 1] + hop_array[hop_ind, 2]
                    dy = connected_sites[site_ind, 2] + hop_array[hop_ind, 3]
                    dz = connected_sites[site_ind, 3] + hop_array[hop_ind, 4]
                    new_site = np.array(
                        [hop_array[hop_ind, 1], dx, dy, dz], ndmin=2)
                    new_connected_sites = np.append(
                        new_connected_sites, new_site, axis=0)
                if hop_array[hop_ind, 1] == this_site:
                    dx = connected_sites[site_ind, 1] - hop_array[hop_ind, 2]
                    dy = connected_sites[site_ind, 2] - hop_array[hop_ind, 3]
                    dz = connected_sites[site_ind, 3] - hop_array[hop_ind, 4]
                    new_site = np.array(
                        [hop_array[hop_ind, 0],  dx, dy, dz], ndmin=2)
                    new_connected_sites = np.append(
                        new_connected_sites, new_site, axis=0)

        # remove duplicate sites
        new_connected_sites = np.unique(new_connected_sites, axis=0)
        connected_sites = new_connected_sites
        if verbose:
            print("\nCurrently connected sites:")
            print(connected_sites)

        # check if any site is visited in more than one unit cell
        _, site_counts = np.unique(connected_sites[:, 0], return_counts=True)
        if np.any(site_counts > 1):
            is_cyclic = True
            if verbose:
                print("Found return to a site in a different unit cell, ending")

        # check whether to keep hopping out
        loop_num = loop_num + 1
        if is_cyclic or loop_num >= numCycles:
            keep_going = False

    return is_cyclic


def get_NN_dist(structure, specie=None):
    r"""
    Gets the nearest neighbor distance in a structure, in angstroms.

    If optional input "specie" is supplied (must be an element name string, eg. "Fe"), then the
    nearest neighbor distance on that sublattice is returned. Otherwise the smallest distance
    overall is returned.
    """
    lat_vecs = structure.lattice.matrix
    dist_mat = np.round(structure.distance_matrix, 5)

    # finds the sites that are of this specie
    sublat_ind_to_lat_ind = []
    for site_ind in range(len(structure.sites)):
        if structure.sites[site_ind].species_string == specie:
            sublat_ind_to_lat_ind.append(site_ind)

    # get nearest neighbor distance on this sublattice
    if specie is not None:
        this_dist_mat = dist_mat[sublat_ind_to_lat_ind, :]
        this_dist_mat = this_dist_mat[:, sublat_ind_to_lat_ind]
    else:
        this_dist_mat = dist_mat
    nn_dist = np.sort(np.unique(this_dist_mat))[1]

    for lat_vec_ind in range(3):
        if np.linalg.norm(lat_vecs[lat_vec_ind]) < nn_dist:
            nn_dist = np.linalg.norm(lat_vecs[lat_vec_ind])

    return nn_dist


# %% plot TB, if requested
def do_TB_model(param, structure, print_str, write_str, this_save_path, best_name, mat_ID):
    # define lattice vectors
    lat_vecs = structure.lattice.matrix

    # get list of unique site types (by element composition)
    specie_list = set()
    for site_ind in range(len(structure.sites)):
        specie_list.add(structure.sites[site_ind].species_string)
    specie_list = list(specie_list)
    n_species = len(specie_list)
    print_str = f"{print_str} - Detected {n_species} unique site types\n"

    # do TB for each sublattice
    dist_mat = np.round(structure.distance_matrix, 5)
    has_FBs = False
    gras = []
    n_FBs_list = []
    fullhop_arrays = []
    sublat_graphs = []
    k_vec_list = []
    evals_list = []
    comps_with_FBs_list = []
    for specie_ind in range(n_species):
        print_str = print_str + '\n'

        # create list of site fractional coordinates in this sublattice
        orb = None
        sublat_ind_to_lat_ind = []
        lat_ind_to_sublat_ind = [0]*len(structure.sites)
        for site_ind in range(len(structure.sites)):
            if structure.sites[site_ind].species_string == specie_list[specie_ind]:
                if orb is None:
                    orb = structure.sites[site_ind].frac_coords
                    lat_ind_to_sublat_ind[site_ind] = 0
                else:
                    orb = np.vstack(
                        (orb, structure.sites[site_ind].frac_coords))
                    lat_ind_to_sublat_ind[site_ind] = np.shape(orb)[0]-1
                sublat_ind_to_lat_ind.append(site_ind)

        n_this_species = len(sublat_ind_to_lat_ind)
        print_str = (f"{print_str} - Now considering the {specie_list[specie_ind]} sublattice"
                     f" (containing {n_this_species} sites)\n")

        # don't do TB for just a single atom sublattice
        if n_this_species > 1:
            # get nearest neighbor distance on this sublattice
            this_dist_mat = dist_mat[sublat_ind_to_lat_ind, :]
            this_dist_mat = this_dist_mat[:, sublat_ind_to_lat_ind]
            nn_dist = np.sort(np.unique(this_dist_mat))[1]

            for lat_vec_ind in range(3):
                if np.linalg.norm(lat_vecs[lat_vec_ind]) < nn_dist:
                    nn_dist = np.linalg.norm(lat_vecs[lat_vec_ind])

            print_str = f"{print_str}   - Nearest-neighbor distance is {np.round(nn_dist, 2)}\n"

            # decide on param.max_dist and param.decay_rate for this sublattice:
            if param.dist_units == 'AA':
                this_max_dist = param.max_dist
                # rate at which hopping decays in units of angstroms
                this_decay_rate = param.decay_rate
            else:
                this_max_dist = param.max_dist * nn_dist
                if param.decay_rate is not None:
                    this_decay_rate = param.decay_rate * nn_dist
                else:
                    this_decay_rate = param.decay_rate

            if param.decay_rate is not None:
                this_amplitude = np.exp(- nn_dist / (this_decay_rate))
            else:
                this_amplitude = 1.0

            # find the smallest neighbor distance in this sublattice NOT considered:
            # checks out to the longest bond distance allowed plus the shortest of
            # the three lattice constants a, b, c (times 1.05 for tolerance)
            diag_dist = 1.05 * np.amin([np.linalg.norm(lat_vecs[0]),
                                        np.linalg.norm(lat_vecs[1]), np.linalg.norm(lat_vecs[2])])
            NNN_neighs = structure.get_all_neighbors(
                r=this_max_dist+diag_dist, numerical_tol=1e-8,
                sites=[structure.sites[i] for i in sublat_ind_to_lat_ind])

            nnn_dist = this_max_dist + diag_dist + 1
            nn_dist_max = nn_dist
            for sublat_ind_i in range(n_this_species):
                lat_ind_i = sublat_ind_to_lat_ind[sublat_ind_i]
                if len(NNN_neighs[sublat_ind_i]) != 0:
                    for neigh_ind in range(len(NNN_neighs[sublat_ind_i])):
                        lat_ind_j = NNN_neighs[sublat_ind_i][neigh_ind].index
                        sublat_ind_j = lat_ind_to_sublat_ind[lat_ind_j]
                        if (lat_ind_j in sublat_ind_to_lat_ind):  # and (lat_ind_i <= lat_ind_j)
                            ind_R = NNN_neighs[sublat_ind_i][neigh_ind].image
                            this_dist = structure.get_distance(
                                lat_ind_i, lat_ind_j, ind_R)
                            if this_dist > nn_dist_max and this_dist < (this_max_dist + 1e-8):
                                nn_dist_max = this_dist
                            if this_dist > (this_max_dist + 1e-8) and this_dist < nnn_dist:
                                nnn_dist = this_dist

            print_str = f"{print_str}   - First non-considered neighbor distance is {np.round(nnn_dist, 2)}\n"

            # print near-neighbor distances being used, if requested
            if param.print_NN_dists:
                neighs = structure.get_all_neighbors(
                    r=this_max_dist, numerical_tol=1e-8,
                    sites=[structure.sites[i] for i in sublat_ind_to_lat_ind])
                dist_list = []
                for sublat_ind_i in range(n_this_species):
                    lat_ind_i = sublat_ind_to_lat_ind[sublat_ind_i]
                    if False:
                        print(f"\nthis site sublattice ind: {sublat_ind_i}")
                        print(f"this site lattice ind: {lat_ind_i}\n")
                    if len(neighs[sublat_ind_i]) != 0:
                        for neigh_ind in range(len(neighs[sublat_ind_i])):
                            lat_ind_j = neighs[sublat_ind_i][neigh_ind].index
                            sublat_ind_j = lat_ind_to_sublat_ind[lat_ind_j]
                            if ((lat_ind_i <= lat_ind_j) and
                                    (lat_ind_j in sublat_ind_to_lat_ind)):
                                ind_R = neighs[sublat_ind_i][neigh_ind].image
                                this_dist = structure.get_distance(
                                    lat_ind_i, lat_ind_j, ind_R)
                                dist_list.append(this_dist)
                                if False:
                                    print(
                                        repr(neighs[sublat_ind_i][neigh_ind]))
                                    print(f"this neighbor ind: {neigh_ind}")
                                    print(
                                        f"this neighbor lattice ind: {lat_ind_j}")
                                    print(
                                        f"this neighbor sublattice ind: {sublat_ind_j}")
                                    print(f"this neighbor ind_R: {ind_R}")
                                    print(f"this neighbor dist: {this_dist}\n")
                                    input()

                dist_list = np.round(dist_list, 2)
                dist_list, dist_counts = np.unique(
                    dist_list, return_counts=True)
                for dist_ind in range(len(dist_list)):
                    print_str = (f"{print_str}   - {dist_counts[dist_ind]} neighbors at "
                                 f"distance {dist_list[dist_ind]}\n")

            # create TB model object
            gra = tb_model(3, 3, lat_vecs, orb)

            # create hopping parameters based on exponential distance
            neighs = structure.get_all_neighbors(
                r=this_max_dist, numerical_tol=1e-8,
                sites=[structure.sites[i] for i in sublat_ind_to_lat_ind])
            # graph of atom connections
            sublat_graph = np.zeros((n_this_species, n_this_species))
            fullhop_array = None  # list of all hoppings with cell-border-crossing information
            # whether atom has a hop out of the unit cell
            leaves_cell = [False] * n_this_species
            for sublat_ind_i in range(n_this_species):
                lat_ind_i = sublat_ind_to_lat_ind[sublat_ind_i]
                if len(neighs[sublat_ind_i]) != 0:
                    for neigh_ind in range(len(neighs[sublat_ind_i])):
                        lat_ind_j = neighs[sublat_ind_i][neigh_ind].index
                        sublat_ind_j = lat_ind_to_sublat_ind[lat_ind_j]
                        if (lat_ind_j in sublat_ind_to_lat_ind):  # and (lat_ind_i <= lat_ind_j)
                            ind_R = neighs[sublat_ind_i][neigh_ind].image
                            is_first = True  # bool for not including hopping pairs
                            if lat_ind_i > lat_ind_j:
                                is_first = False
                            if lat_ind_i == lat_ind_j:
                                if ind_R[0] < 0:
                                    is_first = False
                                if ind_R[0] == 0 and ind_R[1] < 0:
                                    is_first = False
                                if ind_R[0] == 0 and ind_R[1] == 0 and ind_R[2] < 0:
                                    is_first = False
                            if is_first:
                                this_dist = structure.get_distance(
                                    lat_ind_i, lat_ind_j, ind_R)
                                # print(str(sublat_ind_i)+', '+str(sublat_ind_j)+', '+str(ind_R))
                                if this_decay_rate is not None:
                                    gra.set_hop(-this_amplitude * np.exp(-this_dist/(this_decay_rate)),
                                                sublat_ind_i, sublat_ind_j, ind_R, 'add')
                                else:
                                    gra.set_hop(-this_amplitude,
                                                sublat_ind_i, sublat_ind_j, ind_R, 'add')

                                # constructs scipy sparse graph of atom connections in sublattice
                                sublat_graph[sublat_ind_i][sublat_ind_j] = 1
                                sublat_graph[sublat_ind_j][sublat_ind_i] = 1
                                if (ind_R[0]**2 + ind_R[1]**2 + ind_R[2]**2) > 0.001:
                                    leaves_cell[sublat_ind_i] = True
                                    leaves_cell[sublat_ind_j] = True

                                # builds list of hops
                                if fullhop_array is None:
                                    fullhop_array = np.array(
                                        [sublat_ind_i, sublat_ind_j,
                                            ind_R[0], ind_R[1], ind_R[2]],
                                        ndmin=2, dtype=int)
                                else:
                                    fullhop_array = np.append(
                                        fullhop_array,
                                        [[sublat_ind_i, sublat_ind_j, ind_R[0], ind_R[1],
                                          ind_R[2]]],
                                        axis=0)
            fullhop_arrays.append(fullhop_array.tolist())

            # counts number of atoms in components that do not exit unit cell
            sublat_graph = csr_matrix(sublat_graph)
            sublat_graphs.append((sublat_graph.toarray()).tolist())
            n_components, labels = connected_components(
                csgraph=sublat_graph, directed=False, return_labels=True)
            comps_leave_cell = np.unique(labels[leaves_cell])
            comps_no_leave_cell = np.setdiff1d(labels, comps_leave_cell)
            atoms_with_no_hops = 0
            for i_comp in range(len(comps_no_leave_cell)):
                atoms_with_no_hops = (atoms_with_no_hops
                                      + np.count_nonzero(labels == comps_no_leave_cell[i_comp]))
            # double checks whether the components that *do* leave the cell are *really*
            # not trivially localized
            fullhop_array = fullhop_array.astype(np.int32)  # makes all ints
            comps_list = []
            for i_comp in range(len(comps_leave_cell)):
                # gets sites in this component
                in_this_comp = np.flatnonzero(
                    labels == comps_leave_cell[i_comp])
                # finds which lines of the hopping array are from this component
                temp = np.isin(fullhop_array, in_this_comp)
                mask = np.logical_or(temp[:, 0], temp[:, 1])
                # filters to just hopping in this component
                thishop_array = fullhop_array[mask, :]
                if not is_cyclic_hopping(thishop_array, start_site=in_this_comp[0],
                                         numCycles=param.check_to_N_hops):
                    atoms_with_no_hops = atoms_with_no_hops + len(in_this_comp)
                else:
                    comps_list.append(in_this_comp.tolist())
            # reports number of trivially localized sites
            if atoms_with_no_hops != 0:
                print_str = (f"{print_str}   - {atoms_with_no_hops} sites with only localized "
                             f"hopping\n")

            # solve model on a high symm. path in k-space
            if mat_ID == "mp-773163":
                kpath = KPathLatimerMunro(structure)
            else:
                kpath = KPathSetyawanCurtarolo(structure)
            # kpath._kpath["kpoints"]
            # kpath = HighSymmKpath(structure)
            k_vec = kpath.get_kpoints(100, False)[0]
            k_lab = kpath.get_kpoints(100, False)[1]
            # (k_vec,k_dist,k_node)=gra.k_path(k, 100)
            evals = gra.solve_all(k_vec)
            evals_list.append(evals.tolist())
            save_k_vec = k_vec
            for vec_ind in range(len(k_vec)):
                save_k_vec[vec_ind] = k_vec[vec_ind].tolist()
            k_vec_list.append(save_k_vec)
            n_pts = len(k_vec)

            # create figure
            if param.plot_TB:
                plt.rcParams["font.family"] = "serif"
                fig = plt.figure(constrained_layout=True)
                gs = GridSpec(1, 4, figure=fig)

                # get DOS, if requested
                if param.plot_TBDOS:
                    grid_spacing = [16, 16, 16]
                    k_point_mesh = gra.k_uniform_mesh(grid_spacing)

                    energy_array = gra.solve_all(k_point_mesh)
                    n_k_pts = grid_spacing[0] * \
                        grid_spacing[1] * grid_spacing[2]
                    Ehist, Ebins = np.histogram(energy_array, bins=100)

                    # plot results
                    axDOS = fig.add_subplot(gs[:, -1])
                    Evals = (Ebins[:-1] + Ebins[1:]) / 2
                    axDOS.fill_betweenx(
                        Evals, 0, Ehist, edgecolor='k', facecolor='#929591')
                    axDOS.set_xlabel('DOS')
                    Erange = np.ptp(Evals)
                    axDOS.set_ylim(np.min(Evals) - Erange * 0.03,
                                   np.max(Evals) + Erange * 0.03)
                    axDOS.set_xlim(0, np.max(Ehist))

                # plot bandstructure
                if param.plot_TBDOS:
                    ax = fig.add_subplot(gs[:, :-1], sharey=axDOS)
                else:
                    ax = fig.add_subplot(gs[:, :])
                for plot_ind in range(np.shape(evals)[0]):
                    ax.plot(np.arange(0, n_pts), evals[plot_ind, :])
                ax.set_xlabel('Momentum')
                ax.set_ylabel('E (arb. units)')
                fig.suptitle(f"T.B. for {specie_list[specie_ind]} sublattice of {best_name} "
                             f"({mat_ID})")
                # fig.set_tight_layout(True)

                # label x-axis
                tick_list = []
                tick_labels = []
                for tick_ind in range(n_pts):
                    if '\\' in k_lab[tick_ind]:
                        # make latex friendly
                        k_lab[tick_ind] = f"${k_lab[tick_ind]}$"
                for tick_ind in range(n_pts):
                    if bool(k_lab[tick_ind]):
                        if (tick_ind == 0) or (tick_ind == n_pts - 1):
                            tick_list.append(tick_ind)
                            tick_labels.append(k_lab[tick_ind])
                        elif bool(k_lab[tick_ind - 1]):
                            tick_list.append((2 * tick_ind - 1) / 2)
                            if k_lab[tick_ind] == k_lab[tick_ind - 1]:
                                tick_labels.append(k_lab[tick_ind])
                            else:
                                tick_labels.append(
                                    f"{k_lab[tick_ind - 1]}|{k_lab[tick_ind]}")
                ax.set_xticks(tick_list)
                ax.set_xticklabels(tick_labels)
                ax.set_xlim(tick_list[0], tick_list[-1])
                for tick_ind in range(len(tick_list)):
                    plt.axvline(x=tick_list[tick_ind], linewidth=1, color='k')

            # display TB model parameters, if requested
            if param.print_TB_params and param.verbose:
                gra.display()

            # check for flat band in TB, if requested
            n_FBs = 0
            comps_with_FBs = []
            fb_str = ""
            if param.check_TB_flat:
                for comp_ind in range(len(comps_list)):
                    # remove all except for this component
                    rmv_list = []
                    for rmv_ind in range(n_this_species):
                        if rmv_ind not in comps_list[comp_ind]:
                            rmv_list.append(rmv_ind)
                    this_gra = gra.remove_orb(rmv_list)
                    if param.high_symm_only:
                        energy_array = this_gra.solve_all(k_vec)
                        n_k_pts = n_pts
                    else:
                        grid_spacing = [16, 16, 16]
                        n_k_pts = grid_spacing[0] * \
                            grid_spacing[1] * grid_spacing[2]
                        k_point_mesh = this_gra.k_uniform_mesh(grid_spacing)
                        # k_point_mesh = np.random.rand(n_k_pts, 3)
                        energy_array = this_gra.solve_all(k_point_mesh)

                    # determine number of nontrivial flat bands:
                    energy_array = np.round(energy_array, 12)
                    hist, bins = np.histogram(
                        energy_array, bins=int(np.ceil(1 / param.max_fb_width)))
                    if param.full_flat:
                        check_inds = np.nonzero(
                            np.floor(hist / (n_k_pts * param.flatness_tol)))[0]
                        new_FBs = 0
                        if len(check_inds) > 0:
                            for check_ind in check_inds:
                                new_FBs = new_FBs + np.amin(np.count_nonzero(
                                    np.logical_and(energy_array >= bins[check_ind],
                                                   energy_array <= bins[check_ind+1]), axis=0))
                    else:
                        new_FBs = int(
                            np.sum(np.floor(hist / (n_k_pts * param.flatness_tol))))
                    n_FBs = n_FBs + new_FBs
                    if new_FBs > 0:
                        comps_with_FBs.append(
                            [i for i in comps_list[comp_ind]])
                        fb_str = fb_str + \
                            f"{new_FBs}/{len(comps_list[comp_ind])} "

                        # create systre input file, run systre, get key from systre
                        if param.save_cgd:
                            component_num = len(comps_with_FBs)-1
                            cgd_name = f"{mat_ID}_{specie_list[specie_ind]}_{component_num}.cgd"
                            with open(this_save_path / cgd_name, 'w') as f_cgd:
                                write_cgd = make_cgd_string(mat_ID,
                                                            specie_list[specie_ind],
                                                            component_num,
                                                            fullhop_array,
                                                            comps_with_FBs[component_num])
                                f_cgd.write(write_cgd)
                            out_name = f"{mat_ID}_{specie_list[specie_ind]}_{component_num}.out"
                            systre_path = param.script_path/"Systre-exp-fix2.jar"
                            systre_cmd = ("timeout 600 "
                                          + f"java -cp {systre_path} "
                                          + f"org.gavrog.apps.systre.SystreCmdline "
                                          + f"{this_save_path/cgd_name} --systreKey"
                                          + f"> {this_save_path/out_name}")
                            try:
                                os.system(systre_cmd)
                            except Exception:
                                pass

                n_FBs = int(n_FBs)
                comps_with_FBs_list.append(comps_with_FBs)
                print_str = (f"{print_str}   - Contains {n_FBs} (potentially) interesting "
                             f"flat bands\n")

            if param.just_save_plots:
                plt.ion()
                plt.ioff()
            if param.save_results and param.plot_TB:
                fig.savefig(this_save_path / (f"{n_FBs}_FBs_{mat_ID}_{best_name}_"
                                              f"{specie_list[specie_ind]}_sublat_TB.png"))

            n_FBs_list.append(n_FBs)
            write_str = (f"{write_str}{mat_ID}, {best_name}, {specie_list[specie_ind]}, "
                         f"{n_FBs}, {n_this_species}, {np.round(nn_dist, 5)}, {np.round(nn_dist_max, 5)}, "
                         f"{np.round(nnn_dist, 5)}, {fb_str}\n")
            gras.append(gra)

        else:
            print_str = f"{print_str}   - Only contains one atom per unit cell, skipping...\n"
            gras.append(None)
            fullhop_arrays.append(None)
            sublat_graphs.append(None)
            evals_list.append(None)
            k_vec_list.append(None)
            n_FBs_list.append(0)
            comps_with_FBs_list.append([])
            write_str = (f"{write_str}{mat_ID}, {best_name}, {specie_list[specie_ind]}, "
                         f"0, 1, 0, 0, 0, na\n")

    return (print_str, write_str, gras, n_FBs_list, specie_list, fullhop_arrays, sublat_graphs,
            k_vec_list, evals_list, comps_with_FBs_list)


# %% calls all the calculations for a given material
def do_mat_calcs(param, matInd):
    # initialization stuff
    mat_ID = param.material_IDs[matInd]
    this_start_time = perf_counter()
    num_started.value = num_started.value + 1
    this_comp_num = num_started.value
    plt.close(fig='all')
    this_save_path = save_path / mat_ID
    write_str = ""
    ok_to_save = True

    if not os.path.isdir(this_save_path):
        os.makedirs(this_save_path)

    print_str = f"\nResults for material {mat_ID}:\n"
    print_str = f"{print_str} - Compound:   {int(this_comp_num)} of {num_mats}\n"
    # do all the desired calculations
    try:
        # print material chemical formula
        best_name = param.names_list[matInd]
        nsites = param.nsites_list[matInd]
        print_str = f"{print_str} - Chemical formula:   {best_name}\n"

        # load chemical structure
        structure = load_structure(param, mat_ID)

        # plot 3D structure, if requested
        if param.plot_structure_3D:
            plot_structure(param, structure, best_name, mat_ID,
                           plot_save_path=this_save_path)

        # print structure data, if requested
        if param.print_structure_data:
            print_str = f"{print_str}\n{repr(structure)}\n"

        # plotting of BS, DOS, and 1BZ
        if param.plot_BS or param.plot_DOS or param.plot_DOS_and_BS or param.plot_BZ:
            print_str = plotting_bs_stuff(param, mat_ID, print_str)

        # plot TB, if requested
        if param.plot_TB or param.check_TB_flat:
            specie_list = None
            (print_str, write_str, gras, n_FBs_list, specie_list, fullhop_arrays, sublat_graphs,
             k_vec_list, evals_list, comps_with_FBs_list) = do_TB_model(
                param, structure, print_str, write_str, this_save_path, best_name, mat_ID)
            if specie_list is None:
                ok_to_save = False

    # keeps going if there is an error accessing the database for a material
    except KeyboardInterrupt:
        lock.acquire()
        print("\nStopping at user request...")
        ok_to_save = False
        lock.release()
        raise
    except Exception:
        lock.acquire()
        print_str = (f"{print_str} - There was an error retrieving data on this compound (see "
                     f"above error message)!!\n")
        error_msg = traceback.format_exc()
        write_str = f"ERROR - {mat_ID}\n" + str(error_msg)
        print(error_msg)
        ok_to_save = False
        lock.release()

    lock.acquire()
    if param.verbose:
        print(print_str)
    if param.check_TB_flat and param.save_results:
        f_txt.write(write_str)
        f_txt.flush()

    # save calculated results to json file.
    if param.save_results:
        if (param.plot_TB or param.check_TB_flat) and ok_to_save:
            with open(this_save_path / (f"{mat_ID}_{best_name}_results.json"), 'w') as f_json:
                save_dict = {"best_name": best_name,
                             "nsites": int(nsites),
                             "mat_ID": mat_ID,
                             "write_str": write_str,
                             "print_str": print_str,
                             "n_FBs_list": n_FBs_list,
                             "specie_list": specie_list,
                             "structure": structure.as_dict(),
                             "fullhop_arrays": fullhop_arrays,
                             "sublat_graphs": sublat_graphs,
                             # "k_vec_list": k_vec_list,
                             # "evals_list": evals_list,
                             "comps_with_FBs_list": comps_with_FBs_list
                             }
                json.dump(save_dict, f_json, indent=4)
            with open(this_save_path / (f"{mat_ID}_{best_name}_tbmodel.pickle"), 'wb') as f_pickle:
                pickle.dump(gras, f_pickle)
        elif ok_to_save:
            with open(this_save_path / (f"{mat_ID}_{best_name}_results.json"), 'w') as f_json:
                save_dict = {"best_name": best_name,
                             "nsites": int(nsites),
                             "mat_ID": mat_ID,
                             "write_str": write_str,
                             "print_str": print_str,
                             "structure": structure.as_dict()
                             }
                json.dump(save_dict, f_json, indent=4)

    # gets time it took to do all the calculations for this material
    num_complete.value = num_complete.value + 1
    this_run_time = perf_counter() - this_start_time
    total_time = perf_counter() - start_time_all

    avg_time = total_time/(num_complete.value+1)  # in seconds
    if param.verbose:
        print(f"This material took {np.round(this_run_time,3)} s to run.\nAverage time per "
              f"material: {np.round(avg_time,3)} s ("
              f"{np.round((num_mats-num_complete.value-1) * avg_time / 3600, 3)} hrs est. left)\n")
    lock.release()
    return


def run_all_calcs(param=paramObj()):
    r"""Runs all calculations for all materials in param.material_nums given settings of param."""
    global num_started, save_path, num_mats, lock, names_list, nsites_list, f_txt
    global num_complete, start_time_all

    # make output directory
    try:
        time_str = datetime.now().strftime("%m%d%y_%H%M%S")
        save_path = param.script_path / (f"outputs/TB_{time_str}/")
        if not os.path.isdir(save_path) and param.save_results:
            os.makedirs(save_path)
    except Exception:
        print("Something went wrong with establishing the output directory or finding this"
              " script\'s directory")
        print("Make sure you are not running inside an interpreter, but on the command line!!!")
        print(traceback.format_exc())
        pass

    # get lists of materials info:
    num_mats = len(param.material_nums)

    # initialize variables
    plt.close(fig='all')
    # surpresses superfluous warnings that comes from pymatgen "KPathSetyawanCurtarolo" function
    warnings.filterwarnings('ignore', '.*magmom.*',)
    warnings.filterwarnings('ignore', '.*standard primitive!.*',)
    warnings.filterwarnings('ignore', '.*fractional co-ordinates.*',)

    # stuff for multiprocessing
    start_time_all = perf_counter()
    lock = multiprocessing.Lock()
    num_started = multiprocessing.Value('d', 0.0)
    num_complete = multiprocessing.Value('d', 0.0)
    do_mat_calcs_parr = partial(do_mat_calcs, param)

    # initialize save file for TB search results if requested
    if param.check_TB_flat and param.save_results:
        f_txt = init_file(param, param.script_path / "outputs/", time_str)

    # Do calculations (main loop)
    if param.run_parallel:
        num_procs = multiprocessing.cpu_count()
        print(f"Beginning calculations with {num_procs} processes!!!\n")
        with multiprocessing.Pool() as pool:  # processes=num_procs
            # , chunksize = np.max([np.floor(num_mats/num_procs),1]))
            pool.map(do_mat_calcs_parr, param.material_nums)
    else:
        for matInd in param.material_nums:
            do_mat_calcs(param, matInd)
    print(
        f"\nFinished calculations in {np.round((perf_counter()-start_time_all)/3600,3)} hrs.\n")

    # clean up
    if param.check_TB_flat and param.save_results:
        f_txt.close()
    return


def continue_calc(time_str, del_bad=False):
    r"""Continues an incomplete calculation if given a time string of the form MMDDYY_HHMMSS."""
    global num_started, save_path, num_mats, lock, names_list, nsites_list, f_txt
    global num_complete, start_time_all

    # make output directory
    script_path = Path(os.path.dirname(os.path.realpath(__file__)))
    try:
        param = pickle.load(
            open(script_path / f"outputs/{time_str}_TB_params.pickle", "rb"))["param"]
    except Exception:
        print("Couldn't find param pickle for this run.")
        print(traceback.format_exc())
        raise
    save_path = param.script_path / (f"outputs/TB_{time_str}/")
    print("\nChecking for materials with errors...")

    # delete folder for any materials with an error
    with open(param.script_path / (f"outputs/{time_str}_TB_search_results.txt"), "r") as f_t:
        for line in f_t:
            if line[:8] == "ERROR - ":
                bad_mat_ID = line[8:-1]
                print(f"{bad_mat_ID} had an error")
                if del_bad:
                    try:
                        shutil.rmtree(save_path/bad_mat_ID)
                    except Exception:
                        pass

    # check every material to see if there are any failed calculations
    output_vals = TBoutputs(
        script_path / f"outputs/{time_str}_TB_search_results.txt", rmv_nonflat=False)
    mat_names, mat_inds, mat_counts = np.unique(
        output_vals.mp_id, return_counts=True, return_inverse=True)
    missing_mat_nums = []
    num_mats = len(param.material_nums)
    for mat_ind in range(len(param.material_IDs)):
        good_mat = True
        mat_ID = param.material_IDs[mat_ind]
        best_name = param.names_list[mat_ind]
        output_ind = np.nonzero(mat_names == mat_ID)[0]
        lat_inds = np.nonzero(mat_inds == output_ind)[0]
        if mat_ind % 10000 == 0:
            print(
                f"{np.round(100 * mat_ind / num_mats, 1)} % checked ({mat_ind} of {num_mats})")

        # bad if any species are missing in the calculation
        composition = Composition(best_name)
        specie_list = set()
        for spec_ind in range(len(composition.elements)):
            specie_list.add(composition.elements[spec_ind].symbol)
        calced_species = set(output_vals.specie_names[lat_inds])

        if not calced_species == specie_list:
            print(f"{mat_ID} missing {calced_species ^ specie_list} species")
            good_mat = False

        # bad if the json of calculation results was never completed
        if not os.path.exists(save_path/mat_ID/f"{mat_ID}_{best_name}_results.json"):
            print(f"{mat_ID} missing json")
            good_mat = False

        # bad if any species with flat bands is missing a cgd file
        if good_mat:
            for lat_ind in lat_inds:
                has_cgd = False
                this_species = output_vals.specie_names[lat_ind]
                f_start = f"{mat_ID}_{this_species}_"
                if output_vals.nFBs[lat_ind] > 0:
                    for fname in os.listdir(save_path/mat_ID):
                        if fname[-4:] == ".cgd" and fname.startswith(f_start):
                            has_cgd = True
                    if not has_cgd:
                        good_mat = False
                        print(
                            f"{mat_ID} missing cgd file for {this_species} sublattice")

        # if bad calculation, add to list of bad materials, remove results folder
        # and remove lines with this material from the results txt file
        if not good_mat:
            missing_mat_nums.append(mat_ind)
            if del_bad:
                if os.path.exists(save_path/mat_ID):
                    try:
                        shutil.rmtree(save_path/mat_ID)
                    except Exception:
                        pass

    # prepare to complete calculations
    param.material_nums = missing_mat_nums
    num_mats = len(param.material_nums)
    print(f"\n{num_mats} left to compute")

    if num_mats > 0:
        # initialize variables
        plt.close(fig='all')
        # surpresses superfluous warnings that comes from pymatgen "KPathSetyawanCurtarolo" function
        warnings.filterwarnings('ignore', '.*magmom.*',)
        warnings.filterwarnings('ignore', '.*standard primitive!.*',)
        warnings.filterwarnings('ignore', '.*fractional co-ordinates.*',)

        # stuff for multiprocessing
        start_time_all = perf_counter()
        lock = multiprocessing.Lock()
        num_started = multiprocessing.Value('d', 0.0)
        num_complete = multiprocessing.Value('d', 0.0)
        do_mat_calcs_parr = partial(do_mat_calcs, param)

        # initialize save file for TB search results if requested
        if param.check_TB_flat and param.save_results:
            f_txt = open(param.script_path / "outputs" /
                         (f"{time_str}_TB_search_results.txt"), "a", buffering=1)
        # Do calculations (main loop)
        num_procs = np.amin([multiprocessing.cpu_count(), num_mats])
        print(f"Beginning calculations with {num_procs} processes!!!\n")
        with multiprocessing.Pool(num_procs) as pool:  # processes=num_procs
            pool.map(do_mat_calcs_parr, param.material_nums)
        print(
            f"\nFinished calculations in {np.round((perf_counter()-start_time_all)/3600,3)} hrs.\n")

        # clean up
        if param.check_TB_flat and param.save_results:
            f_txt.close()

    # delete duplicate lines in results file:
    if del_bad:
        print("Removing duplicate lines from output text file...")
        with open(param.script_path / (f"outputs/{time_str}_TB_search_results.txt"), "r") as f_t:
            save_file_lines = f_t.readlines()
        n0 = len(save_file_lines)
        _, sort_inds = np.unique(save_file_lines, return_index=True)
        save_file_lines = [save_file_lines[i] for i in np.sort(sort_inds)]
        n1 = len(save_file_lines)
        with open(param.script_path / (f"outputs/{time_str}_TB_search_results.txt"), "w") as f_t:
            f_t.writelines(save_file_lines)
        print(f"Finished rewriting. {n1} of {n0} original lines retained.\n")
    return


def recalc_systre_keys(time_str):
    r"""Retries calculations of Systre keys from calculation with timestring MMDDYY_HHMMSS."""

    # make output directory
    script_path = Path(os.path.dirname(os.path.realpath(__file__)))
    try:
        param = pickle.load(
            open(script_path / f"outputs/{time_str}_TB_params.pickle", "rb"))["param"]
    except Exception:
        print("Couldn't find param pickle for this run.")
        print(traceback.format_exc())
        raise
    save_path = param.script_path / (f"outputs/TB_{time_str}/")
    systre_path = param.script_path / "Systre-exp-fix2.jar"
    num_mats = len(param.material_IDs)

    # get list of systre out files to parse through
    cmd_list = []
    file_list = os.listdir(save_path)
    num_recalced = 0
    print("Beginning systre recalculations")
    for mat_ind in range(num_mats):
        mat_ID = param.material_IDs[mat_ind]
        best_name = param.names_list[mat_ind]
        try:
            for fname in os.listdir(save_path / mat_ID):
                if fname[-4:] == ".cgd":
                    should_recalc = False
                    out_name = str(save_path / mat_ID / fname[:-4]) + ".out"
                    if not os.path.exists(out_name):
                        should_recalc = True
                    else:
                        systre_key, _, _, _, _ = parse_systre(
                            save_path / mat_ID / out_name)
                        if systre_key in ["LADDER", "OP_MESS", "INV_ERR", "NULL_ERR"]:
                            should_recalc = True
                    if should_recalc:
                        systre_cmd = ("timeout 600 "
                                      + f"java -cp {systre_path} "
                                      + f"org.gavrog.apps.systre.SystreCmdline "
                                      + f"{save_path/mat_ID/fname} --systreKey"
                                      + f"> {out_name}")
                        cmd_list.append(systre_cmd)
                        print(f"recalculating Systre key for {fname}...")
                        os.system(systre_cmd)
                        num_recalced += 1

        except Exception:
            print(traceback.format_exc())
            pass
            # with multiprocessing.Pool(processes=94) as pool:  # processes=num_procs
            #    pool.map(os.system, cmd_list)
    print(f"{num_recalced} materials Systre strings recalculated\n")
    return


def collect_systre_keys(time_str):
    r"""Extracts systre strings from calculation with timestring MMDDYY_HHMMSS."""

    # get output directory
    script_path = Path(os.path.dirname(os.path.realpath(__file__)))
    try:
        param = pickle.load(
            open(script_path / f"outputs/{time_str}_TB_params.pickle", "rb"))["param"]
    except Exception:
        print("Couldn't find param pickle for this run.")
        print(traceback.format_exc())
        raise
    save_path = param.script_path / (f"outputs/TB_{time_str}/")
    num_mats = len(param.material_IDs)

    # get list of systre out files to parse through
    systre_files = []
    mat_IDs = []
    specie_list = []
    comp_list = []
    file_list = os.listdir(save_path)
    print("Identifying systre output files...")
    for mat_ind in range(num_mats):
        mat_ID = param.material_IDs[mat_ind]
        best_name = param.names_list[mat_ind]
        if mat_ind % 10000 == 0:
            print(
                f"{np.round(100 * mat_ind / num_mats, 1)} % loaded ({mat_ind} of {num_mats})")
        try:
            for fname in os.listdir(save_path / mat_ID):
                if fname[-4:] == ".out":
                    systre_files.append(save_path / mat_ID / fname)
                    _inds = [i for i, ltr in enumerate(fname) if ltr == "_"]
                    mat_IDs.append(fname[:_inds[0]])
                    specie_list.append(fname[_inds[0]+1:_inds[1]])
                    comp_list.append(fname[_inds[1]+1:-4])
        except Exception:
            print(traceback.format_exc())
            pass
    num_FBs = len(systre_files)
    print(f"{num_FBs} systre output files found\n")

    # get systre results
    systre_keys = []
    dim_list = []
    systre_sg_list = []
    rcsr_nm_list = []
    print("Now parsing sytre output files...")
    for sys_ind, systre_file in enumerate(systre_files):
        if sys_ind % 10000 == 0:
            print(
                f"{np.round(100 * sys_ind / num_FBs, 1)} % complete ({sys_ind} of {num_FBs})")
        systre_key, dim, systre_sg, rcsr_nm, fb_vec = parse_systre(systre_file)
        systre_keys.append(systre_key)
        dim_list.append(dim)
        systre_sg_list.append(systre_sg)
        rcsr_nm_list.append(rcsr_nm)
    print("Finished parsing systre output files.\n")

    # save results
    print(f"Writing systre results to {time_str}_systre_keys.txt")
    with open(param.script_path / "outputs" / f"{time_str}_systre_keys.txt", "w") as f_sys:
        f_sys.write(
            "File containing the Systre keys (see Gavrog project) of all the flat\n")
        f_sys.write(
            f"-band models found in tight binding search {time_str}.\n\n")
        f_sys.write(
            "mat_ID, species, FB_comp, FB_dim, systre_key, spacegroup, rcsr_name\n")
        for sys_ind in range(num_FBs):
            write_str = (f"{mat_IDs[sys_ind]}, {specie_list[sys_ind]}, "
                         + f"{comp_list[sys_ind]}, {dim_list[sys_ind]}, "
                         + f"{systre_keys[sys_ind]}, {systre_sg_list[sys_ind]}, "
                         + f"{rcsr_nm_list[sys_ind]}, {fb_vec}\n")
            f_sys.write(write_str)
    print("Finished writing to file.\n")
    return


# do useful calculations
if __name__ == "__main__":
    # THIS ONLY NEEDS TO BE RUN ONCE TO SAVE THE MATERIALS FROM THE MATERIALS PROJECT
    param=paramObj()
    refresh_MP_data(param, '.')

    # THIS RUNS THE MAIN SEARCH FOR VARIOUS DIFFERENT PARAMETERS
    max_dists = [1.02, 1.05, 1.1, 1.2, 1.4, 1.02, 1.05, 1.1, 1.2, 1.4]
    decay_rates = [False, False, False, False, False, True, True, True, True, True]
    for run_ind in range(len(max_dists)):
        this_param=paramObj()#np.random.randint(0,120000,size=100)
        this_param.max_dist = max_dists[run_ind]
        this_param.decay_rate = decay_rates[run_ind]
        run_all_calcs(param=this_param)
        time.sleep(700)

    # CAN OPTIONALLY RUN BELOW TO FIX SEARCH RESULTS
    ts_list = ["063021_180536", "063021_210232", "070121_013146",
               "070121_080341", "070121_174549", "070221_053445",
               "070221_082906", "070221_124729", "070221_185354",
               "070321_034927"] # REPLACE THESE WITH WHATEVER FOLDERS WERE MADE BY THE ABOVE CODE
    for ts_val in ts_list:
        print(f"Now cleaning run {ts_val}")
        # check if any calcs are missing
        continue_calc(ts_val, del_bad=True)
        # check if any systre files could be fixed
        recalc_systre_keys(ts_val)
        # collect systre key values
        collect_systre_keys(ts_val)
