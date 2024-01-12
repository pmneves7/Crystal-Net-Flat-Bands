# -*- coding: utf-8 -*-
"""
Created on 02/18/21 by Paul Neves
"""

import numpy as np
from scipy.stats import mode
import pickle
import json
from time import perf_counter
import multiprocessing
from functools import partial
from pathlib import Path
from auto_TB_MP_v2 import *
from figures import *
import warnings
import traceback
import os
import shutil
import concurrent.futures
from pymatgen.core import IStructure
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter


def cgd_to_array(fname):
    """converts a cgd file with a single net to an Nx5 numpy array"""
    with open(fname, "r") as f_cgd:
        file_lines = f_cgd.readlines()

    start_ind = None
    for line_ind, line in enumerate(file_lines):
        if line.startswith("EDGES"):
            start_ind = line_ind + 1
        if line.startswith("END") and not start_ind is None:
            nrows = line_ind - start_ind

    return np.loadtxt(fname, dtype=int, skiprows=start_ind, max_rows=nrows)


def get_nsites(vector_rep):
    """Gets the number of nodes in the graph"""
    site_list = np.unique(np.append(vector_rep[:, 0], vector_rep[:, 1]))
    return len(site_list)


def get_nedges(vector_rep):
    """Gets the number of edges in the graph"""
    return np.shape(vector_rep)[0]


def get_td10_from_cgd(vector_rep):
    """Returns TD10 of a periodic graph

    Definition of TD10 from http://rcsr.anu.edu.au/help/td10.html:

    td10 is the cumulative sum of the first 10 shells of topological neighbors
    (i.e. the first 10 terms of the coordination sequence). For structures with
    more than one kind of vertex the value given is a weighted average over the
    vertices. Related nets usually have similar values of td10, so this is a
    good way to find related nets of the same coordination number. Note that,
    by convention, the coordination sequence starts with 1 for the reference
    vertex (shell 0) and this is included in the cumulative total given for
    each vertex.
    """
    # find list of unique sites
    site_list = np.unique(np.append(vector_rep[:, 0], vector_rep[:, 1]))
    nedges = get_nedges(vector_rep)

    # make coordination sequence for each site
    coord_seq_list = []
    for site_ind in site_list:
        this_coord_seq = [1]
        connected_sites = np.array([site_ind, 0, 0, 0], ndmin=2)
        last_neighbors = np.array([site_ind, 0, 0, 0], ndmin=2)

        # consider the first 10 shells of topological neighbors
        for shell_ind in range(10):
            this_coord_seq.append(0)
            new_connected_sites = None

            # for each site on the n-1th shell of neighbors, look for nth neighbors
            for sub_site_ind in range(np.shape(last_neighbors)[0]):

                # for each site, consider all hoppings from that site
                for hop_ind in range(nedges):
                    if vector_rep[hop_ind, 0] == last_neighbors[sub_site_ind, 0]:
                        dx = last_neighbors[sub_site_ind, 1] + vector_rep[hop_ind, 2]
                        dy = last_neighbors[sub_site_ind, 2] + vector_rep[hop_ind, 3]
                        dz = last_neighbors[sub_site_ind, 3] + vector_rep[hop_ind, 4]
                        new_site = np.array(
                            [vector_rep[hop_ind, 1], dx, dy, dz], ndmin=2
                        )
                        # if this is a newly visited site, add it
                        if not any(np.equal(connected_sites, new_site).all(1)):
                            if new_connected_sites is None:
                                new_connected_sites = new_site
                                this_coord_seq[-1] += 1
                            elif not any(
                                np.equal(new_connected_sites, new_site).all(1)
                            ):
                                new_connected_sites = np.append(
                                    new_connected_sites, new_site, axis=0
                                )
                                this_coord_seq[-1] += 1

                    if vector_rep[hop_ind, 1] == last_neighbors[sub_site_ind, 0]:
                        dx = last_neighbors[sub_site_ind, 1] - vector_rep[hop_ind, 2]
                        dy = last_neighbors[sub_site_ind, 2] - vector_rep[hop_ind, 3]
                        dz = last_neighbors[sub_site_ind, 3] - vector_rep[hop_ind, 4]
                        new_site = np.array(
                            [vector_rep[hop_ind, 0], dx, dy, dz], ndmin=2
                        )
                        # if this is a newly visited site, add it
                        if not any(np.equal(connected_sites, new_site).all(1)):
                            if new_connected_sites is None:
                                new_connected_sites = new_site
                                this_coord_seq[-1] += 1
                            elif not any(
                                np.equal(new_connected_sites, new_site).all(1)
                            ):
                                new_connected_sites = np.append(
                                    new_connected_sites, new_site, axis=0
                                )
                                this_coord_seq[-1] += 1

            connected_sites = np.append(connected_sites, new_connected_sites, axis=0)
            last_neighbors = new_connected_sites

        coord_seq_list.append(this_coord_seq)
        td10 = 0
        for coord_seq in coord_seq_list:
            td10 += sum(coord_seq)
        td10 = td10 / get_nsites(vector_rep)

    return td10, coord_seq_list


def get_bs_invariants(gra, keep_list, structure):
    """Returns invariants of a lattice from bandstructure

    Need to include the pythTB gra object and a list of included
    sites (ie, some subset of the gra sites that are connected
    within the component of the lattice of interest)

    keep_list is a list of integers of lattice sites to keep
    """

    max_fb_width = 0.001
    flatness_tol = 1.0

    # remove all except for this component
    rmv_list = []
    for rmv_ind in range(gra.get_num_orbitals()):
        if rmv_ind not in keep_list:
            rmv_list.append(rmv_ind)
    this_gra = gra.remove_orb(rmv_list)

    # get hopping parameter and check if uniform
    hopping_parameter = None
    uniform_hopping = True
    for hop_ind, hopping in enumerate(this_gra._hoppings):
        if hopping_parameter is None:
            hopping_parameter = hopping[0]
        else:
            if hopping_parameter - hopping[0] > 0.0001:
                uniform_hopping = False
        this_gra._hoppings[hop_ind][0] = hopping_parameter
    if not uniform_hopping:
        print("TB model not uniform hopping!")
    uniform_hopping = True

    # calculate bs at a bunch of random points
    n_k_pts = 4096
    k_point_mesh = np.random.rand(n_k_pts, 3)
    # add some high symmetry points for flavor
    try:
        kpath = KPathLatimerMunro(structure)
        k_vec = kpath.get_kpoints(100, False)[0]
        k_point_mesh = np.append(k_point_mesh, k_vec, axis=0)
    except Exception:
        pass
    energy_array = this_gra.solve_all(k_point_mesh)

    # find maximum and minimum energies
    max_E = np.amax(energy_array) / np.abs(hopping_parameter)
    min_E = np.amin(energy_array) / np.abs(hopping_parameter)

    # determine number of nontrivial flat bands:
    energy_array = np.round(energy_array, 12)
    hist, bins = np.histogram(energy_array, bins=int(np.ceil(1 / max_fb_width)))

    check_inds = np.nonzero(np.floor(hist / (n_k_pts * flatness_tol)))[0]
    FB_E_count = []
    if len(check_inds) > 0:
        for check_ind in check_inds:
            this_n_FBs = np.amin(
                np.count_nonzero(
                    np.logical_and(
                        energy_array >= bins[check_ind],
                        energy_array <= bins[check_ind + 1],
                    ),
                    axis=0,
                )
            )
            this_E = (bins[check_ind] + bins[check_ind + 1]) / 2
            this_E = this_E / np.abs(hopping_parameter)
            if this_n_FBs > 0:
                FB_E_count.append([this_E, this_n_FBs])

    return max_E, min_E, FB_E_count, uniform_hopping


def get_invariants(unique_FBs, param, mat_ind, do_systre=False):
    """Calculates all the invariants for a choice of a flat band sublattice"""
    # for displaying timing info at the end
    this_start_time = perf_counter()
    num_started.value = num_started.value + 1
    this_comp_num = num_started.value
    ts_list = [
        "063021_180536",
        "063021_210232",
        "070121_013146",
        "070121_080341",
        "070121_174549",
        "070221_053445",
        "070221_082906",
        "070221_124729",
        "070221_185354",
        "070321_034927",
    ]
    max_dists = [1.02, 1.05, 1.1, 1.2, 1.4, 1.02, 1.05, 1.1, 1.2, 1.4]
    decay_rates = [False, False, False, False, False, True, True, True, True, True]

    # figure out what folder this sublattice calc is in
    mp_id = unique_FBs.mp_id[mat_ind]
    chem_name = unique_FBs.chem_name[mat_ind]
    specie_names = unique_FBs.specie_names[mat_ind]
    if unique_FBs.p02_no_decay[mat_ind]:
        mat_path = Path("outputs") / f"TB_{ts_list[0]}" / mp_id
    elif unique_FBs.p05_no_decay[mat_ind]:
        mat_path = Path("outputs") / f"TB_{ts_list[1]}" / mp_id
    elif unique_FBs.p1_no_decay[mat_ind]:
        mat_path = Path("outputs") / f"TB_{ts_list[2]}" / mp_id
    elif unique_FBs.p2_no_decay[mat_ind]:
        mat_path = Path("outputs") / f"TB_{ts_list[3]}" / mp_id
    elif unique_FBs.p4_no_decay[mat_ind]:
        mat_path = Path("outputs") / f"TB_{ts_list[4]}" / mp_id
    elif unique_FBs.p02_decay[mat_ind]:
        mat_path = Path("outputs") / f"TB_{ts_list[5]}" / mp_id
    elif unique_FBs.p05_decay[mat_ind]:
        mat_path = Path("outputs") / f"TB_{ts_list[6]}" / mp_id
    elif unique_FBs.p1_decay[mat_ind]:
        mat_path = Path("outputs") / f"TB_{ts_list[7]}" / mp_id
    elif unique_FBs.p2_decay[mat_ind]:
        mat_path = Path("outputs") / f"TB_{ts_list[8]}" / mp_id
    elif unique_FBs.p4_decay[mat_ind]:
        mat_path = Path("outputs") / f"TB_{ts_list[9]}" / mp_id

    # find the species index and comps_with_FBs_list for this sublattice
    with open(mat_path / f"{mp_id}_{chem_name}_results.json", "rb") as f:
        results = json.load(f)

    for specie_ind in range(len(results["specie_list"])):
        if results["specie_list"][specie_ind] == unique_FBs.specie_names[mat_ind]:
            comps_with_FBs_list = results["comps_with_FBs_list"][specie_ind]
            break

    # get tb object for this sublattice
    with open(mat_path / f"{mp_id}_{chem_name}_tbmodel.pickle", "rb") as f:
        tbmodel = pickle.load(f)
    gra = tbmodel[specie_ind]

    # do calculations for this sublattice
    this_invariants_list = []
    for comp_ind in range(len(comps_with_FBs_list)):
        invariants = {}
        base_fname = f"{mp_id}_{unique_FBs.specie_names[mat_ind]}_{comp_ind}"
        systre_key, _, _, _, _ = parse_systre(mat_path / (base_fname + ".out"))

        if not do_systre:
            get_invs = False
            if systre_key is None:
                get_invs = True
            elif not systre_key[0].isdigit():
                get_invs = True
        else:
            get_invs = True
            if systre_key is None:
                get_invs = False
            elif not systre_key[0].isdigit():
                get_invs = False

        if get_invs:
            # basic information about sublattice
            invariants["mp_id"] = mp_id
            invariants["chem_name"] = chem_name
            invariants["specie_name"] = specie_names
            invariants["NNN_dists"] = unique_FBs.NNN_dists[mat_ind]
            invariants["comp"] = comp_ind

            # graph invariants
            vector_rep = cgd_to_array(mat_path / (base_fname + ".cgd"))
            invariants["nsites"] = get_nsites(vector_rep)
            invariants["nedges"] = get_nedges(vector_rep)
            td10, coord_seq_list = get_td10_from_cgd(vector_rep)
            invariants["td10"] = td10
            invariants["coord_seq_list"] = coord_seq_list
            if do_systre:
                invariants["systre_key"] = systre_key

            # bandstructure invariants
            structure = load_structure(param, mp_id)
            max_E, min_E, FB_E_count, uniform_hopping = get_bs_invariants(
                gra, comps_with_FBs_list[comp_ind], structure
            )
            invariants["max_E"] = max_E
            invariants["min_E"] = min_E
            invariants["FB_E_count"] = FB_E_count
            invariants["uniform_hopping"] = uniform_hopping

            # save invariants to list
            # lock.acquire()
            this_invariants_list.append(invariants)
            # lock.release()

    # print timing info
    num_complete.value = num_complete.value + 1
    this_run_time = perf_counter() - this_start_time
    total_time = perf_counter() - start_time_all

    avg_time = total_time / (num_complete.value + 1)  # in seconds
    lock.acquire()
    print(
        f"This sublattice took {np.round(this_run_time,3)} s to run.\nAverage time per "
        f"sublattice is {np.round(avg_time,3)} s ("
        f"{np.round((n_sublattices-num_complete.value-1) * avg_time / 3600, 3)} hrs est. left)\n"
    )
    lock.release()

    return this_invariants_list


def reduce_invariants(invariants_list):
    """Reduces invariants under the assumption that if they could be doubled unit cells based
    on the invariants alone, they are"""
    for list_ind, invariant_dict in enumerate(invariants_list):
        # start making gcd list
        gcd_list = [1] * (
            2
            + len(invariant_dict["FB_E_count"])
            + len(invariant_dict["coord_seq_list"])
        )
        gcd_list[0] = invariant_dict["nsites"]
        gcd_list[1] = invariant_dict["nedges"]

        # if a unit cell is doubled, you get twice as many of each coord_seq
        for seq_ind, coord_seq in enumerate(invariant_dict["coord_seq_list"]):
            gcd_list[2 + seq_ind] = invariant_dict["coord_seq_list"].count(coord_seq)

        # if a unit cell is doubled, you double the counts of all the flat bands
        for count_ind, FB_E_count in enumerate(invariant_dict["FB_E_count"]):
            gcd_list[2 + count_ind + invariant_dict["nsites"]] = FB_E_count[1]

        # get gcd
        gcd = np.gcd.reduce(gcd_list)

        # reduce invariant_dict if gcd > 1
        if gcd > 1:
            invariant_dict["nsites"] = int(invariant_dict["nsites"] / gcd)
            invariant_dict["nedges"] = int(invariant_dict["nedges"] / gcd)
            for count_ind in range(len(invariant_dict["FB_E_count"])):
                invariant_dict["FB_E_count"][count_ind][1] = int(
                    invariant_dict["FB_E_count"][count_ind][1] / gcd
                )
            unique_seq_lists = []
            new_coord_seq_list = []
            for coord_seq in invariant_dict["coord_seq_list"]:
                if coord_seq not in unique_seq_lists:
                    unique_seq_lists.append(coord_seq)
                    for _ in range(
                        int(invariant_dict["coord_seq_list"].count(coord_seq) / gcd)
                    ):
                        new_coord_seq_list.append(coord_seq)
            invariant_dict["coord_seq_list"] = new_coord_seq_list

        # save back to list of invariant_dicts
        invariants_list[list_ind] = invariant_dict

    return invariants_list


def find_unique(invariants_list, do_systre=False):
    """Finds the unique flat band nets given the set of net invariants"""
    test_ind = []
    unique_invariants_list = []
    counts = []
    examples = []
    if do_systre:
        n_wrong_systre = 0
        systre_count_dict = {}
    else:
        n_wrong_systre = None
    while len(invariants_list) > 0:
        this_invar_dict = invariants_list[0]
        unique_invariants_list.append(this_invar_dict)
        counts.append(1)
        examples.append([this_invar_dict])
        # list of all systre keys in this group
        if do_systre:
            this_systre_key_list = [this_invar_dict["systre_key"]]
        remove_inds = []
        for comp_ind, comp_invar_dict in enumerate(invariants_list[1:]):
            is_same = False
            # lattices with hopping decay must be treated seperately
            uniform_hopping = True
            if (
                this_invar_dict["uniform_hopping"]
                and comp_invar_dict["uniform_hopping"]
            ):
                uniform_hopping = False
            # check basic comparisons
            same_nsites = False
            if this_invar_dict["nsites"] == comp_invar_dict["nsites"]:
                same_nsites = True
            same_nedges = False
            if this_invar_dict["nedges"] == comp_invar_dict["nedges"]:
                same_nedges = True
            same_td10 = False
            if this_invar_dict["td10"] == comp_invar_dict["td10"]:
                same_td10 = True

            # check energy range
            if same_nsites and same_nedges and same_td10:
                E_range = min(
                    this_invar_dict["max_E"] - this_invar_dict["min_E"],
                    comp_invar_dict["max_E"] - comp_invar_dict["min_E"],
                )
                same_max_E = False
                if (
                    abs(this_invar_dict["max_E"] - comp_invar_dict["max_E"])
                    < E_range / 50
                ):
                    same_max_E = True
                same_min_E = False
                if (
                    abs(this_invar_dict["max_E"] - comp_invar_dict["max_E"])
                    < E_range / 50
                ):
                    same_min_E = True
                if not uniform_hopping:
                    same_max_E = True
                    same_min_E = True

                # check if share the same FB energies
                if same_max_E and same_min_E:
                    n_same_FBs = 0
                    tol_FBs = 0
                    for FB_E_count in this_invar_dict["FB_E_count"]:
                        tol_FBs += FB_E_count[1]
                        if uniform_hopping:
                            for comp_FB_E_count in comp_invar_dict["FB_E_count"]:
                                if (
                                    abs(FB_E_count[0] - comp_FB_E_count[0])
                                    < E_range / 50
                                    and FB_E_count[1] == comp_FB_E_count[1]
                                ):
                                    n_same_FBs += comp_FB_E_count[1]
                                    if this_invar_dict["mp_id"] == "mp-17128":
                                        if comp_invar_dict["mp_id"] in ["mp-560667"]:
                                            print(
                                                abs(FB_E_count[0] - comp_FB_E_count[0])
                                            )
                                            print(
                                                E_range / 100
                                                and FB_E_count[1] == comp_FB_E_count[1]
                                            )
                    if not uniform_hopping:
                        for comp_FB_E_count in comp_invar_dict["FB_E_count"]:
                            n_same_FBs += comp_FB_E_count[1]

                    # check if they share the same coord_seq_list
                    if n_same_FBs == tol_FBs:
                        unique_csls, csl_counts = np.unique(
                            this_invar_dict["coord_seq_list"], return_counts=True
                        )
                        comp_unique_csls, comp_csl_counts = np.unique(
                            this_invar_dict["coord_seq_list"], return_counts=True
                        )
                        n_same_csls = 0
                        for csl_ind, unique_csl in enumerate(unique_csls):
                            if unique_csl in comp_unique_csls:
                                same_ind = np.nonzero(unique_csl == comp_unique_csls)[0]
                                if csl_counts[csl_ind] == comp_csl_counts[same_ind]:
                                    n_same_csls += 1
                        if n_same_csls == len(csl_counts):
                            is_same = True

            # if the same, add to output list info
            if is_same:
                counts[-1] += 1
                examples[-1] = examples[-1] + [comp_invar_dict]
                remove_inds.append(comp_ind)
                if do_systre:
                    this_systre_key_list.append(comp_invar_dict["systre_key"])

        # delete invarian_dicts that have been accounted for
        if len(remove_inds) > 0:
            remove_inds.reverse()
            for remove_ind in remove_inds:
                del invariants_list[remove_ind + 1]
        del invariants_list[0]

        # get most common systre key in this group and add to list of misclassified systre
        if do_systre:
            this_systre_key_list, systre_counts = np.unique(
                this_systre_key_list, return_counts=True
            )
            most_common_systre = this_systre_key_list[np.argmax(systre_counts)]
            unique_invariants_list[-1]["systre_key"] = most_common_systre
            if most_common_systre in systre_count_dict:
                if np.max(systre_counts) > systre_count_dict[most_common_systre]:
                    n_wrong_systre = (
                        n_wrong_systre + systre_count_dict[most_common_systre]
                    )
                    systre_count_dict[most_common_systre] = np.max(systre_counts)
            else:
                systre_count_dict[most_common_systre] = np.max(systre_counts)
            n_wrong_systre = (
                n_wrong_systre + np.sum(systre_counts) - np.max(systre_counts)
            )

    return unique_invariants_list, counts, examples, n_wrong_systre


if __name__ == "__main__":
    recalc_invariants = True
    reclean_invariants = True
    consider_decay = False
    do_systre = False
    if do_systre:
        fname_end = "systre"
    else:
        fname_end = "collision"
    if consider_decay:
        file_suffix = "_all"
    else:
        file_suffix = "_unif"
    if recalc_invariants:
        # define starting variables
        global lock, start_time_all, num_started, num_complete, n_sublattices
        lock = multiprocessing.Lock()
        num_started = multiprocessing.Value("d", 0.0)
        num_complete = multiprocessing.Value("d", 0.0)
        param = paramObj()

        # import info of unique flat band sublattices
        with open(f"outputs/unique_results{file_suffix}.pickle", "rb") as f_unique:
            unique_FBs = pickle.load(f_unique)
            unique_FBs = unique_FBs["unique_FBs"]
        nmats = len(unique_FBs.systre_keys)

        # check each unique flat band sublattice
        mat_inds_to_calc = []
        n_comps = 0
        for mat_ind, key_list in enumerate(unique_FBs.systre_keys):
            first_bad = True
            for key in key_list:
                if not do_systre:
                    if not key[0].isdigit():
                        if first_bad:
                            mat_inds_to_calc.append(mat_ind)
                            first_bad = False
                        n_comps += 1
                else:
                    if key[0].isdigit():
                        if first_bad:
                            mat_inds_to_calc.append(mat_ind)
                            first_bad = False
                        n_comps += 1

        n_sublattices = len(mat_inds_to_calc)
        print(f"{n_sublattices} sublattices with {n_comps} components")

        # surpresses superfluous warnings that comes from pymatgen "KPathSetyawanCurtarolo" function
        warnings.filterwarnings(
            "ignore",
            ".*magmom.*",
        )
        warnings.filterwarnings(
            "ignore",
            ".*standard primitive!.*",
        )
        warnings.filterwarnings(
            "ignore",
            ".*fractional co-ordinates.*",
        )

        # get invariants in parallel
        start_time_all = perf_counter()
        get_invariants_parr = partial(
            get_invariants, unique_FBs, param, do_systre=do_systre
        )

        this_n_comps = 0
        for mat_ind in mat_inds_to_calc:
            for key in unique_FBs.systre_keys[mat_ind]:
                if not do_systre:
                    if not key[0].isdigit():
                        this_n_comps += 1
                else:
                    if key[0].isdigit():
                        this_n_comps += 1

        print(f"{this_n_comps-n_comps} <- IF 0 REPLACE this_n_comps")
        with multiprocessing.Pool(96) as p:
            results = p.map(get_invariants_parr, mat_inds_to_calc)
            p.close()
            p.join()
            invariants_list = []
            for result in results:
                invariants_list = invariants_list + result

        print(f"expected {this_n_comps} components, got {len(invariants_list)}")

        # save graph invariants
        with open(
            f"outputs/{fname_end}_graph_invariants{file_suffix}.pickle", "wb"
        ) as f_pickle:
            pickle.dump({"invariants_list": invariants_list}, f_pickle)
    else:
        # load graph invariants
        with open(
            f"outputs/{fname_end}_graph_invariants{file_suffix}.pickle", "rb"
        ) as f:
            invariants_list = pickle.load(f)["invariants_list"]

    # get grouped invariants
    if reclean_invariants:
        invariants_list = reduce_invariants(invariants_list)
        unique_invariants_list, counts, examples, n_wrong_systre = find_unique(
            invariants_list, do_systre=do_systre
        )
        with open(
            f"outputs/{fname_end}_graph_invariants_sorted{file_suffix}.pickle", "wb"
        ) as f_pickle:
            pickle.dump(
                {
                    "unique_invariants_list": unique_invariants_list,
                    "counts": counts,
                    "examples": examples,
                    "n_wrong_systre": n_wrong_systre,
                },
                f_pickle,
            )
    else:
        with open(
            f"outputs/{fname_end}_graph_invariants_sorted{file_suffix}.pickle", "rb"
        ) as f_pickle:
            temp = pickle.load(f_pickle)
            unique_invariants_list = temp["unique_invariants_list"]
            counts = temp["counts"]
            examples = temp["examples"]
            n_wrong_systre = temp["n_wrong_systre"]

    # print stats on these groups
    print(counts[1:30])
    print(
        f"{len(counts)} groups found from invariants in {np.sum(counts)} crystal nets"
    )
    print(f"mode of crystal net counts: {mode(counts)}")

    # stats on systre cross-validation
    if do_systre:
        print(f"{n_wrong_systre} components with systre key misclassified")
        systre_key_list = []
        for group_ind in range(len(unique_invariants_list)):
            systre_key_list.append(unique_invariants_list[group_ind]["systre_key"])
        n_dup_systre = len(systre_key_list) - len(np.unique(systre_key_list))
        sys_unique, sys_counts = np.unique(systre_key_list, return_counts=True)
        many_systre = sys_unique[np.argmax(sys_counts)]
        many_inds = []
        all_many_inds = []
        for key_ind in range(len(systre_key_list)):
            if systre_key_list[key_ind] == many_systre:
                many_inds.append(key_ind)
            if systre_key_list[key_ind] in sys_unique[np.nonzero(sys_counts > 1)[0]]:
                all_many_inds.append(key_ind)
        print(
            f"{n_dup_systre} groups with duplicate systre keys containing "
            f"{np.sum([counts[i] for i in all_many_inds])} nets"
        )
        print(f"{[counts[i] for i in all_many_inds]} components in these groups")

    # stats on number of groups
    sort_inds = np.argsort(counts)
    sort_inds = np.flip(sort_inds)
    counts = np.array(counts)

    print(f"Most common groups had the following counts: {counts[sort_inds[:30]]}\n")

    for group_ind in range(30):
        print(
            f"Group {group_ind+1} had {counts[sort_inds[group_ind]]} components in it. Invariants:"
        )
        print(unique_invariants_list[sort_inds[group_ind]])

        this_mp_id = unique_invariants_list[sort_inds[group_ind]]["mp_id"]
        this_specie_name = unique_invariants_list[sort_inds[group_ind]]["specie_name"]
        this_NNN_dists = unique_invariants_list[sort_inds[group_ind]]["NNN_dists"]
        # save structure with only one species
        structure = IStructure.from_file(f"./mp_structs/{this_mp_id}.cif")
        frac_coords_list = structure.frac_coords.tolist()
        specie_coord_list = []
        nsites = 0
        for specie_ind, specie in enumerate(structure.species):
            if str(specie) == this_specie_name:
                specie_coord_list.append(frac_coords_list[specie_ind])
                nsites += 1
        structure = Structure(structure.lattice, ["Co"] * nsites, specie_coord_list)
        # structure = structure.get_primitive_structure(tolerance=0.05)
        # structure = structure.get_reduced_structure('niggli')
        # CifWriter(structure).write_file(
        #     f"./common_lats/uncategorized/{group_ind}x{counts[sort_inds[group_ind]]}"
        #     f"_{this_mp_id}_{this_specie_name}_{this_NNN_dists}.cif")

    # make pie chart
    def my_autopct(pct):
        if pct > 2:
            pct_str = "{p:.1f}".format(p=pct)
        else:
            pct_str = ""
        return pct_str

    n_grps = 10
    grp_list = [str(i) for i in np.arange(1, n_grps + 1, 1)]
    grp_counts = []
    for group_ind in range(n_grps):
        grp_counts.append(counts[sort_inds[group_ind]])

    cs = cc.m_CET_R3(np.arange(n_grps + 1) / (n_grps + 1))
    grp_list.append("Other")
    grp_counts.append(sum(counts[sort_inds[n_grps:]]))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(grp_counts, labels=grp_list, autopct=my_autopct, pctdistance=0.8, colors=cs)
    ax.axis("equal")
    plt.savefig("coll_pie.png")
