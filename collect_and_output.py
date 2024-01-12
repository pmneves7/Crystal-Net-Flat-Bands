import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import colorcet as cc
import pickle
import copy
from pymatgen.core import IStructure
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.core.composition import Composition

# plt.style.use("paul_style.mplstyle")
from auto_TB_MP_v2 import *

plt.close(fig="all")


class mp_saved_info:
    def __init__(self, file_name="MP_info.txt"):
        r"""load general info about MP materials"""
        self.mp_ids = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=0, dtype=str
        )
        self.formulas = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=1, dtype=str
        )
        self.nmatsites = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=2, dtype=int
        )
        self.nelements = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=3, dtype=int
        )
        self.spacegroup_symbol = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=4, dtype=str
        )
        self.spacegroup_number = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=5, dtype=int
        )
        self.energy_per_atom = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=6, dtype=float
        )
        self.band_gap = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=7, dtype=float
        )
        self.has_bs = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=8, dtype=int
        )
        self.icsd_ids = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=9, dtype=int
        )


class gathered_results:
    def __init__(self, time_str, mp_info):
        r"""returns and pickles an object with lots of info about a data run"""
        # load info about FBs from a given calc run
        tbout = TBoutputs(f"outputs/{time_str}_TB_search_results.txt")
        self.mp_id = tbout.mp_id
        self.chem_name = tbout.chem_name
        self.specie_names = tbout.specie_names
        self.nFBs = tbout.nFBs
        self.nsites = tbout.nsites
        self.NN_dists = tbout.NN_dists
        self.NN_dists_max = tbout.NN_dists_max
        self.NNN_dists = tbout.NNN_dists
        self.nFB_lats = tbout.nFB_lats

        # load systre results
        file_name = f"outputs/{time_str}_systre_keys.txt"
        systre_ID = np.genfromtxt(
            file_name, delimiter=", ", skip_header=4, usecols=0, dtype=str
        )
        systre_FB_el = np.genfromtxt(
            file_name, delimiter=", ", skip_header=4, usecols=1, dtype=str
        )
        systre_FB_dim = np.genfromtxt(
            file_name, delimiter=", ", skip_header=4, usecols=3, dtype=int
        )
        systre_key = np.genfromtxt(
            file_name, delimiter=", ", skip_header=4, usecols=4, dtype=str
        )
        systre_sg = np.genfromtxt(
            file_name, delimiter=", ", skip_header=4, usecols=5, dtype=str
        )
        systre_rcsr = np.genfromtxt(
            file_name, delimiter=", ", skip_header=4, usecols=6, dtype=str
        )
        systre_vec = np.genfromtxt(
            file_name, delimiter=",", skip_header=4, usecols=7, dtype=str
        )

        # preallocate results variables
        self.nmatsites = np.zeros(self.nFB_lats)
        self.nelements = np.zeros(self.nFB_lats)
        self.spacegroup_symbol = [""] * self.nFB_lats
        self.spacegroup_number = np.zeros(self.nFB_lats)
        self.energy_per_atom = np.zeros(self.nFB_lats)
        self.band_gap = np.zeros(self.nFB_lats)
        self.has_bs = np.zeros(self.nFB_lats)
        self.icsd_ids = np.zeros(self.nFB_lats)
        self.FB_dims = [[]] * self.nFB_lats
        self.systre_keys = [[]] * self.nFB_lats
        self.systre_sg = [[]] * self.nFB_lats
        self.systre_rcsr = [[]] * self.nFB_lats
        self.systre_vec = [[]] * self.nFB_lats
        ind_list = []

        # look at each flat band lattice
        for FB_ind in range(self.nFB_lats):
            this_ind = np.nonzero(self.mp_id[FB_ind] == mp_info.mp_ids)[0]

            # find the corresponding mp information
            self.nmatsites[FB_ind] = mp_info.nmatsites[this_ind]
            self.nelements[FB_ind] = mp_info.nelements[this_ind]
            self.spacegroup_symbol[FB_ind] = mp_info.spacegroup_symbol[this_ind]
            self.spacegroup_number[FB_ind] = mp_info.spacegroup_number[this_ind]
            self.energy_per_atom[FB_ind] = mp_info.energy_per_atom[this_ind]
            self.band_gap[FB_ind] = mp_info.band_gap[this_ind]
            self.has_bs[FB_ind] = mp_info.band_gap[this_ind]
            self.icsd_ids[FB_ind] = mp_info.icsd_ids[this_ind]

            # find the corresponding systre keys
            these_inds = np.nonzero(
                np.logical_and(
                    self.mp_id[FB_ind] == systre_ID,
                    self.specie_names[FB_ind] == systre_FB_el,
                )
            )
            for this_ind in these_inds[0].tolist():
                self.FB_dims[FB_ind] = self.FB_dims[FB_ind] + [
                    int(systre_FB_dim[this_ind])
                ]
                self.systre_keys[FB_ind] = self.systre_keys[FB_ind] + [
                    systre_key[this_ind]
                ]
                self.systre_sg[FB_ind] = self.systre_sg[FB_ind] + [systre_sg[this_ind]]
                self.systre_rcsr[FB_ind] = self.systre_rcsr[FB_ind] + [
                    systre_rcsr[this_ind]
                ]
                self.systre_vec[FB_ind] = self.systre_vec[FB_ind] + [
                    systre_vec[this_ind]
                ]
                ind_list.append(this_ind)
        print(len(ind_list))
        print(len(systre_FB_el))
        print(systre_ID[np.setdiff1d(list(range(len(systre_FB_el))), ind_list)])
        print(np.setdiff1d(list(range(len(systre_FB_el))), ind_list))

        # pickle the results
        with open(f"outputs/{time_str}_TB_more_info.pickle", "wb") as f:
            pickle.dump({"gathered_results": self}, f)


if __name__ == "__main__":
    # get data you need to make the figures
    refresh_data = True
    consider_decay = False
    mp_info = mp_saved_info()
    if consider_decay:
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
        file_suffix = "_all"
    else:
        ts_list = [
            "063021_180536",
            "063021_210232",
            "070121_013146",
            "070121_080341",
            "070121_174549",
        ]
        max_dists = [1.02, 1.05, 1.1, 1.2, 1.4]
        decay_rates = [False, False, False, False, False]
        file_suffix = "_unif"
    data_objs = []
    for ts_ind, ts in enumerate(ts_list):
        if refresh_data:
            data_objs.append(gathered_results(ts, mp_info))
        else:
            with open(f"outputs/{ts}_TB_more_info.pickle", "rb") as f:
                data_objs.append(pickle.load(f)["gathered_results"])
                print(
                    f"max_dist {max_dists[ts_ind]} with decay "
                    + f"{decay_rates[ts_ind]} had {data_objs[ts_ind].nFB_lats} flat band lattices"
                )

    # find unique flatband lattices
    if refresh_data:
        unique_FBs = copy.deepcopy(data_objs[0])
        unique_FBs.flat_with_no_decay = [True] * len(unique_FBs.mp_id)
        unique_FBs.flat_with_decay = [False] * len(unique_FBs.mp_id)
        unique_FBs.max_dists = [1.02] * len(unique_FBs.mp_id)
        unique_FBs.p02_no_decay = [True] * len(unique_FBs.mp_id)
        unique_FBs.p05_no_decay = [False] * len(unique_FBs.mp_id)
        unique_FBs.p1_no_decay = [False] * len(unique_FBs.mp_id)
        unique_FBs.p2_no_decay = [False] * len(unique_FBs.mp_id)
        unique_FBs.p4_no_decay = [False] * len(unique_FBs.mp_id)
        if consider_decay:
            unique_FBs.p02_decay = [False] * len(unique_FBs.mp_id)
            unique_FBs.p05_decay = [False] * len(unique_FBs.mp_id)
            unique_FBs.p1_decay = [False] * len(unique_FBs.mp_id)
            unique_FBs.p2_decay = [False] * len(unique_FBs.mp_id)
            unique_FBs.p4_decay = [False] * len(unique_FBs.mp_id)
        for search_ind, search in enumerate(data_objs[1:]):
            for mat_ind, mat_id in enumerate(search.mp_id):
                spec_nm = search.specie_names[mat_ind]

                # look for other entries with the same ID, species, and NNN distance
                corr_ind = np.intersect1d(
                    np.nonzero(mat_id == unique_FBs.mp_id)[0],
                    np.nonzero(spec_nm == unique_FBs.specie_names)[0],
                )
                new_lat = True
                for test_ind in corr_ind:
                    if (
                        np.abs(
                            unique_FBs.NNN_dists[test_ind] - search.NNN_dists[mat_ind]
                        )
                        < 0.0001
                    ):
                        new_lat = False
                        # adds info on flat band hopping decay dependence
                        if decay_rates[search_ind + 1]:
                            unique_FBs.flat_with_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.02:
                                unique_FBs.p02_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.05:
                                unique_FBs.p05_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.1:
                                unique_FBs.p1_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.2:
                                unique_FBs.p2_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.4:
                                unique_FBs.p4_decay[test_ind] = True
                        else:
                            unique_FBs.flat_with_no_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.02:
                                unique_FBs.p02_no_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.05:
                                unique_FBs.p05_no_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.1:
                                unique_FBs.p1_no_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.2:
                                unique_FBs.p2_no_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.4:
                                unique_FBs.p4_no_decay[test_ind] = True

                # if it's new, add it to the unique_FBs object
                if new_lat:
                    unique_FBs.mp_id = np.append(
                        unique_FBs.mp_id, search.mp_id[mat_ind]
                    )
                    unique_FBs.chem_name = np.append(
                        unique_FBs.chem_name, search.chem_name[mat_ind]
                    )
                    unique_FBs.specie_names = np.append(
                        unique_FBs.specie_names, search.specie_names[mat_ind]
                    )
                    unique_FBs.nFBs = np.append(unique_FBs.nFBs, search.nFBs[mat_ind])
                    unique_FBs.nmatsites = np.append(
                        unique_FBs.nmatsites, search.nmatsites[mat_ind]
                    )
                    unique_FBs.nsites = np.append(
                        unique_FBs.nsites, search.nsites[mat_ind]
                    )
                    unique_FBs.NN_dists = np.append(
                        unique_FBs.NN_dists, search.NN_dists[mat_ind]
                    )
                    unique_FBs.NN_dists_max = np.append(
                        unique_FBs.NN_dists_max, search.NN_dists_max[mat_ind]
                    )
                    unique_FBs.NNN_dists = np.append(
                        unique_FBs.NNN_dists, search.NNN_dists[mat_ind]
                    )
                    unique_FBs.nelements = np.append(
                        unique_FBs.nelements, search.nelements[mat_ind]
                    )
                    unique_FBs.spacegroup_symbol.append(
                        search.spacegroup_symbol[mat_ind]
                    )
                    unique_FBs.spacegroup_number = np.append(
                        unique_FBs.spacegroup_number, search.spacegroup_number[mat_ind]
                    )
                    unique_FBs.energy_per_atom = np.append(
                        unique_FBs.energy_per_atom, search.energy_per_atom[mat_ind]
                    )
                    unique_FBs.band_gap = np.append(
                        unique_FBs.band_gap, search.band_gap[mat_ind]
                    )
                    unique_FBs.has_bs = np.append(
                        unique_FBs.has_bs, search.has_bs[mat_ind]
                    )
                    unique_FBs.icsd_ids = np.append(
                        unique_FBs.icsd_ids, search.icsd_ids[mat_ind]
                    )
                    unique_FBs.FB_dims.append(search.FB_dims[mat_ind])
                    unique_FBs.systre_keys.append(search.systre_keys[mat_ind])
                    unique_FBs.systre_sg.append(search.systre_sg[mat_ind])
                    unique_FBs.systre_rcsr.append(search.systre_rcsr[mat_ind])
                    unique_FBs.systre_vec.append(search.systre_vec[mat_ind])
                    unique_FBs.flat_with_decay.append(decay_rates[search_ind + 1])
                    unique_FBs.flat_with_no_decay.append(
                        not decay_rates[search_ind + 1]
                    )
                    unique_FBs.max_dists.append(max_dists[search_ind + 1])
                    unique_FBs.p02_no_decay.append(False)
                    unique_FBs.p05_no_decay.append(False)
                    unique_FBs.p1_no_decay.append(False)
                    unique_FBs.p2_no_decay.append(False)
                    unique_FBs.p4_no_decay.append(False)
                    if consider_decay:
                        unique_FBs.p02_decay.append(False)
                        unique_FBs.p05_decay.append(False)
                        unique_FBs.p1_decay.append(False)
                        unique_FBs.p2_decay.append(False)
                        unique_FBs.p4_decay.append(False)
                    if decay_rates[search_ind + 1]:
                        if max_dists[search_ind + 1] == 1.02:
                            unique_FBs.p02_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.05:
                            unique_FBs.p05_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.1:
                            unique_FBs.p1_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.2:
                            unique_FBs.p2_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.4:
                            unique_FBs.p4_decay[-1] = True
                    else:
                        if max_dists[search_ind + 1] == 1.02:
                            unique_FBs.p02_no_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.05:
                            unique_FBs.p05_no_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.1:
                            unique_FBs.p1_no_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.2:
                            unique_FBs.p2_no_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.4:
                            unique_FBs.p4_no_decay[-1] = True
        print(f"{len(unique_FBs.mp_id)} flat band sublattices found in mp")

        with open(f"outputs/unique_results{file_suffix}.pickle", "wb") as f:
            pickle.dump({"unique_FBs": unique_FBs}, f)
    else:
        with open(f"outputs/unique_results{file_suffix}.pickle", "rb") as f:
            unique_FBs = pickle.load(f)["unique_FBs"]
            print(
                f"loaded all {len(unique_FBs.mp_id)} unique flat band sublattices found in the MP"
            )
            print(f"")
            print(
                f"{len(np.unique(unique_FBs.mp_id))} "
                f"({np.round(100*len(np.unique(unique_FBs.mp_id))/len(unique_FBs.mp_id), 3)} %) "
                f"materials contain at least one flat band sublattice"
            )
            print(
                f"{len(mp_info.mp_ids)} materials in Materials Project with "
                f"{np.sum(mp_info.nelements)} elemental sublattices\n"
            )

    # find most common lattices
    # define common lattices:
    com_lat_keys = [
        "NN_COLLISION",
        "2 1 2 0 0 1 2 0 1 1 3 0 0 1 3 1 0 2 3 0 -1 2 3 1 0",
        "1 1 2 0 1 2 1 2 3 0",
        "3 1 2 0 0 0 1 2 0 0 1 1 3 0 0 0 1 3 0 1 0 1 4 0 0 0 1 4 1 0 0 2 3 0 0 -1 2 3 0 1 0 2 4 0 0 -1 2 4 1 0 0 3 4 0 0 0 3 4 1 -1 0",
        "2 1 2 0 0 1 2 0 1 1 3 0 0 1 3 1 0",
        "3 1 2 0 0 0 1 3 0 0 0 1 4 0 0 0 1 5 0 0 0 2 3 0 0 0 2 5 1 0 0 2 6 0 0 0 3 4 0 0 1 3 6 0 1 0 4 5 0 0 0 4 6 0 1 -1 5 6 -1 0 0",
        "3 1 2 0 0 0 1 2 0 0 1 1 2 0 1 0 1 2 1 0 0 2 3 0 0 0 2 3 0 0 1 2 3 0 1 0 2 3 1 0 0",
        "2 1 2 0 0 1 2 0 1 1 2 1 0 2 3 0 0",
        "2 1 2 0 0 1 3 0 0 1 4 0 0 2 3 0 0 2 5 0 0 3 6 0 0 4 5 0 1 4 6 1 0 5 6 1 -1",
        "2 1 2 0 0 1 2 0 1 1 2 1 0 2 3 0 0 2 3 0 1 2 3 1 0",
        "2 1 2 0 0 1 3 0 0 1 4 0 0 2 5 0 0 3 5 1 0 4 5 0 1",
        "1 1 2 0 1 3 0 1 4 0 2 3 0 2 4 1",
        "2 1 2 0 0 1 2 0 1 1 3 0 0 1 3 1 0 1 4 0 0 2 3 0 -1 2 3 1 0 2 4 0 -1 3 4 0 0",
        "2 1 2 0 0 1 3 0 0 1 3 1 0 1 4 0 0 2 3 0 0 2 4 0 0 2 4 0 1 3 4 0 0",
        "2 1 2 0 0 1 3 0 0 1 4 0 0 2 3 1 0 2 4 0 0 3 4 0 1",
        "1 1 2 0 1 3 0 1 4 0 2 5 0 4 5 1",
        "3 1 2 0 0 0 1 3 0 0 0 1 4 0 0 0 1 5 0 0 0 2 3 0 0 0 2 4 0 0 0 2 6 0 0 0 3 4 0 0 0 3 7 0 0 0 4 8 0 0 0 5 6 0 0 1 5 7 0 1 0 5 8 1 0 0 6 7 0 1 -1 6 8 1 0 -1 7 8 1 -1 0",
        "3 1 2 0 0 0 1 2 0 0 1 1 2 1 0 0 1 2 1 0 1 1 3 0 0 0 1 3 0 1 0",
        "3 1 2 0 0 0 1 2 0 1 0 1 3 0 0 0 1 3 1 0 0 1 4 0 0 0 1 5 0 0 0 2 3 0 -1 0 2 3 1 0 0 2 4 0 -1 0 2 5 0 -1 0 3 4 0 0 0 3 5 0 0 0 4 6 0 0 0 4 7 0 0 0 4 8 0 0 0 5 6 0 0 1 5 7 0 0 1 5 8 0 0 1 6 7 0 -1 0 6 7 0 0 0 6 8 -1 -1 0 6 8 0 0 0 7 8 -1 0 0 7 8 0 0 0",
        "3 1 2 0 0 0 1 3 0 0 0 1 4 0 0 0 2 5 0 0 0 2 6 0 0 0 3 4 0 0 0 3 7 0 0 0 4 8 0 0 0 5 6 0 0 0 5 9 0 0 0 6 10 0 0 0 7 10 1 0 0 7 11 0 0 0 8 9 0 1 0 8 12 0 0 0 9 12 0 -1 0 10 11 -1 0 0 11 12 0 0 1",
        "2 1 2 0 0 1 2 0 1 1 3 0 0 1 3 1 0 1 4 0 0 2 3 0 -1 2 3 1 0 2 4 0 -1 3 4 0 0 4 5 0 0 4 6 0 0 4 7 0 0 5 6 0 -1 5 6 0 0 5 7 -1 -1 5 7 0 0 6 7 -1 0 6 7 0 0",
        "2 1 2 0 0 1 3 0 0 1 3 1 0 1 4 0 0 2 3 0 0 2 4 0 1 3 4 -1 0",
        "1 1 2 0 1 3 0 1 4 0 2 4 1 2 5 0",
        "3 1 2 0 0 0 1 2 0 0 1 1 3 0 0 0 1 3 0 1 0 1 4 0 0 0 1 4 1 0 0 2 5 0 0 0 2 5 0 1 0 3 5 0 0 0 3 5 0 0 1 3 6 0 0 0 3 6 1 0 0 4 6 0 0 0 4 6 0 1 0 5 7 0 0 0 5 7 1 0 0 6 7 0 0 0 6 7 0 0 1",
        "2 1 2 0 0 1 2 0 1 1 2 1 0 1 2 1 1 2 3 0 0",
        "3 1 2 0 0 0 1 3 0 0 0 1 4 0 0 0 1 5 0 0 0 2 6 0 0 0 3 6 1 0 0 4 6 0 1 0 5 6 0 0 1",
        "3 1 2 0 0 0 1 3 0 0 0 1 4 0 0 0 1 5 0 0 0 1 6 0 0 0 1 7 0 0 0 2 3 0 0 0 2 4 0 0 0 2 7 1 0 0 2 8 0 0 0 2 9 0 0 0 3 4 0 0 0 3 8 0 1 0 3 10 0 0 0 3 11 0 0 0 4 5 0 0 1 4 10 -1 -1 1 4 12 0 0 0 5 6 0 0 0 5 7 0 0 0 5 10 -1 -1 0 5 12 0 0 -1 6 7 0 0 0 6 9 0 1 -1 6 11 0 0 0 6 12 0 1 -1 7 8 -1 0 0 7 9 -1 0 0 8 9 0 0 0 8 10 0 -1 0 8 11 0 -1 0 9 11 0 -1 1 9 12 0 0 0 10 11 0 0 0 10 12 1 1 -1 11 12 0 1 -1",
        "3 1 2 0 0 0 1 3 0 0 0 1 4 0 0 0 1 5 0 0 0 1 6 0 0 0 1 7 0 0 0 1 8 0 0 0 2 3 0 0 0 2 4 0 0 0 2 9 0 0 0 2 10 0 0 0 2 11 0 0 0 2 12 0 0 0 3 6 0 0 0 3 12 0 0 0 3 13 0 0 0 3 14 0 0 0 3 15 0 0 0 4 6 0 0 0 4 12 0 0 0 4 16 0 0 0 4 17 0 0 0 4 18 0 0 0 5 7 0 0 0 5 8 0 0 0 5 9 0 0 1 5 14 1 0 0 5 16 0 1 0 5 19 0 0 0 6 12 0 0 0 6 19 -1 -1 1 6 20 0 0 0 6 21 0 0 0 7 8 0 0 0 7 11 -1 0 1 7 13 0 0 1 7 17 0 1 0 7 22 0 0 0 8 10 0 0 1 8 15 1 0 0 8 18 0 0 0 8 23 0 0 0 9 10 0 0 0 9 11 0 0 0 9 14 1 0 -1 9 16 0 1 -1 9 24 0 0 0 10 11 0 0 0 10 15 1 0 -1 10 20 1 0 -1 10 23 0 0 -1 11 13 1 0 0 11 21 1 1 -1 11 22 1 0 -1 12 22 0 -1 0 12 23 0 0 -1 12 24 -1 -1 1 13 14 0 0 0 13 15 0 0 0 13 17 0 1 -1 13 21 0 1 -1 14 15 0 0 0 14 19 -1 0 0 14 24 -1 0 1 15 18 -1 0 0 15 20 0 0 0 16 17 0 0 0 16 18 0 0 0 16 19 0 -1 0 16 24 0 -1 1 17 18 0 0 0 17 21 0 0 0 17 22 0 -1 0 18 20 1 0 0 18 23 0 0 0 19 20 1 1 -1 19 21 1 1 -1 19 24 0 0 1 20 21 0 0 0 20 23 -1 0 0 21 22 0 -1 0 22 23 0 1 -1 22 24 -1 0 1 23 24 -1 -1 2",
        "3 1 2 0 0 0 1 2 0 0 1 1 2 0 1 0 1 3 0 0 0 1 3 1 0 0 1 4 0 0 0 1 4 1 0 0 1 5 0 0 0 1 5 1 0 0 2 3 0 0 0 2 3 1 0 0 2 4 0 -1 0 2 4 1 -1 0 2 5 0 0 -1 2 5 1 0 -1 3 4 0 -1 0 3 4 0 0 0 3 5 0 0 -1 3 5 0 0 0 4 5 0 0 0 4 5 0 1 -1",
        "2 1 2 0 0 1 3 0 0 1 4 0 0 1 5 0 0 2 5 1 0 3 4 0 1",
    ]
    # , "Extended Honeycomb", "Dual Stub", "Half Augmented Kagome", "Diamond Octagon", "hnb", "Extended Pyrochlore?"]
    com_lat_nms = [
        "Collision",
        "Kagome",
        "Stub",
        "Pyrochlore",
        "NNN Lieb",
        "lcv",
        "flu",
        "Stub Honeycomb",
        "hca",
        "Dice",
    ]
    if True:
        # get stats on number of nets
        all_stats = {"mp_id": [], "element": [], "NNN_dist": [], "FB_ids": []}
        FB_id_list = {}
        FB_id_num = 0
        systre_keys = {}
        sublat_lists = {}
        mp_id_lists = {}
        for sublist_ind, sublist in enumerate(unique_FBs.systre_keys):
            if unique_FBs.flat_with_no_decay[sublist_ind]:
                these_keys = []
                for key_ind, key in enumerate(sublist):
                    if key[0].isdigit():
                        these_keys.append(key)
                    if key not in systre_keys.keys():
                        systre_keys[key] = 1
                        sublat_lists[key] = [
                            unique_FBs.mp_id[sublist_ind]
                            + "_"
                            + unique_FBs.specie_names[sublist_ind]
                        ]
                        mp_id_lists[key] = [unique_FBs.mp_id[sublist_ind]]
                        # makes label for this systre key
                        if key[0].isdigit():
                            FB_id_list[key] = FB_id_num
                            FB_id_num = FB_id_num + 1
                    else:
                        systre_keys[key] = systre_keys[key] + 1
                        sublat_lists[key].append(
                            unique_FBs.mp_id[sublist_ind]
                            + "_"
                            + unique_FBs.specie_names[sublist_ind]
                        )
                        mp_id_lists[key].append(unique_FBs.mp_id[sublist_ind])
                # all_stats for later labelling
                all_stats["mp_id"].append(unique_FBs.mp_id[sublist_ind])
                all_stats["element"].append(unique_FBs.specie_names[sublist_ind])
                all_stats["NNN_dist"].append(unique_FBs.NNN_dists[sublist_ind])
                all_stats["FB_ids"].append([FB_id_list[i] for i in these_keys])
        comp_count_systre = np.asarray(list(systre_keys.values()))
        sublat_count_systre = np.array(
            [len(np.unique(sublat_lists[i])) for i in systre_keys.keys()]
        )
        mat_count_systre = np.array(
            [len(np.unique(mp_id_lists[i])) for i in systre_keys.keys()]
        )
        key_list = np.asarray(list(systre_keys.keys()))
        # count number not systre
        n_keys_not_systre = 0
        n_comps_not_systre = 0
        for key_ind, key in enumerate(key_list):
            if not key[0].isdigit():
                n_keys_not_systre = n_keys_not_systre + 1
                n_comps_not_systre = n_comps_not_systre + comp_count_systre[key_ind]

        sort_inds_systre = np.argsort(mat_count_systre)[::-1]
        print(key_list[sort_inds_systre[:31]])
        print(comp_count_systre[sort_inds_systre[:31]])
        print(
            f"\nSystre classified {len(systre_keys)-n_keys_not_systre} "
            f"unique lattice types from {np.sum(comp_count_systre)} FB lattice"
            f" components in {len(np.unique(unique_FBs.mp_id))} materials"
        )
        print(f"Systre could not classify {n_comps_not_systre} components\n")

        # save key list for posterity
        if True:
            with open(f"outputs/key_dictionary{file_suffix}.pickle", "wb") as f:
                pickle.dump(
                    {
                        "key_list": key_list[sort_inds_systre],
                        "key_ind": np.arange(len(key_list)),
                        "comp_count_systre": comp_count_systre[sort_inds_systre],
                        "sublat_lists": sublat_lists,
                        "mp_id_lists": mp_id_lists,
                    },
                    f,
                )

        # print examples of common FB nets
        folder_name = f"outputs/TB_{ts_list[2]}/"
        to_print = key_list[sort_inds_systre[:21]]
        with open(Path("common_lats") / "common_lats.cgd", "w") as f_cgd:
            for key_ind, key in enumerate(to_print):
                if key in com_lat_keys and key_ind < 10:
                    print(
                        f"{com_lat_nms[com_lat_keys.index(key)]} has "
                        f"{comp_count_systre[sort_inds_systre[key_ind]]} components "
                        f"in {sublat_count_systre[sort_inds_systre[key_ind]]} sublattices "
                        f"in {mat_count_systre[sort_inds_systre[key_ind]]} materials"
                    )
                if key[0].isdigit():
                    for this_ind, key_list in enumerate(data_objs[2].systre_keys):
                        if key == key_list[0]:
                            print(
                                f"{data_objs[2].mp_id[this_ind]}_"
                                f"{data_objs[2].specie_names[this_ind]}_"
                                f"{data_objs[2].NNN_dists[this_ind]}"
                            )
                            break
                    this_folder_name = folder_name + data_objs[2].mp_id[this_ind] + "/"
                    this_file_name = (
                        f"{data_objs[2].mp_id[this_ind]}_"
                        f"{data_objs[2].specie_names[this_ind]}_0.cgd"
                    )
                    # save structure with only one species
                    structure = IStructure.from_file(
                        f"./mp_structs/{data_objs[2].mp_id[this_ind]}.cif"
                    )
                    frac_coords_list = structure.frac_coords.tolist()
                    specie_coord_list = []
                    nsites = 0
                    for specie_ind, specie in enumerate(structure.species):
                        if str(specie) == data_objs[2].specie_names[this_ind]:
                            specie_coord_list.append(frac_coords_list[specie_ind])
                            nsites += 1
                    structure = Structure(
                        structure.lattice, ["Co"] * nsites, specie_coord_list
                    )
                    structure = structure.get_primitive_structure(tolerance=0.05)
                    structure = structure.get_reduced_structure("niggli")
                    CifWriter(structure).write_file(
                        f"./common_lats/{key_ind}x{mat_count_systre[sort_inds_systre[key_ind]]}_"
                        f"{data_objs[2].mp_id[this_ind]}"
                        f"_{data_objs[2].specie_names[this_ind]}_"
                        f"{data_objs[2].NNN_dists[this_ind]}.cif"
                    )
                    with open(this_folder_name + this_file_name, "r") as f:
                        cgd_str = f.read()
                        f_cgd.write(cgd_str)
        # now get statistics on non-systre nets from lattice invariants
        with open(
            f"outputs/collision_graph_invariants_sorted_unif.pickle", "rb"
        ) as f_pickle:
            temp = pickle.load(f_pickle)
            unique_invariants_list = temp["unique_invariants_list"]
            comp_count_coll = temp["counts"]
            examples = temp["examples"]
            n_wrong_systre = temp["n_wrong_systre"]
        sublat_count_coll = []
        mat_count_coll = []
        FB_id_num_systre_max = FB_id_num
        for example_ind, example in enumerate(examples):
            this_mp_id_list = []
            this_sublat_list = []
            for case_ind, case in enumerate(example):
                # makes label for this systre key
                FB_id_list[str(case_ind)] = FB_id_num
                FB_id_num = FB_id_num + 1
                if case["uniform_hopping"]:
                    this_mp_id_list.append(case["mp_id"])
                    this_sublat_list.append(case["mp_id"] + "_" + case["specie_name"])
                    # all_stats for later labelling
                    if refresh_data:
                        new_sublat = True
                        for stat_ind in range(len(all_stats["mp_id"])):
                            if (
                                all_stats["mp_id"][stat_ind] == case["mp_id"]
                                and all_stats["element"][stat_ind]
                                == case["specie_name"]
                                and all_stats["NNN_dist"][stat_ind] == case["NNN_dists"]
                            ):
                                all_stats["FB_ids"][stat_ind].append(FB_id_num)
                                new_sublat = False
                        if new_sublat:
                            all_stats["mp_id"].append(case["mp_id"])
                            all_stats["element"].append(case["specie_name"])
                            all_stats["NNN_dist"].append(case["NNN_dists"])
                            all_stats["FB_ids"].append([FB_id_num])
            sublat_count_coll.append(len(np.unique(this_sublat_list)))
            mat_count_coll.append(len(np.unique(this_mp_id_list)))
        sorted_inds_coll = np.flipud(np.argsort(mat_count_coll))
        if refresh_data:
            with open(f"outputs/FB_ids.pickle", "wb") as f:
                pickle.dump({"all_stats": all_stats, "FB_id_list": FB_id_list}, f)
        else:
            with open(f"outputs/FB_ids.pickle", "rb") as f:
                temp = pickle.load(f)
                all_stats = temp["all_stats"]
                FB_id_list = temp["FB_id_list"]

        for ind, i in enumerate(sorted_inds_coll[:20]):
            print(
                f"This invariant grouped lattice has {comp_count_coll[i]} "
                f"components in {sublat_count_coll[i]} sublattices in "
                f"{mat_count_coll[i]} materials. The invariants are:"
            )
            print(examples[i][0])
            print("\n")
            # save structure with only one species
            structure = IStructure.from_file(
                f"./mp_structs/{examples[i][0]['mp_id']}.cif"
            )
            frac_coords_list = structure.frac_coords.tolist()
            specie_coord_list = []
            nsites = 0
            for specie_ind, specie in enumerate(structure.species):
                if str(specie) == examples[i][0]["specie_name"]:
                    specie_coord_list.append(frac_coords_list[specie_ind])
                    nsites += 1
            structure = Structure(structure.lattice, ["Co"] * nsites, specie_coord_list)
            structure = structure.get_primitive_structure(tolerance=0.05)
            structure = structure.get_reduced_structure("niggli")
            CifWriter(structure).write_file(
                f"./common_lats/collision/{ind+1}x{mat_count_coll[i]}_"
                f"{examples[i][0]['mp_id']}_"
                f"{examples[i][0]['specie_name']}_"
                f"{examples[i][0]['NNN_dists']}.cif"
            )

        # print stats on s, p, d, f, dimension, etc
        s_count = 0
        p_count = 0
        d_count = 0
        f_count = 0
        systre_compat = 0
        systre_incompat = 0
        count1d = 0
        count2d = 0
        count3d = 0
        for sublat_ind in range(len(unique_FBs.specie_names)):
            if unique_FBs.specie_names[sublat_ind] in [
                "H",
                "He",
                "Li",
                "Be",
                "Na",
                "Mg",
                "K",
                "Ca",
                "Rb",
                "Sr",
                "Cs",
                "Ba",
                "Fr",
                "Ra",
            ]:
                s_count += 1
            if unique_FBs.specie_names[sublat_ind] in [
                "B",
                "C",
                "N",
                "O",
                "F",
                "Ne",
                "Al",
                "Si",
                "P",
                "S",
                "Cl",
                "Ar",
                "Ga",
                "Ge",
                "As",
                "Se",
                "Br",
                "Kr",
                "In",
                "Sn",
                "Sb",
                "Te",
                "I",
                "Xe",
                "Tl",
                "Pb",
                "Bi",
                "Po",
                "At",
                "Rn",
                "Nh",
                "Fl",
                "Mc",
                "Lv",
                "Ts",
                "Og",
            ]:
                p_count += 1
            if unique_FBs.specie_names[sublat_ind] in [
                "Sc",
                "Ti",
                "V",
                "Cr",
                "Mn",
                "Fe",
                "Co",
                "Ni",
                "Cu",
                "Zn",
                "Y",
                "Zr",
                "Nb",
                "Mo",
                "Tc",
                "Ru",
                "Rh",
                "Pd",
                "Ag",
                "Cd",
                "La",
                "Hf",
                "Ta",
                "W",
                "Re",
                "Os",
                "Ir",
                "Pt",
                "Au",
                "Hg",
                "Ac",
                "Rf",
                "Db",
                "Sg",
                "Bh",
                "Hs",
                "Mt",
                "Ds",
                "Rg",
                "Cn",
            ]:
                d_count += 1
            if unique_FBs.specie_names[sublat_ind] in [
                "Ce",
                "Pr",
                "Nd",
                "Pm",
                "Sm",
                "Eu",
                "Gd",
                "Tb",
                "Dy",
                "Ho",
                "Er",
                "Tm",
                "Yb",
                "Lu",
                "Th",
                "Pa",
                "U",
                "Np",
                "Pu",
                "Am",
                "Cm",
                "Bk",
                "Cf",
                "Es",
                "Fm",
                "Md",
                "No",
                "Lr",
            ]:
                f_count += 1
            all_systre = True
            for key in unique_FBs.systre_keys[sublat_ind]:
                if not key[0].isdigit():
                    all_systre = False
            if all_systre:
                systre_compat += 1
            else:
                systre_incompat += 1
            has1d = False
            has2d = False
            has3d = False
            for dim in unique_FBs.FB_dims[sublat_ind]:
                if dim == 1:
                    has1d = True
                if dim == 2:
                    has2d = True
                if dim == 3:
                    has3d = True
            if has1d:
                count1d += 1
            if has2d:
                count2d += 1
            if has3d:
                count3d += 1
        print(f"{s_count} sublattices with s block elements")
        print(f"{p_count} sublattices with p block elements")
        print(f"{d_count} sublattices with d block elements")
        print(f"{f_count} sublattices with f block elements")
        print(f"{systre_compat} sublattices completely compatible with systre")
        print(f"{systre_incompat} sublattices not completely compatible with systre")
        print(f"{count1d} sublattices have at least one 1D component")
        print(f"{count2d} sublattices have at least one 2D component")
        print(f"{count3d} sublattices have at least one 3D component")

    # print some candidates for realistic TB
    if False:
        nn_ratio = np.divide(NN_1p4.NNN_dists, NN_1p4.NN_dists)
        n_mats = 0
        for mat_ind in range(NN_1p4.nFB_lats):
            good_mat = True
            if NN_1p4.band_gap[mat_ind] != 0:
                good_mat = False
            if nn_ratio[mat_ind] < 1.5:
                good_mat = False
            if NN_1p4.icsd_ids[mat_ind] == -1:
                good_mat = False
            if NN_1p4.NN_dists[mat_ind] > 3:
                good_mat = False
            if NN_1p4.nsites[mat_ind] > 8:
                good_mat = False
            # if NN_1p4.has_bs[mat_ind] == 0:
            #    good_mat = False
            if good_mat:
                print(
                    f"{NN_1p4.mp_id[mat_ind]} {NN_1p4.chem_name[mat_ind]} {NN_1p4.specie_names[mat_ind]}"
                )
                n_mats += 1
        print(n_mats)

    # save the unique info to excel
    if False:
        import pandas as pd

        d = {
            "id": unique_FBs.mp_id,
            "name": unique_FBs.chem_name,
            "species": unique_FBs.specie_names,
            "nFBs": unique_FBs.nFBs,
            "nsites_tot": unique_FBs.nmatsites,
            "nsites_FB": unique_FBs.nsites,
            "NN_dist": unique_FBs.NN_dists,
            "NN_dists_max": unique_FBs.NN_dists_max,
            "NNN_dist": unique_FBs.NNN_dists,
            "NN_max_over_NN": np.divide(unique_FBs.NN_dists_max, unique_FBs.NN_dists),
            "NNN_over_NN_max": np.divide(unique_FBs.NNN_dists, unique_FBs.NN_dists_max),
            "NNN_over_NN": np.divide(unique_FBs.NNN_dists, unique_FBs.NN_dists),
            "nelements": unique_FBs.nelements,
            "spacegroup": unique_FBs.spacegroup_symbol,
            "spacegroup num": unique_FBs.spacegroup_number,
            "E per atom": unique_FBs.energy_per_atom,
            "bandgap": unique_FBs.band_gap,
            "has_bs": unique_FBs.has_bs,
            "ICSD": unique_FBs.icsd_ids,
            "FB_dim": unique_FBs.FB_dims,
            "Systre key": unique_FBs.systre_keys,
            "Systre spacegroup": unique_FBs.systre_sg,
            "RCSR name": unique_FBs.systre_rcsr,
            "flat_with_decay": unique_FBs.flat_with_decay,
            "flat_with_no_decay": unique_FBs.flat_with_no_decay,
            "chi_1p02_no_decay": unique_FBs.p02_no_decay,
            "chi_1p05_no_decay": unique_FBs.p05_no_decay,
            "chi_1p1_no_decay": unique_FBs.p1_no_decay,
            "chi_1p2_no_decay": unique_FBs.p2_no_decay,
            "chi_1p4_no_decay": unique_FBs.p4_no_decay,
            "chi_1p02_decay": unique_FBs.p02_decay,
            "chi_1p05_decay": unique_FBs.p05_decay,
            "chi_1p1_decay": unique_FBs.p1_decay,
            "chi_1p2_decay": unique_FBs.p2_decay,
            "chi_1p4_decay": unique_FBs.p4_decay,
        }
        df = pd.DataFrame(data=d)
        df.to_excel(f"TB_search_flat_bands{file_suffix}.xlsx")

    # save the unique info to excel with lattice invariant classification as well
    if True:
        import pandas as pd

        # get stats on number of systre nets
        systre_keys = {}
        lat_ids = copy.deepcopy(unique_FBs.systre_keys)
        for sublist in unique_FBs.systre_keys:
            for key in sublist:
                if key[0].isdigit():  # not for non-systre
                    if key not in systre_keys.keys():
                        systre_keys[key] = 1
                    else:
                        systre_keys[key] = systre_keys[key] + 1
        comp_count_systre = np.asarray(list(systre_keys.values()))
        systre_key_list = np.asarray(list(systre_keys.keys()))
        sorted_inds_systre = np.flipud(np.argsort(comp_count_systre))
        comp_count_systre = comp_count_systre[sorted_inds_systre]
        systre_key_list = systre_key_list[sorted_inds_systre]
        print(comp_count_systre[0])
        # comp_rank_systre = len(comp_count_systre) - np.argsort(np.argsort(comp_count_systre))

        # gets stats on number of non-systre nets
        with open(f"outputs/collision_graph_invariants_sorted_all.pickle", "rb") as f_pickle:
            temp = pickle.load(f_pickle)
            unique_invariants_list = temp["unique_invariants_list"]
            counts_invs = temp["counts"]
            examples_invs = temp["examples"]
            n_wrong_systre = temp["n_wrong_systre"]
        sorted_inds_invs = np.flipud(np.argsort(counts_invs))
        print(sorted_inds_invs)
        print(type(sorted_inds_invs))
        counts_invs_sorted = []
        examples_invs_sorted =[]
        for sort_ind in sorted_inds_invs:
            counts_invs_sorted.append(counts_invs[sort_ind])
            examples_invs_sorted.append(examples_invs[sort_ind])
        examples_invs = examples_invs_sorted
        counts_invs = counts_invs_sorted
        print(counts_invs[0])
        print(examples_invs[0][0])
        # comp_rank_systre = len(comp_count_systre) - np.argsort(np.argsort(comp_count_systre))

        if True:
            if False: # load what has been saved so far
                with open('outputs/comp_IDs.pickle', "rb") as f_comp_IDs:
                    unique_FBs.comp_IDs = pickle.load(f_comp_IDs)
            else:
                unique_FBs.comp_IDs = [[] for i in range(len(unique_FBs.mp_id))]
            # give each flat band sublattice an ID number
            for mat_ind in range(len(unique_FBs.mp_id)):
                if len(unique_FBs.comp_IDs[mat_ind]) == 0:
                    try:
                        print(f'{mat_ind} of {len(unique_FBs.mp_id)}')
                        # find all the systre key indices
                        for this_key in unique_FBs.systre_keys[mat_ind]:
                            if this_key[0].isdigit():
                                for test_ind, test_key in enumerate(systre_key_list):
                                    if this_key == test_key:
                                        unique_FBs.comp_IDs[mat_ind].append(f'SK-{test_ind+1}')
                                        break
                        # now find all the invariants that match
                        for examples_ind, this_examples in enumerate(examples_invs):
                            for this_example in this_examples:
                                if (this_example["mp_id"] == unique_FBs.mp_id[mat_ind] and 
                                    this_example["specie_name"] == unique_FBs.specie_names[mat_ind] and 
                                    np.abs(this_example["NNN_dists"] - unique_FBs.NNN_dists[mat_ind]) < 0.001):
                                    # and not this_example["systre_key"][0].isdigit()):
                                    unique_FBs.comp_IDs[mat_ind].append(f'LI-{examples_ind+1}')
                        if np.mod(mat_ind, 10000)==0: # save along the way
                            with open('outputs/comp_IDs.pickle', "wb") as f_comp_IDs:
                                pickle.dump(unique_FBs.comp_IDs, f_comp_IDs)
                    except Exception as e:
                        print(e)
                        pass
            
            with open('outputs/comp_IDs.pickle', "wb") as f_comp_IDs:
                pickle.dump(unique_FBs.comp_IDs, f_comp_IDs)
        else:
            with open('outputs/comp_IDs.pickle', "rb") as f_comp_IDs:
                unique_FBs.comp_IDs = pickle.load(f_comp_IDs)
                for ind in range(len(unique_FBs.comp_IDs)):
                    if unique_FBs.comp_IDs[ind]:
                        print(ind)
                #print(unique_FBs.comp_IDs)

        # make anonymous formulas
        unique_FBs.anon_names = np.copy(unique_FBs.chem_name)
        for ind in range(len(unique_FBs.anon_names)):
            unique_FBs.anon_names[ind] = Composition(unique_FBs.chem_name[ind]).anonymized_formula

        # build excel file contents
        d = {
            "id": unique_FBs.mp_id,
            "name": unique_FBs.chem_name,
            "anon_name": unique_FBs.anon_names,
            "species": unique_FBs.specie_names,
            "nFBs": unique_FBs.nFBs,
            "nsites_tot": unique_FBs.nmatsites,
            "nsites_FB": unique_FBs.nsites,
            "NN_dist": unique_FBs.NN_dists,
            "NN_dists_max": unique_FBs.NN_dists_max,
            "NNN_dist": unique_FBs.NNN_dists,
            "NN_max_over_NN": np.divide(unique_FBs.NN_dists_max, unique_FBs.NN_dists),
            "NNN_over_NN_max": np.divide(unique_FBs.NNN_dists, unique_FBs.NN_dists_max),
            "NNN_over_NN": np.divide(unique_FBs.NNN_dists, unique_FBs.NN_dists),
            "nelements": unique_FBs.nelements,
            "spacegroup": unique_FBs.spacegroup_symbol,
            "spacegroup num": unique_FBs.spacegroup_number,
            "E per atom": unique_FBs.energy_per_atom,
            "bandgap": unique_FBs.band_gap,
            "has_bs": unique_FBs.has_bs,
            "ICSD": unique_FBs.icsd_ids,
            "FB_dim": unique_FBs.FB_dims,
            "Comp IDs": unique_FBs.comp_IDs,
            "Systre key": unique_FBs.systre_keys,
            "Systre spacegroup": unique_FBs.systre_sg,
            "RCSR name": unique_FBs.systre_rcsr,
            "flat_with_decay": unique_FBs.flat_with_decay,
            "flat_with_no_decay": unique_FBs.flat_with_no_decay,
            "chi_1p02_no_decay": unique_FBs.p02_no_decay,
            "chi_1p05_no_decay": unique_FBs.p05_no_decay,
            "chi_1p1_no_decay": unique_FBs.p1_no_decay,
            "chi_1p2_no_decay": unique_FBs.p2_no_decay,
            "chi_1p4_no_decay": unique_FBs.p4_no_decay,
            "chi_1p02_decay": unique_FBs.p02_decay,
            "chi_1p05_decay": unique_FBs.p05_decay,
            "chi_1p1_decay": unique_FBs.p1_decay,
            "chi_1p2_decay": unique_FBs.p2_decay,
            "chi_1p4_decay": unique_FBs.p4_decay,
        }
        df = pd.DataFrame(data=d)
        df.to_excel(f"TB_search_flat_bands{file_suffix}_lat_ids.xlsx")

    # pie charts
    if False:
        def my_autopct(pct):
            if pct > 2:
                pct_str = "{p:.1f}".format(p=pct)
            else:
                pct_str = ""
            return pct_str

        # get MP data on elements and space groups
        if True:
            param = paramObj()
            mat_IDs, _, _ = load_all_MP_names(info_path=".")
            with MPRester(param.MP_KEY, notify_db_version=False) as m:
                entries = m.query(
                    criteria={"material_id": {"$in": mat_IDs.tolist()}},
                    properties=["material_id", "elements", "spacegroup"],
                    mp_decode=True,
                    chunk_size=5000,
                )
            mpid_list = []
            el_list = []
            sg_list = []
            for ind, entry in enumerate(entries):
                mpid_list.append(entry["material_id"])
                el_list = el_list + entry["elements"]
                sg_list.append(entry["spacegroup"]["number"])
            with open("els_sgs.pickle", "wb") as f:
                pickle.dump(
                    {"mpid_list": mpid_list, "el_list": el_list, "sg_list": sg_list}, f
                )
        else:
            with open("els_sgs.pickle", "rb") as f:
                load_obj = pickle.load(f)
                mpid_list = load_obj["mpid_list"]
                el_list = load_obj["el_list"]
                sg_list = load_obj["sg_list"]

        # element pie chart
        el_list, el_count = np.unique(el_list, return_counts=True)
        sort_inds_systre = np.argsort(el_count)[::-1]
        el_list = el_list[sort_inds_systre]
        el_count = el_count[sort_inds_systre]
        n_els = 23
        cs = cc.m_CET_R3(np.arange(n_els + 1) / (n_els + 1))
        el_list = np.append(el_list[:n_els], "Other")
        el_count = np.append(el_count[:n_els], sum(el_count[n_els:]))

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(el_count, labels=el_list, autopct=my_autopct, pctdistance=0.8, colors=cs)
        ax.axis("equal")
        plt.savefig("el_pie.png")

        # space group pie chart
        sg_list, sg_count = np.unique(sg_list, return_counts=True)
        sort_inds_systre = np.argsort(sg_count)[::-1]
        sg_list = sg_list[sort_inds_systre]
        sg_count = sg_count[sort_inds_systre]
        n_els = 19
        cs = cc.m_CET_R3(np.arange(n_els + 1) / (n_els + 1))
        sg_list = np.append(sg_list[:n_els], "Other")
        sg_count = np.append(sg_count[:n_els], sum(sg_count[n_els:]))

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(sg_count, labels=sg_list, autopct=my_autopct, pctdistance=0.8, colors=cs)
        ax.axis("equal")
        plt.savefig("sg_pie.png")

        # FB lattice pie chart
        systre_keys = {}
        for sublist in unique_FBs.systre_keys:
            for key in sublist:
                if key not in systre_keys.keys():
                    systre_keys[key] = 1
                else:
                    systre_keys[key] = systre_keys[key] + 1
        comp_count_systre = np.asarray(list(systre_keys.values()))
        key_list = np.asarray(list(systre_keys.keys()))

        tot_FB_lats = sum(comp_count_systre)
        lat_names = []
        lat_counts = []
        lats_left = tot_FB_lats
        for lat_ind in range(len(com_lat_nms)):
            this_count = systre_keys[com_lat_keys[lat_ind]]
            if this_count / tot_FB_lats > 0.03:
                lat_names.append(com_lat_nms[lat_ind])
            else:
                lat_names.append("")
            lat_counts.append(this_count)
            print(
                f"{this_count} ({np.round(100*this_count/tot_FB_lats ,3)}%) lattices of type {com_lat_nms[lat_ind]} found"
            )
            lats_left = lats_left - this_count
        lat_names.append("Other")
        lat_counts.append(lats_left)

        fig, ax = plt.subplots(figsize=(4, 4))

        ax.pie(
            lat_counts,
            labels=lat_names,
            autopct=my_autopct,
            colors=com_lat_clrs,
            pctdistance=0.8,
        )
        # ax.axis('equal')
        plt.savefig("FB_pie.png")

    # barcharts
    if False:
        com_lat_clrs = cc.m_CET_R3(np.arange(11) / 12.0)
        one_scale = [1.0, 2.0, 2.0, 2.0, 2.0]
        O_scale = [4.0, 6.0, 8.0, 4.0, 8.0]
        fig = plt.figure(figsize=(10, 9), constrained_layout=False)
        left = 0.09
        right = 0.8
        top = 0.96
        bottom = 0.07
        wspace = 0.2
        hspace = 0.3
        gs = fig.add_gridspec(
            5,
            2,
            wspace=wspace,
            hspace=hspace,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )
        hist_data_objs = data_objs[:4]  # .append(unique_FBs)
        hist_data_objs.append(unique_FBs)
        hist_max_dists = max_dists[:4]  # .append("Aggregated")
        hist_max_dists.append("all")
        for data_ind, data_obj in enumerate(hist_data_objs):
            # plot barchart of spacegroups
            sgn_list, sgn_count = np.unique(
                data_obj.spacegroup_number, return_counts=True
            )
            sort_inds_systre = np.argsort(sgn_count)[::-1]
            sgn_list = sgn_list[sort_inds_systre]
            sgn_list = [int(i) for i in sgn_list]
            sgn_count = sgn_count[sort_inds_systre]
            n_sg = 20
            current_bottom = np.zeros(n_sg)
            count_left = sgn_count[:n_sg]
            count_left[0] = count_left[0] / one_scale[data_ind]
            for lat_ind in range(9):
                lat_key = com_lat_keys[lat_ind]
                curr_count = np.zeros(n_sg)
                for sg_ind, sg_nm in enumerate(sgn_list[:n_sg]):
                    for mat_ind in np.nonzero(data_obj.spacegroup_number == sg_nm)[0]:
                        if lat_key in data_obj.systre_keys[mat_ind]:
                            frac_count = data_obj.systre_keys[mat_ind].count(
                                lat_key
                            ) / len(data_obj.systre_keys[mat_ind])
                            curr_count[sg_ind] += 1
                # , color=com_lat_clrs[lat_ind])
                curr_count[0] = curr_count[0] / one_scale[data_ind]
                ax = fig.add_subplot(gs[data_ind, 0])
                ax.bar(
                    np.arange(n_sg),
                    curr_count,
                    bottom=current_bottom,
                    color=com_lat_clrs[lat_ind],
                )
                count_left = count_left - curr_count
                current_bottom = current_bottom + curr_count
            # , color=com_lat_clrs[lat_ind+1])
            ax.bar(
                np.arange(n_sg),
                count_left,
                bottom=current_bottom,
                color=com_lat_clrs[lat_ind + 1],
            )
            ax.set_xticks(np.arange(n_sg))
            ax.set_xticklabels(sgn_list[:n_sg], rotation=90)
            ax.set_ylabel("Occurences")
            ax.text(
                0.95,
                0.9,
                r"$\chi_{NN}$=" + str(hist_max_dists[data_ind]),
                ha="right",
                va="top",
                transform=ax.transAxes,
            )
            if data_ind == 4:
                ax.set_xlabel("Spacegroup")

            # plot barchart of elements
            sgn_list, sgn_count = np.unique(data_obj.specie_names, return_counts=True)
            sort_inds_systre = np.argsort(sgn_count)[::-1]
            sgn_list = sgn_list[sort_inds_systre]
            sgn_count = sgn_count[sort_inds_systre]
            n_sg = 20
            current_bottom = np.zeros(n_sg)
            count_left = sgn_count[:n_sg]
            count_left[0] = count_left[0] / O_scale[data_ind]
            bar_list = []
            for lat_ind in range(9):
                lat_key = com_lat_keys[lat_ind]
                curr_count = np.zeros(n_sg)
                for sg_ind, sg_nm in enumerate(sgn_list[:n_sg]):
                    for mat_ind in np.nonzero(data_obj.specie_names == sg_nm)[0]:
                        if lat_key in data_obj.systre_keys[mat_ind]:
                            frac_count = data_obj.systre_keys[mat_ind].count(
                                lat_key
                            ) / len(data_obj.systre_keys[mat_ind])
                            curr_count[sg_ind] += 1
                # , color=com_lat_clrs[lat_ind])
                curr_count[0] = curr_count[0] / O_scale[data_ind]
                ax = fig.add_subplot(gs[data_ind, 1])
                temp = ax.bar(
                    np.arange(n_sg),
                    curr_count,
                    bottom=current_bottom,
                    color=com_lat_clrs[lat_ind],
                )
                bar_list.append(temp)
                count_left = count_left - curr_count
                current_bottom = current_bottom + curr_count
            # , color=com_lat_clrs[lat_ind+1])
            count_left[0] = count_left[0]
            temp = ax.bar(
                np.arange(n_sg),
                count_left,
                bottom=current_bottom,
                color=com_lat_clrs[lat_ind + 1],
            )
            bar_list.append(temp)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax.set_xticks(np.arange(n_sg))
            ax.set_xticklabels(sgn_list[:n_sg], rotation="vertical")
            # ax.set_ylabel("Occurences")
            ax.text(
                0.95,
                0.9,
                r"$\chi_{NN}$=" + str(hist_max_dists[data_ind]),
                ha="right",
                va="top",
                transform=ax.transAxes,
            )

            # plt.figtext((left+right)/2, bottom+(data_ind+1)*(top-bottom)/5, f"NN Distance {hist_max_dists[data_ind]}", ha="center", va="center")

            if data_ind == 4:
                ax.set_xlabel("Element")

            if data_ind == 0:
                ax.legend(
                    [i[0] for i in bar_list],
                    com_lat_nms[: lat_ind + 1] + ["Other"],
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                )

        plt.savefig("sgn_el_bar.png")

    # make latex tables
    if False:
        # makes lattice table
        # loads in the unique lattice key encyclopedia
        with open(f"outputs/key_dictionary{file_suffix}.pickle", "rb") as f:
            key_dict = pickle.load(f)
            key_list = key_dict["key_list"]
            key_ind_list = key_dict["key_ind"]
            comp_count_systre = key_dict["comp_count_systre"]
            n_keys = len(key_list)
            print(f"loaded the {n_keys} unique identified Systre keys")

        # starts writing to tex file
        with open("latex_tables/lats_table.tex", "w") as f_lats:
            # write header
            with open("latex_tables/lats_table_header.txt", "r") as f_lats_header:
                f_lats.write(f_lats_header.read())
                f_lats.write("\n")

            # write each row of the table
            # need to flatten the ragged unique_FB arrays for fast searching
            n_found_lats = len(unique_FBs.systre_keys)
            n_found_keys = 0
            for ind in range(n_found_lats):
                n_found_keys += len(unique_FBs.FB_dims[ind])
            print(f"{n_found_keys} flat band lattice components found in MP")
            unique_systre_keys = [""] * n_found_keys
            unique_systre_sgs = [""] * n_found_keys
            unique_systre_dims = [3] * n_found_keys
            unique_systre_vec = [""] * n_found_keys
            unique_systre_rcsr = [""] * n_found_keys
            current_ind = 0
            for ind in range(n_found_lats):
                num_now = len(unique_FBs.FB_dims[ind])
                unique_systre_keys[
                    current_ind : current_ind + num_now
                ] = unique_FBs.systre_keys[ind]
                unique_systre_sgs[
                    current_ind : current_ind + num_now
                ] = unique_FBs.systre_sg[ind]
                unique_systre_dims[
                    current_ind : current_ind + num_now
                ] = unique_FBs.FB_dims[ind]
                unique_systre_vec[
                    current_ind : current_ind + num_now
                ] = unique_FBs.systre_vec[ind]
                unique_systre_rcsr[
                    current_ind : current_ind + num_now
                ] = unique_FBs.systre_rcsr[ind]
                current_ind += num_now

            n_broken = 0
            for key in unique_systre_keys:
                if not key[0].isdigit():
                    n_broken += 1
            print(
                f"{n_broken} ({np.round(100*n_broken/n_found_keys,3)} %) lattices unable to be classified by systre"
            )

            # now we can find the right info for each entry and make the tex line
            for ind in key_ind_list:
                if ind % 1000 == 0:
                    print(
                        f"{np.round(100*ind/n_keys,1)}% of unique keys saved to tex ({ind} of {n_keys})"
                    )
                if (key_list[ind])[0].isdigit():
                    # find an example of this lattice in unique_FBs
                    this_ind = unique_systre_keys.index(key_list[ind])
                    sg_str = unique_systre_sgs[this_ind]
                    sg_str = sg_str.replace("_", "\_")
                    if unique_systre_dims[this_ind] == 1:
                        ind_str = "\\textcolor{Mahogany}{" + str(ind) + "}"
                    elif unique_systre_dims[this_ind] == 2:
                        ind_str = "\\textcolor{RoyalBlue}{" + str(ind) + "}"
                    else:
                        ind_str = str(ind)
                    # cut off last newline in unique_systre_vec
                    vec_str = unique_systre_vec[this_ind]
                    vec_str = vec_str[:-8]
                    write_str = f"    {ind_str} & {unique_systre_dims[this_ind]} & {sg_str} & {key_list[ind]} & {vec_str} & {comp_count_systre[ind]} & {unique_systre_rcsr[this_ind]} \\\\\n    \\hline\n"
                    f_lats.write(write_str)

            # write footer
            with open("latex_tables/lats_table_footer.txt", "r") as f_lats_footer:
                f_lats.write(f_lats_footer.read())

        # makes material table
        # gets list of materials to print, each will be a different major row of the table
        mat_ID_list_unique = np.unique(unique_FBs.mp_id)
        n_mats = len(mat_ID_list_unique)

        # starts writing to tex file
        with open("latex_tables/mats_table.tex", "w") as f_mats:
            # write header
            with open("latex_tables/mats_table_header.txt", "r") as f_mats_header:
                f_mats.write(f_mats_header.read())
                f_mats.write("\n")

            # for a material...
            for mat_ind, mat_ID in enumerate(mat_ID_list_unique):
                if mat_ind % 10000 == 0:
                    print(
                        f"{np.round(100*mat_ind/n_mats,1)}% of found mats saved to tex ({mat_ind} of {n_mats})"
                    )
                these_inds = np.nonzero(unique_FBs.mp_id == mat_ID)[0]
                these_els = np.unique(unique_FBs.specie_names[these_inds])

                # format name string
                ID_str = ""
                ID_inds = np.arange(0, len(mat_ID), 10)
                mat_ID = str(mat_ID)
                for ID_ind in ID_inds:
                    if len(mat_ID) - ID_ind > 10:
                        ID_str = ID_str + mat_ID[ID_ind : ID_ind + 10] + "\\newline"
                    elif ID_ind < len(mat_ID):
                        ID_str = ID_str + mat_ID[ID_ind : len(mat_ID)]

                # format spacegroup string
                sg_str = unique_FBs.spacegroup_symbol[these_inds[0]][0]
                sg_str = sg_str.replace("_", "\_")
                sg_str = f"{sg_str} \\newline {int(unique_FBs.spacegroup_number[these_inds[0]])}"

                # format ICSD string
                ICSD_str = int(unique_FBs.icsd_ids[these_inds[0]])
                if ICSD_str == -1:
                    ICSD_str = ""
                else:
                    ICSD_str = str(ICSD_str)

                # make all the other complex column strings
                el_str = ""
                nFBs_str = ""
                NN_str = ""
                NNN_str = ""
                key_str = ""
                for el in these_els:
                    # sort out which lattices to plot in what order
                    these_el_inds = np.nonzero(
                        unique_FBs.specie_names[these_inds] == el
                    )[0]
                    sort_inds_systre = np.argsort(unique_FBs.NNN_dists[these_el_inds])
                    these_el_inds = these_inds[these_el_inds[sort_inds_systre]]

                    # element name and NN dist different for each sublattice
                    el_str = el_str + el
                    NN_str = NN_str + str(unique_FBs.NN_dists[these_el_inds[0]])

                    # for each FB lattice found in a material's sublattice
                    for this_el_ind in these_el_inds:
                        nFBs_str = nFBs_str + str(unique_FBs.nFBs[this_el_ind])
                        if (
                            unique_FBs.flat_with_decay[this_el_ind]
                            and unique_FBs.flat_with_no_decay[this_el_ind]
                        ):
                            NNN_str = (
                                NNN_str
                                + "\\textcolor{OliveGreen}{"
                                + str(unique_FBs.NNN_dists[this_el_ind])
                                + "}"
                            )
                        elif unique_FBs.flat_with_decay[this_el_ind]:
                            NNN_str = (
                                NNN_str
                                + "\\textcolor{Bittersweet}{"
                                + str(unique_FBs.NNN_dists[this_el_ind])
                                + "}"
                            )
                        else:
                            NNN_str = NNN_str + str(unique_FBs.NNN_dists[this_el_ind])

                        # print (possibly many) keys
                        key_inds = []
                        for key in unique_FBs.systre_keys[this_el_ind]:
                            this_key_ind = key_ind_list[np.nonzero(key == key_list)[0]]
                            key_inds.append(int(this_key_ind))
                        sort_inds_systre = np.argsort(key_inds)
                        key_inds = [key_inds[i] for i in sort_inds_systre]
                        dim_list = [
                            unique_FBs.FB_dims[this_el_ind][i] for i in sort_inds_systre
                        ]
                        for dim_ind, key_ind in enumerate(key_inds):
                            if dim_list[dim_ind] == 1:
                                key_str = (
                                    key_str
                                    + "\\textcolor{Mahogany}{"
                                    + str(key_ind)
                                    + "} \\newline "
                                )
                            elif dim_list[dim_ind] == 2:
                                key_str = (
                                    key_str
                                    + "\\textcolor{RoyalBlue}{"
                                    + str(key_ind)
                                    + "} \\newline "
                                )
                            elif int(key_ind) == 0:
                                key_str = key_str + "- \\newline "
                            else:
                                key_str = key_str + str(key_ind) + " \\newline "

                            # make all the other strings even with this one
                            el_str = el_str + " \\newline "
                            NN_str = NN_str + " \\newline "
                            nFBs_str = nFBs_str + " \\newline "
                            NNN_str = NNN_str + " \\newline "

                # remove unneeded last newlines
                if el_str.endswith(" \\newline "):
                    el_str = el_str[:-10]
                if nFBs_str.endswith(" \\newline "):
                    nFBs_str = nFBs_str[:-10]
                if NN_str.endswith(" \\newline "):
                    NN_str = NN_str[:-10]
                if NNN_str.endswith(" \\newline "):
                    NNN_str = NNN_str[:-10]
                if key_str.endswith(" \\newline "):
                    key_str = key_str[:-10]

                write_str = f"    {ID_str} & {unique_FBs.chem_name[these_inds[0]]} & {sg_str} & {el_str} & {nFBs_str} & {unique_FBs.nsites[these_inds[0]]} & {ICSD_str} & {np.round(unique_FBs.energy_per_atom[these_inds[0]], 3)} & {NN_str} & {NNN_str} & {key_str} \\\\\n    \\hline\n"
                f_mats.write(write_str)

            # write footer
            with open("latex_tables/mats_table_footer.txt", "r") as f_mats_footer:
                f_mats.write(f_mats_footer.read())

    print('done!')