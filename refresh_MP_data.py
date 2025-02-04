from main_search import *
import os
import shutil
from pathlib import Path
import sys
from functools import partial
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

usr_opts = str(sys.argv)


# COMMAND LINE OPTIONS:
"""
    REFRESH STRUCTURES
    "R" will delete current MP materials lists (all_mp_IDs.txt, all_mp_names.txt,
        all_mp_nsites.txt) and structures (mp_structs folder),
        then redownload everything
    CHECK STRUCTURES
    "C" will check to see if any materials listed in all_mp_IDs.txt don't have
        structures in the mp_structs folder, then tries to download them
        (you may need to run this multiple times until the number of materials
        left is 0)
    REFRESH ELECTRONIC DATA
    "B" will delete all current electronic information stored from the MP
        (the mp_bs and mp_dos folders) and attempt to redownload everything
    CHECK ELECTRONIC DATA
    "E" will check if any materials listed in all_mp_IDs.txt that are reported
        as having BS in the materials project do not have anything saved in
        the mp_bs or mp_dos folders, then tries to download them (you may
        need to run this multiple times until the number remaining are 0)
    SAVE GENERAL MP INFORMATION
    "D" will save various relevant information to a text file "MP_info.txt"
"""


# refresh MP lists and structures
if "R" in usr_opts:
    # first delete cached data
    if os.path.exists("all_mp_IDs.txt"):
        os.remove("all_mp_IDs.txt")
    if os.path.exists("all_mp_names.txt"):
        os.remove("all_mp_names.txt")
    if os.path.exists("all_mp_nsites.txt"):
        os.remove("all_mp_nsites.txt")
    try:
        shutil.rmtree("mp_structs")
    except Exception:
        print("Failed to remove mp_structs folder")

    # download data again
    refresh_MP_data(paramObj(), ".")


# check mp_structs completeness
if "C" in usr_opts:
    missing_structs = []
    param = paramObj()
    save_path = Path("mp_structs")

    file_list = os.listdir("mp_structs")
    mat_IDs, _, _ = load_all_MP_names(info_path=".")

    for mat_ID in mat_IDs:
        this_fname = f"{mat_ID}.cif"
        this_fpath = save_path/this_fname
        good_struct = False
        if os.path.exists(this_fpath):
            if os.path.getsize(this_fpath) > 0:
                good_struct = True
        if not good_struct:
            missing_structs.append(mat_ID)

    # try to save missing structs
    save_all_struct_files = partial(save_struct_file, param, save_path)
    if False:
        with multiprocessing.Pool() as pool:
            pool.map(save_all_struct_files, missing_structs)
    if True:
        with ThreadPoolExecutor() as executor:
            executor.map(save_all_struct_files, missing_structs)

    # print(missing_structs)
    print(f"{len(missing_structs)} out of {len(mat_IDs)} structures missing")


# refresh electronic data
if "B" in usr_opts:
    # delete current cached data
    try:
        shutil.rmtree("/data/mp_bs")
        shutil.rmtree("/data/mp_dos")
    except Exception:
        print("Failed to remove folders")
    os.makedirs("/data/mp_bs")
    os.makedirs("/data/mp_dos")

    # get list of mats to check
    param = paramObj()
    if False:
        with MPRester(param.MP_KEY) as m:
            entries = m.query(
                criteria={"has_bandstructure": True}, properties=["material_id"])
        mat_IDs = [i["material_id"] for i in entries]
    if True:
        mat_IDs, _, _ = load_all_MP_names(info_path=".")
    print(
        f"Attempting to download electronic structure of {len(mat_IDs)} materials...")

    # try to download bs for all mats
    save_path_bs = Path("/data/mp_bs")
    save_path_dos = Path("/data/mp_dos")

    save_all_bs_files = partial(save_bs_file, param, save_path_bs)
    save_all_dos_files = partial(save_dos_file, param, save_path_dos)
    if False:
        with multiprocessing.Pool() as pool:
            pool.map(save_all_bs_files, mat_IDs)
        with multiprocessing.Pool() as pool:
            pool.map(save_all_dos_files, mat_IDs)
    if True:
        with ThreadPoolExecutor() as executor:
            executor.map(save_all_bs_files, mat_IDs)
        with ThreadPoolExecutor() as executor:
            executor.map(save_all_dos_files, mat_IDs)


if "T" in usr_opts:
    # converts the pickled bs objects to json
    save_path_bs = Path("/data/mp_bs")
    save_path_dos = Path("/data/mp_dos")

    # file_list_bs = os.listdir(save_path_bs)
    # for file_nm in file_list_bs:
    #     file_nm = str(save_path_bs / file_nm)
    #     if file_nm.endswith(".pickle"):
    #         with open(file_nm, "rb") as f:
    #             bs = pickle.load(f)["bs"]
    #         with open(file_nm[:-6]+"json", "w") as f:
    #             json.dump(bs.as_dict(), f)

    # file_list_dos = os.listdir(save_path_dos)
    # for file_nm in file_list_dos:
    #     file_nm = str(save_path_dos / file_nm)
    #     if file_nm.endswith(".pickle"):
    #         with open(file_nm, "rb") as f:
    #             dos = pickle.load(f)["dos"]
    #         with open(file_nm[:-6]+"json", "w") as f:
    #             json.dump(dos.as_dict(), f)


# check electronic data
if "E" in usr_opts:
    # setup
    param = paramObj()

    missing_bs = []
    missing_dos = []

    save_path_bs = Path("/data/mp_bs")
    save_path_dos = Path("/data/mp_dos")

    file_list_bs = os.listdir(save_path_bs)
    file_list_dos = os.listdir(save_path_dos)

    # get list of materials to check against
    if True:
        with MPRester(param.MP_KEY, notify_db_version=False) as m:
            entries = m.query(criteria={"has_bandstructure": True},
                              properties=["material_id"],
                              chunk_size=100000)
        mat_IDs = [i["material_id"] for i in entries]
    if False:
        mat_IDs, _, _ = load_all_MP_names(info_path=".")

    # check which are missing
    for mat_ID in mat_IDs:
        this_fname = f"{mat_ID}.pickle"
        this_fname_json = f"{mat_ID}.json"
        this_fpath_bs = save_path_bs/this_fname
        this_fpath_dos = save_path_dos/this_fname
        this_fpath_bs_json = save_path_bs/this_fname
        this_fpath_dos_json = save_path_dos/this_fname
        good_bs = False
        good_dos = False
        if os.path.exists(this_fpath_bs):
            if os.path.getsize(this_fpath_bs) > 100:
                good_bs = True
        if os.path.exists(this_fpath_dos):
            if os.path.getsize(this_fpath_dos) > 100:
                good_dos = True
        if os.path.exists(this_fpath_bs_json):
            if os.path.getsize(this_fpath_bs_json) > 100:
                good_bs = True
        if os.path.exists(this_fpath_dos_json):
            if os.path.getsize(this_fpath_dos_json) > 100:
                good_dos = True
        if not good_bs:
            missing_bs.append(mat_ID)
        if not good_dos:
            missing_dos.append(mat_ID)
    print(f"{len(missing_bs)} out of {len(mat_IDs)} bs missing")
    print(f"{len(missing_dos)} out of {len(mat_IDs)} dos missing")

    # save the bs and dos
    save_all_bs_files = partial(save_bs_file, param, save_path_bs)
    save_all_dos_files = partial(save_dos_file, param, save_path_dos)
    if False:
        n = 10  # chunksize to grab at once
        missing_bs_list = [
            missing_bs[i * n:(i + 1) * n] for i in range((len(missing_bs) + n - 1) // n)]
        with MPRester(param.MP_KEY) as m:
            for ind, mis_bs_l in enumerate(missing_bs_list):
                print(
                    f"\n\nNow getting materials {ind*n+1}-{ind*n+n} of {len(missing_bs)}\n\n")
                entries = m.query(criteria={"material_id": {"$in": mis_bs_l}},
                                  properties=["material_id", "band_structure"],
                                  mp_decode=True)  # "bandstructure_uniform"
                print(entries)
                exit()
                for _, entry in enumerate(entries):
                    this_bs = entry["bandstructure"]
                    this_mat_ID = entry["material_id"]
                    newfile_path = save_path_bs / (f"{this_mat_ID}.pickle")
                    with open(newfile_path, 'wb') as f_bs:
                        pickle.dump({"bs": this_bs}, f_bs)
    if False:
        with MPRester(param.MP_KEY) as m:
            for ind, mat_ID in enumerate(missing_bs):
                print(f"now saving bs of {mat_ID}")
                if ind % 100 == 0:
                    print(f"{np.round(100*ind/len(missing_bs),1)}% complete\n")
                try:
                    bs = m.get_bandstructure_by_material_id(
                        material_id=mat_ID, line_mode=True)
                    newfile_path = save_path_bs / (f"{mat_ID}.pickle")
                    with open(newfile_path, 'wb') as f_bs:
                        pickle.dump({"bs": bs}, f_bs)
                except Exception:
                    pass
    if False:
        with multiprocessing.Pool() as pool:
            pool.map(save_all_bs_files, mat_IDs)
        with multiprocessing.Pool() as pool:
            pool.map(save_all_dos_files, mat_IDs)
    if True:
        with ThreadPoolExecutor(max_workers=32) as executor:
            executor.map(save_all_bs_files, missing_bs)
        with ThreadPoolExecutor(max_workers=32) as executor:
            executor.map(save_all_dos_files, missing_dos)


# get MP general info
if "D" in usr_opts:
    # get list of all materials
    mat_IDs, _, _ = load_all_MP_names(info_path=".")

    # get properties from MP
    param = paramObj()
    with MPRester(param.MP_KEY) as m:
        entries = m.query(criteria={"material_id": {"$in": mat_IDs.tolist()}},
                          properties=["material_id", "pretty_formula",
                                      "band_gap", "formation_energy_per_atom", "nsites",
                                      "nelements", "spacegroup", "has_bandstructure",
                                      "icsd_ids"],
                          mp_decode=True,
                          chunk_size=500000)

    # save to text file
    with open("MP_info.txt", "w") as f_mp:
        f_mp.write("material_id, formula, nsites, nelements, "
                   + "spacegroup_symbol, spacegroup_number, "
                   + "energy_per_atom, band_gap, has_bandstructure, "
                   + "icsd_ids\n")
        for ind, entry in enumerate(entries):
            thismatid = entry["material_id"]
            thisform = entry["pretty_formula"]
            thisnsites = entry["nsites"]
            thisnel = entry["nelements"]
            this_sgs = entry["spacegroup"]["symbol"]
            thissgn = entry["spacegroup"]["number"]
            thisE = entry["formation_energy_per_atom"]
            thisbg = entry["band_gap"]
            thishbs = entry["has_bandstructure"]
            if len(entry["icsd_ids"]) > 0:
                icsd_str = entry["icsd_ids"][0]
            else:
                icsd_str = "-1"

            f_mp.write(f"{thismatid}, {thisform}, "
                       + f"{thisnsites}, {thisnel}, "
                       + f"{this_sgs}, {thissgn}, "
                       + f"{thisE}, {thisbg}, "
                       + f"{thishbs}, {icsd_str}\n")
