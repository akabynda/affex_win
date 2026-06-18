# fmt: off
ATOM_NAMES = [
    "C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3", "CG", "CG1", "CG2", "CH2",
    "CZ", "CZ2", "CZ3",
    "H", "H2", "H3", "HA", "HA2", "HA3", "HB", "HB1", "HB2", "HB3",
    "HD1", "HD11", "HD12", "HD13", "HD2", "HD21", "HD22", "HD23", "HD3",
    "HE", "HE1", "HE2", "HE21", "HE22", "HE3",
    "HG", "HG1", "HG11", "HG12", "HG13", "HG2", "HG21", "HG22", "HG23", "HG3",
    "HH", "HH11", "HH12", "HH2", "HH21", "HH22", "HZ", "HZ1", "HZ2", "HZ3",
    "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
    "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "OXT",
    "SD", "SG",
]
# fmt: on
ATOMS_INDICES = {x: i for i, x in enumerate(ATOM_NAMES, start=1)}
RESIDUES = "ARNDCQEGHILKMFPSTWYV"
RESIDUE_INDICES = {x: i for i, x in enumerate(RESIDUES, start=1)}
# ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE
# LEU LYS MET PHE PRO SER THR TRP TYR VAL
