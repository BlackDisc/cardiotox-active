from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from itertools import chain, repeat, islice
import pandas as pd
import re

from PyBioMed.PyMolecule.fingerprint import CalculateECFP2Fingerprint
from PyBioMed.PyMolecule.fingerprint import CalculatePubChemFingerprint


def extract_features_from_smile(
    smiles,
    feature_sets=["DESC", "GR", "SV", "FV", "FP"],
    desc_file_path="CardioTox/data/des_file.txt",
):
    features = []

    if "DESC" in feature_sets:
        features.append(extract_ds_features(smiles, desc_file_path))
    if "GR" in feature_sets:
        features.append(extract_gr_features(smiles))
    if "SV" in feature_sets:
        features.append(extract_sv_features(smiles))
    if "FV" in feature_sets:
        features.append(extract_fv_features(smiles))
    if "FP" in feature_sets:
        features.append(extract_fp_features(smiles))

    return pd.concat(features, axis=1)


def extract_ds_features(smiles, desc_file_path):
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]

    with open(desc_file_path, "r") as fp:
        selected_columns = [line.strip() for line in fp.readlines()]

    df = calc.pandas(mols)[selected_columns]

    return df


def extract_gr_features(smiles):
    features = []
    adj = []
    maxNumAtoms = 50
    for smile in smiles:
        iMol = Chem.MolFromSmiles(smile)
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)

        iFeature = np.zeros((maxNumAtoms, 65))
        iFeatureTmp = []
        for atom in iMol.GetAtoms():
            iFeatureTmp.append(atom_feature(atom))
        iFeature[0 : len(iFeatureTmp), 0:65] = iFeatureTmp

        # Adj-preprocessing
        iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
        iAdj[0 : len(iFeatureTmp), 0 : len(iFeatureTmp)] = iAdjTmp + np.eye(
            len(iFeatureTmp)
        )
        features.append(iFeature.flatten())
        adj.append(iAdj[np.triu_indices(iAdj.shape[1])])
    # features = np.asarray(features)
    # adj = np.asarray(adj)

    df_features = pd.DataFrame(features)
    df_features = df_features.add_prefix("gr_feat_")

    df_adj = pd.DataFrame(adj)
    df_adj = df_adj.add_prefix("gr_adj_")

    return pd.concat([df_features, df_adj], axis=1)


def extract_sv_features(smiles):
    items_list = [
        "$",
        "^",
        "#",
        "(",
        ")",
        "-",
        ".",
        "/",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "=",
        "Br",
        "C",
        "Cl",
        "F",
        "I",
        "N",
        "O",
        "P",
        "S",
        "[2H]",
        "[Br-]",
        "[C@@H]",
        "[C@@]",
        "[C@H]",
        "[C@]",
        "[Cl-]",
        "[H]",
        "[I-]",
        "[N+]",
        "[N-]",
        "[N@+]",
        "[N@@+]",
        "[NH+]",
        "[NH2+]",
        "[NH3+]",
        "[N]",
        "[Na+]",
        "[O-]",
        "[P+]",
        "[S+]",
        "[S-]",
        "[S@+]",
        "[S@@+]",
        "[SH]",
        "[Si]",
        "[n+]",
        "[n-]",
        "[nH+]",
        "[nH]",
        "[o+]",
        "[se]",
        "\\",
        "c",
        "n",
        "o",
        "s",
        "!",
        "E",
    ]
    charset = list(set(items_list))
    charset.sort()
    char_to_int = dict((c, i) for i, c in enumerate(charset))
    pattern = "|".join(re.escape(item) for item in items_list)

    X_smiles_array = np.asarray(smiles)

    def pad_infinite(iterable, padding=None):
        return chain(iterable, repeat(padding))

    def pad(iterable, size, padding=None):
        return islice(pad_infinite(iterable, padding), size)

    token_list = []
    X = []
    for smiles in X_smiles_array:
        tokens = re.findall(pattern, smiles)
        tokens = list(pad(tokens, 97, "E"))

        x = [char_to_int[k] for k in tokens]

        token_list.append(tokens)
        X.append(x)

    df_sv_features = pd.DataFrame(X)
    df_sv_features = df_sv_features.add_prefix("sv_")

    # X=np.asarray(X)

    return df_sv_features


def extract_fv_features(smiles):
    bit_size = 1024
    Max_len = 93
    dataX = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bit_size)
        fp = np.array(fp)
        dataX.append(fp)

    dataX = np.array(dataX)

    data_x = []

    for i in range(len(dataX)):
        fp = [0] * Max_len
        n_ones = 0
        for j in range(bit_size):
            if dataX[i][j] == 1:
                fp[n_ones] = j + 1
                n_ones += 1
        data_x.append(fp)

    df_fv_features = pd.DataFrame(np.array(data_x, dtype=np.int32))
    df_fv_features = df_fv_features.add_prefix("fv_")

    return df_fv_features


def extract_fp_features(smiles):
    features = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        mol_fingerprint = CalculateECFP2Fingerprint(mol)
        pubchem_mol_fingerprint = CalculatePubChemFingerprint(mol)
        feature1 = mol_fingerprint[0]
        feature2 = pubchem_mol_fingerprint
        feature = list(feature1) + list(feature2)
        features.append(feature)
    df_fp_features = pd.DataFrame(features)
    df_fp_features = df_fp_features.add_prefix("fp_")

    return df_fp_features


def atom_feature(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C",
                "N",
                "O",
                "S",
                "F",
                "H",
                "Si",
                "P",
                "Cl",
                "Br",
                "Li",
                "Na",
                "K",
                "Mg",
                "Ca",
                "Fe",
                "As",
                "Al",
                "I",
                "B",
                "V",
                "Tl",
                "Sb",
                "Sn",
                "Ag",
                "Pd",
                "Co",
                "Se",
                "Ti",
                "Zn",
                "Ge",
                "Cu",
                "Au",
                "Ni",
                "Cd",
                "Mn",
                "Cr",
                "Pt",
                "Hg",
                "Pb",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        + one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
        + [atom.GetIsAromatic()]
        + get_ring_info(atom)
    )


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def get_ring_info(atom):
    ring_info_feature = []
    for i in range(3, 9):
        if atom.IsInRingSize(i):
            ring_info_feature.append(1)
        else:
            ring_info_feature.append(0)
    return ring_info_feature
