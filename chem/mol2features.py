from collections import Counter
from typing import Callable, Union

import numpy as np
from rdkit import Chem
from typing import List, Tuple, Union

from featuralization.mol2features import register_features_generator
from featuralization.rdDescriptors import RDKit2D

Molecule = Union[str, Chem.Mol]
FeaturesGenerator = Callable[[Molecule], np.ndarray]
RDKIT_PROPS = ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
               'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
               'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
               'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
               'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
               'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
               'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
               'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
               'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
               'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
               'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
               'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
               'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
               'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
               'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
               'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
               'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

BOND_FEATURES = ['BondType', 'Stereo', 'BondDir']
# BOND_FEATURES = ['BondType', 'Stereo']
# BOND_FEATURES = ['Stereo']

MAX_ATOMIC_NUM = 119
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# sum([119 + 1, 6 + 1, 6, 5, 6, 6, 2, 18])

# len(choices) + 1 to include room for uncommon values and unknown value; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


def get_atom_fdim() -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM + 18


@register_features_generator('fgtasklabel')
def rdkit_functional_group_label_features_generator(mol: Molecule) -> np.ndarray:
    """
    Generates functional group label for a molecule in RDKit.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :return: A 1D numpy array containing the RDKit 2D features.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    generator = RDKit2D(RDKIT_PROPS)
    features = generator.process(smiles)[1:]
    features = np.array(features)
    features[features != 0] = 1
    return features

def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    if min(choices) < 0:
        index = value
    else:
        index = choices.index(value) if value in choices else -1 # uncommon value
    encoding[index] = 1

    return encoding


def atom_features(self, atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]
    # len(features) = ATOM_FDIM
    atom_idx = atom.GetIdx()
    features = features + \
               onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
               [atom_idx in self.hydrogen_acceptor_match] + \
               [atom_idx in self.hydrogen_donor_match] + \
               [atom_idx in self.acidic_match] + \
               [atom_idx in self.basic_match] + \
               [self.ring_info.IsAtomInRingOfSize(atom_idx, 3),
                self.ring_info.IsAtomInRingOfSize(atom_idx, 4),
                self.ring_info.IsAtomInRingOfSize(atom_idx, 5),
                self.ring_info.IsAtomInRingOfSize(atom_idx, 6),
                self.ring_info.IsAtomInRingOfSize(atom_idx, 7),
                self.ring_info.IsAtomInRingOfSize(atom_idx, 8)]
    return features


def atom_to_vocab(mol, atom):
    """
    Convert atom to vocabulary. The convention is based on atom type and bond type.
    :param mol: the molecular.
    :param atom: the target atom.
    :return:
    """
    nei = Counter()
    for a in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx())
        nei[str(a.GetSymbol()) + "-" + str(bond.GetBondType())] += 1
    keys = nei.keys()
    keys = list(keys)
    keys.sort()
    output = atom.GetSymbol()
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])

    # The generated atom_vocab is too long?
    return output

def bond_to_vocab(mol, bond):
    """
    Convert bond to vocabulary. The convention is based on atom type and bond type.
    Considering one-hop neighbor atoms
    :param mol: the molecular.
    :param atom: the target atom.
    :return:
    """
    nei = Counter()
    two_neighbors = (bond.GetBeginAtom(), bond.GetEndAtom())
    two_indices = [a.GetIdx() for a in two_neighbors]
    for nei_atom in two_neighbors:
        for a in nei_atom.GetNeighbors():
            a_idx = a.GetIdx()
            if a_idx in two_indices:
                continue
            tmp_bond = mol.GetBondBetweenAtoms(nei_atom.GetIdx(), a_idx)
            nei[str(nei_atom.GetSymbol()) + '-' + get_bond_feature_name(tmp_bond)] += 1
    keys = list(nei.keys())
    keys.sort()
    output = get_bond_feature_name(bond)
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])
    return output

def get_bond_feature_name(bond):
    """
    Return the string format of bond features.
    Bond features are surrounded with ()

    """
    ret = []
    for bond_feature in BOND_FEATURES:
        fea = eval("bond.Get{}".format(bond_feature))()
        ret.append(str(fea))

    return '(' + '-'.join(ret) + ')'


if __name__ == '__main__':

    # smiles = "CC(C)Nc1c(nc2ncccn12)c3ccc4[nH]ncc4c3"
    smiles = "CC(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    # for i, atom in enumerate(mol.GetAtoms()):
    #     print(atom_to_vocab(mol, atom))
    features = rdkit_functional_group_label_features_generator(mol)
    print(features)


    # for i, bond in enumerate(mol.GetBonds()):
    #     print(bond_to_vocab(mol, bond))
