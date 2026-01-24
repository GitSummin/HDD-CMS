import os
import csv
import pickle
import random
import warnings
from collections import defaultdict

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")


def set_seed(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[WARNING] Invalid SMILES skipped: {smiles}")
        return None

    with warnings.catch_warnings(record=True) as w:
        mol = Chem.AddHs(mol)
        for warn in w:
            print(f"[RDKit Warning] {warn.message}")

    try:
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        logp = Descriptors.MolLogP(mol)
        mol_weight = Descriptors.MolWt(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        ring_count = Descriptors.RingCount(mol)

        feature_values = [
            num_atoms, num_bonds, mol_weight, logp,
            hbd, hba, rotatable_bonds, tpsa, ring_count
        ]

        if any(np.isnan(f) or np.isinf(f) for f in feature_values):
            print(f"[WARNING] NaN or Inf in features for SMILES: {smiles}")
            return None

        return feature_values

    except Exception as e:
        print(f"[ERROR] Failed computing features for {smiles}: {e}")
        return None


def preprocess_data(file_path, N_outputs):
    grouped_data = defaultdict(list)

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    if not file_path or (not os.path.exists(file_path)):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if file_path.lower().endswith(".csv"):
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            first = next(reader, None)
            if first is None:
                return []

            has_header = (len(first) >= 2 and _safe_float(first[1]) is None)

            if has_header:
                header = [h.strip().lower() for h in first]
                idx = {name: i for i, name in enumerate(header)}

                i_smiles = idx.get("smiles", 0)
                i_mz = idx.get("mz", 1)
                i_int = idx.get("intensity", 2)
                i_dep = idx.get("dependency", 3)

                for row in reader:
                    if len(row) <= max(i_smiles, i_mz, i_int):
                        continue
                    smi = row[i_smiles].strip()
                    mz = _safe_float(row[i_mz])
                    inten = _safe_float(row[i_int])
                    dep = _safe_float(row[i_dep]) if i_dep < len(row) else 0.0
                    if mz is None or inten is None:
                        continue
                    if dep is None:
                        dep = 0.0
                    grouped_data[smi].append((mz, inten, dep))
            else:
                rows = [first] + list(reader)
                for row in rows:
                    if len(row) < 3:
                        continue
                    smi = row[0].strip()
                    mz = _safe_float(row[1])
                    inten = _safe_float(row[2])
                    dep = _safe_float(row[3]) if len(row) >= 4 else 0.0
                    if mz is None or inten is None:
                        continue
                    if dep is None:
                        dep = 0.0
                    grouped_data[smi].append((mz, inten, dep))
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 4:
                    continue
                smiles = tokens[0]
                mz = _safe_float(tokens[1])
                intensity = _safe_float(tokens[2])
                dependency = _safe_float(tokens[3])
                if mz is None or intensity is None:
                    continue
                if dependency is None:
                    dependency = 0.0
                grouped_data[smiles].append((mz, intensity, dependency))

    formatted_data = []
    for smiles, values in grouped_data.items():
        values = sorted(values, key=lambda x: x[1], reverse=True)

        if len(values) < N_outputs:
            values += [(0.0, 0.0, 0.0)] * (N_outputs - len(values))

        mz_values, intensity_values, dependency_values = zip(*values)
        formatted_data.append((smiles, list(mz_values), list(intensity_values), list(dependency_values)))

    return formatted_data


def split_dataset_fixed(data, dataset_name, cache_dir="output/splits", train_ratio=0.8, val_ratio=0.1):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_name}_split.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print(f"Split loaded from cache: {cache_path}")
            return pickle.load(f)

    np.random.seed(1234)
    np.random.shuffle(data)

    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    with open(cache_path, "wb") as f:
        pickle.dump((train_data, val_data, test_data), f)
        print(f"Split saved to cache: {cache_path}")

    return train_data, val_data, test_data


def create_atoms(mol, atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    atom_indices = []
    for atom in atoms:
        if atom not in atom_dict:
            atom_dict[atom] = len(atom_dict)
        atom_indices.append(atom_dict[atom])
    return np.array(atom_indices)


def create_bond_dict(mol, bond_dict):
    bond_info = defaultdict(list)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond_type = bond_dict.setdefault(str(b.GetBondType()), len(bond_dict))
        bond_info[i].append((j, bond_type))
        bond_info[j].append((i, bond_type))
    return bond_info


def create_fingerprints(radius, atoms, bond_info, fingerprint_dict, edge_dict, N_fingerprints=20000):
    if len(atoms) == 1 or radius == 0:
        fingerprints = [fingerprint_dict.setdefault(a, min(len(fingerprint_dict), N_fingerprints - 1)) for a in atoms]
    else:
        nodes = atoms
        for _ in range(radius):
            new_nodes = []
            for i in range(len(nodes)):
                neighbors = [(nodes[j], bond) for j, bond in bond_info.get(i, [])]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprint_index = fingerprint_dict.setdefault(fingerprint, len(fingerprint_dict))
                fingerprint_index = min(fingerprint_index, N_fingerprints - 1)
                new_nodes.append(fingerprint_index)
            nodes = new_nodes
        fingerprints = nodes
    return np.array(fingerprints)


def create_datasets(
    dataset_name,
    radius,
    device,
    N_outputs=5,
    checkpoint_dir="output/checkpoint",
    train_file=None,
    val_file=None,
    test_file=None,
):
    fingerprint_dict_path = os.path.join(checkpoint_dir, "fingerprint_dict.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)

    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    def create_dataset_from_list(data_list, split_name="train"):
        dataset = []
        print(f"\n[INFO] {split_name} split raw SMILES count: {len(data_list)}")

        for smiles, mz_values, intensity_values, dependency_values in data_list:
            if len(mz_values) == 0:
                continue

            mol_features = compute_features(smiles)
            if mol_features is None:
                continue

            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = create_atoms(mol, atom_dict)
            bond_info = create_bond_dict(mol, bond_dict)
            fingerprints = create_fingerprints(radius, atoms, bond_info, fingerprint_dict, edge_dict)

            adjacency = Chem.GetAdjacencyMatrix(mol)
            adjacency = torch.FloatTensor(adjacency).to(device)

            fingerprints_tensor = torch.LongTensor(fingerprints).to(device)
            mz_tensor = torch.FloatTensor(mz_values).to(device)
            intensity_tensor = torch.FloatTensor([i / 1000.0 for i in intensity_values]).to(device)
            dependency_tensor = torch.FloatTensor(dependency_values).to(device)

            molecular_sizes = torch.tensor([fingerprints_tensor.shape[0]], device=device)
            feature_tensor = torch.FloatTensor(mol_features).to(device)

            dataset.append((
                smiles,
                fingerprints_tensor,
                adjacency,
                molecular_sizes,
                mz_tensor,
                intensity_tensor,
                dependency_tensor,
                feature_tensor,
            ))

        print(f"[INFO] {split_name} split valid samples (after tensorization): {len(dataset)}")
        return dataset

    default_train = os.path.join("data", dataset_name, "data_train.txt")
    default_test = os.path.join("data", dataset_name, "data_test.txt")

    train_path = train_file or default_train
    test_path = test_file or default_test

    train_data_raw = preprocess_data(train_path, N_outputs=N_outputs)

    if val_file:
        val_data_raw = preprocess_data(val_file, N_outputs=N_outputs)
        train_data = train_data_raw
        val_data = val_data_raw
        test_data = train_data_raw[:0]
    else:
        train_data, val_data, _tmp = split_dataset_fixed(
            train_data_raw,
            dataset_name=dataset_name,
            cache_dir="output/splits",
            train_ratio=0.8,
            val_ratio=0.1,
        )
        test_data = _tmp

    print("\n[SPLIT INFO] (by SMILES)")
    print(f"  Train SMILES: {len(train_data)}")
    print(f"  Val   SMILES: {len(val_data)}")
    print(f"  Test  SMILES: {len(test_data)}")

    dataset_train = create_dataset_from_list(train_data, split_name="train")
    dataset_val = create_dataset_from_list(val_data, split_name="val")

    test_data_raw = preprocess_data(test_path, N_outputs=N_outputs)
    dataset_test = create_dataset_from_list(test_data_raw, split_name="test")

    torch.save(dict(fingerprint_dict), fingerprint_dict_path)
    print(f"\nfingerprint_dict saved: {fingerprint_dict_path}")

    N_fingerprints = len(fingerprint_dict)
    print("N_fingerprints:", N_fingerprints)

    return dataset_train, dataset_val, dataset_test, N_fingerprints, atom_dict


def load_fingerprint_dict(path="output/checkpoint/fingerprint_dict.pth"):
    if os.path.exists(path):
        d = torch.load(path, map_location="cpu")
        return dict(d)
    print(f"[WARN] fingerprint_dict not found at {path}. A new mapping will be created (NOT recommended for inference).")
    return None


def create_external_test_dataset(
    test_path: str,
    radius: int,
    device,
    N_outputs: int = 5,
    checkpoint_dir: str = "output/checkpoint",
):
    """
    Build tensor dataset from an external CSV/TXT file for evaluation.
    Requires fingerprint_dict.pth in checkpoint_dir (generated during training).
    """
    fp_dict_path = os.path.join(checkpoint_dir, "fingerprint_dict.pth")
    fp_loaded = load_fingerprint_dict(fp_dict_path)
    if fp_loaded is None:
        raise RuntimeError(f"[create_external_test_dataset] fingerprint_dict not found: {fp_dict_path}")

    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict), fp_loaded)
    N_fingerprints = len(fp_loaded)

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    data_raw = preprocess_data(test_path, N_outputs=N_outputs)

    def _create_dataset_from_list(data_list):
        dataset = []
        for smiles, mz_values, intensity_values, dependency_values in data_list:
            if len(mz_values) == 0:
                continue

            mol_features = compute_features(smiles)
            if mol_features is None:
                continue

            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = create_atoms(mol, atom_dict)
            bond_info = create_bond_dict(mol, bond_dict)

            fingerprints = create_fingerprints(
                radius,
                atoms,
                bond_info,
                fingerprint_dict,
                edge_dict,
                N_fingerprints=N_fingerprints,
            )

            adjacency = Chem.GetAdjacencyMatrix(mol)
            adjacency = torch.FloatTensor(adjacency).to(device)

            fingerprints_tensor = torch.LongTensor(fingerprints).to(device)
            mz_tensor = torch.FloatTensor(mz_values).to(device)
            intensity_tensor = torch.FloatTensor([i / 1000.0 for i in intensity_values]).to(device)
            dependency_tensor = torch.FloatTensor(dependency_values).to(device)

            molecular_sizes = torch.tensor([fingerprints_tensor.shape[0]], device=device)
            feature_tensor = torch.FloatTensor(mol_features).to(device)

            dataset.append((
                smiles,
                fingerprints_tensor,
                adjacency,
                molecular_sizes,
                mz_tensor,
                intensity_tensor,
                dependency_tensor,
                feature_tensor,
            ))
        return dataset

    dataset_test = _create_dataset_from_list(data_raw)
    return dataset_test, N_fingerprints
