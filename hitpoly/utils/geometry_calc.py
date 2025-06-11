import torch, itertools, copy
from scipy.linalg import block_diag
import numpy as np
from collections import OrderedDict
from multiprocessing import Pool, Manager
import multiprocessing
import sys, os, time
from collections import deque


def get_b(xyz, bonds):
    b = (xyz[:, bonds[:, 1]] - xyz[:, bonds[:, 0]]).pow(2).sum(-1).pow(0.5)
    return b



def get_theta(xyz, angles):
    angle_vec1 = xyz[:, angles[:, 0]] - xyz[:, angles[:, 1]]
    angle_vec2 = xyz[:, angles[:, 2]] - xyz[:, angles[:, 1]]

    norm = (angle_vec1.pow(2).sum(2) * angle_vec2.pow(2).sum(2)).sqrt()
    cos_theta = (angle_vec1 * angle_vec2).sum(2) / norm
    theta = torch.acos(cos_theta)
    return (theta, cos_theta)


def get_phi(xyz, dihedrals):
    vec1 = xyz[:, dihedrals[:, 0]] - xyz[:, dihedrals[:, 1]]
    vec2 = xyz[:, dihedrals[:, 2]] - xyz[:, dihedrals[:, 1]]
    vec3 = xyz[:, dihedrals[:, 1]] - xyz[:, dihedrals[:, 2]]
    vec4 = xyz[:, dihedrals[:, 3]] - xyz[:, dihedrals[:, 2]]

    cross1 = torch.linalg.cross(vec1, vec2, dim=-1)
    cross2 = torch.linalg.cross(vec3, vec4, dim=-1)
    norm = (cross1.pow(2).sum(2) * cross2.pow(2).sum(2)).sqrt()
    cos_phi = (cross1 * cross2).sum(2) / norm
    v = vec2 / (vec2.pow(2).sum(2).pow(0.5).unsqueeze(2))
    PHI1 = vec1 - (vec1 * v).sum(2).unsqueeze(2) * v
    PHI2 = vec4 - (vec4 * v).sum(2).unsqueeze(2) * v
    S = torch.sign((torch.linalg.cross(PHI1, PHI2, dim=-1) * v).sum(2))
    phi = torch.pi + S * (torch.acos(cos_phi / 1.000001) - torch.pi)
    return (phi, cos_phi)


def get_chi(xyz, impropers):
    A, B, C, D = 0, 1, 2, 3
    permutations = [[A, B, C, D], [A, C, D, B], [A, D, B, C]]
    chi = []
    for permutation in permutations:
        A, B, C, D = permutation
        rAB = xyz[:, impropers[:, A]] - xyz[:, impropers[:, B]]
        rAC = xyz[:, impropers[:, A]] - xyz[:, impropers[:, C]]
        rAD = xyz[:, impropers[:, A]] - xyz[:, impropers[:, D]]
        RAB = rAB.pow(2).sum(2).pow(0.5).view(xyz.shape[0], -1, 1)
        RAC = rAC.pow(2).sum(2).pow(0.5).view(xyz.shape[0], -1, 1)
        RAD = rAD.pow(2).sum(2).pow(0.5).view(xyz.shape[0], -1, 1)
        uAB = rAB / RAB
        uAC = rAC / RAC
        uAD = rAD / RAD
        sinCAD = (
            torch.linalg.cross(uAC, uAD, dim=-1)
            .pow(2)
            .sum(2)
            .pow(0.5)
            .view(xyz.shape[0], -1, 1)
        )
        sinchiBACD = (torch.linalg.cross(uAC, uAD, dim=-1) * uAB).sum(2).view(
            xyz.shape[0], -1, 1
        ) / sinCAD
        chiBACD = torch.asin(sinchiBACD)
        chi.append(chiBACD)
    chi = torch.cat(chi, dim=2)
    chi = chi.mean(2)
    return chi


def get_d(xyz, impropers):
    A, B, C, D = 0, 1, 2, 3
    rAB = xyz[:, impropers[:, A]] - xyz[:, impropers[:, B]]
    rAC = xyz[:, impropers[:, A]] - xyz[:, impropers[:, C]]
    rAD = xyz[:, impropers[:, A]] - xyz[:, impropers[:, D]]
    # got an error message for torch. cross changed to torch.linalg.cross - Artur 03/29
    rACD = torch.linalg.cross(rAC, rAD)
    uACD = rACD / (rACD.pow(2).sum(2).pow(0.5).view(xyz.shape[0], -1, 1))
    d = (rAB * uACD).sum(2).view(xyz.shape[0], -1, 1)
    d = d * uACD
    return d


# SHARED DERIVATIVE FUNCTIONS


def get_db(xyz, bonds):
    b = get_b(xyz, bonds)
    device = bonds.device
    r = xyz[:, bonds]
    I = torch.eye(2).to(device)
    b = b.view(-1, bonds.shape[0], 1).unsqueeze(-1)
    rb = r[:, :, [1]] - r[:, :, [0]]
    db = rb * (I[1] - I[0]).view(1, 2, 1) / b
    return db


def get_dtheta(xyz, angles):
    theta, cos_theta = get_theta(xyz, angles)
    device = angles.device
    r = xyz[:, angles]
    r0 = r[:, :, [0]] - r[:, :, [1]]
    R0 = r0.pow(2).sum(3).pow(0.5).unsqueeze(2)
    r2 = r[:, :, [2]] - r[:, :, [1]]
    R2 = r2.pow(2).sum(3).pow(0.5).unsqueeze(2)
    cos_theta = cos_theta.view(xyz.shape[0], -1, 1).unsqueeze(2)
    I = torch.eye(3).to(device)
    A = (r0 / (R0 * R2) - cos_theta * r2 / (R2.pow(2))) * (I[2] - I[1]).view(1, 3, 1)
    B = (r2 / (R0 * R2) - cos_theta * r0 / (R0.pow(2))) * (I[0] - I[1]).view(1, 3, 1)
    dtheta = -((1 - cos_theta.pow(2)).pow(-0.5)) * (A + B)
    return dtheta


def get_dphi(xyz, dihedrals):
    phi, cos_phi = get_phi(xyz, dihedrals)
    device = dihedrals.device
    N = len(dihedrals)
    r = xyz[:, dihedrals].unsqueeze(3)
    e = torch.eye(3).view(1, 1, 3, 3).expand(xyz.shape[0], N, 4, 3, 3).to(device)
    I = torch.eye(4).to(device)
    cos_phi = torch.cos(phi.view(xyz.shape[0], -1, 1)).unsqueeze(3)
    r12 = (r[:, :, [1]] - r[:, :, [0]]).expand(xyz.shape[0], N, 4, 3, 3)
    r23 = (r[:, :, [2]] - r[:, :, [1]]).expand(xyz.shape[0], N, 4, 3, 3)
    r123 = torch.linalg.cross(r12, r23, dim=4)
    R123 = r123.pow(2).sum(4).pow(0.5).unsqueeze(4)
    delta23 = (I[2] - I[1]).view(1, 4, 1, 1).expand(xyz.shape[0], N, 4, 3, 3)
    delta12 = (I[1] - I[0]).view(1, 4, 1, 1).expand(xyz.shape[0], N, 4, 3, 3)
    dr123 = (
        torch.linalg.cross(r12, e, dim=4) * delta23
        - torch.linalg.cross(r23, e, dim=4) * delta12
    )
    dr123 *= -1
    r23 = (r[:, :, [2]] - r[:, :, [1]]).expand(xyz.shape[0], N, 4, 3, 3)
    r34 = (r[:, :, [3]] - r[:, :, [2]]).expand(xyz.shape[0], N, 4, 3, 3)
    r234 = torch.linalg.cross(r23, r34, dim=4)
    R234 = r234.pow(2).sum(4).pow(0.5).unsqueeze(4)
    delta34 = (I[3] - I[2]).view(1, 4, 1, 1).expand(xyz.shape[0], N, 4, 3, 3)
    delta23 = (I[2] - I[1]).view(1, 4, 1, 1).expand(xyz.shape[0], N, 4, 3, 3)
    dr234 = (
        torch.linalg.cross(r23, e, dim=4) * delta34
        - torch.linalg.cross(r34, e, dim=4) * delta23
    )
    dr234 *= -1
    dcos_phi = 0.0
    dcos_phi += (r123 * dr234).sum(4)
    dcos_phi += (r234 * dr123).sum(4)
    dcos_phi /= (R123 * R234).squeeze(4)
    dcos_phi -= cos_phi * ((r234 / R234.pow(2)) * dr234).sum(4)
    dcos_phi -= cos_phi * ((r123 / R123.pow(2)) * dr123).sum(4)
    dcos_phi *= -1
    dphi = -(1 - cos_phi.pow(2)).pow(-0.5) * dcos_phi
    return dphi


def get_dchi(xyz, impropers):
    displacements = xyz.unsqueeze(2).expand(xyz.shape[0], xyz.shape[1], xyz.shape[1], 3)
    displacements = displacements.transpose(1, 2) - displacements
    A, B, C, D = 0, 1, 2, 3
    permutations = [[A, B, C, D], [A, C, D, B], [A, D, B, C]]
    dchi = []
    for permutation in permutations:
        A, B, C, D = permutation
        rAB = displacements[:, impropers[:, A], impropers[:, B]]
        rAC = displacements[:, impropers[:, A], impropers[:, C]]
        rAD = displacements[:, impropers[:, A], impropers[:, D]]

        RAB = rAB.pow(2).sum(2).pow(0.5).view(xyz.shape[0], -1, 1)
        RAC = rAC.pow(2).sum(2).pow(0.5).view(xyz.shape[0], -1, 1)
        RAD = rAD.pow(2).sum(2).pow(0.5).view(xyz.shape[0], -1, 1)
        uAB = rAB / RAB
        uAC = rAC / RAC
        uAD = rAD / RAD
        sinCAD = (
            torch.linalg.cross(uAC, uAD, dim=-1)
            .pow(2)
            .sum(2)
            .pow(0.5)
            .view(xyz.shape[0], -1, 1)
        )
        sinchiBACD = (torch.linalg.cross(uAC, uAD, dim=-1) * uAB).sum(2).view(
            xyz.shape[0], -1, 1
        ) / sinCAD
        chiBACD = torch.asin(sinchiBACD)
        a = (uAB * torch.linalg.cross(uAC, uAD)).sum(2).view(xyz.shape[0], -1, 1, 1)
        r = xyz[:, impropers[:, [A, B, C, D]]]
        I = torch.eye(4).to(impropers.device)
        dcosCAD = 0.0

        dcosCAD += ((r[:, :, [2]] - r[:, :, [1]]) * (I[3] - I[1]).view(1, 4, 1)) / (
            RAC * RAD
        ).unsqueeze(2)
        dcosCAD += ((r[:, :, [3]] - r[:, :, [1]]) * (I[2] - I[1]).view(1, 4, 1)) / (
            RAC * RAD
        ).unsqueeze(2)
        cosCAD = (uAC * uAD).sum(2).view(xyz.shape[0], -1, 1, 1)
        dcosCAD -= (
            cosCAD
            * ((r[:, :, [3]] - r[:, :, [1]]) * (I[3] - I[1]).view(1, 4, 1))
            / (RAD.pow(2)).unsqueeze(2)
        )
        dcosCAD -= (
            cosCAD
            * ((r[:, :, [2]] - r[:, :, [1]]) * (I[2] - I[1]).view(1, 4, 1))
            / (RAC.pow(2)).unsqueeze(2)
        )
        dsinCAD = cosCAD * (-(1 - cosCAD.pow(2)).pow(-0.5)) * dcosCAD

        b = -(1 / sinCAD.unsqueeze(2).pow(2)) * dsinCAD
        c = sinCAD.unsqueeze(2).pow(-1)
        d = 0.0
        RAB = RAB.unsqueeze(2)
        RAC = RAC.unsqueeze(2)
        RAD = RAD.unsqueeze(2)
        N = RAB * RAC * RAD
        dN = 0.0

        dN += (
            RAC
            * RAD
            * ((r[:, :, [0]] - r[:, :, [1]]) * (I[0] - I[1]).view(1, 4, 1))
            / RAB
        )
        dN += (
            RAB
            * RAD
            * ((r[:, :, [2]] - r[:, :, [1]]) * (I[2] - I[1]).view(1, 4, 1))
            / RAC
        )
        dN += (
            RAB
            * RAC
            * ((r[:, :, [3]] - r[:, :, [1]]) * (I[3] - I[1]).view(1, 4, 1))
            / RAD
        )
        e = 0.0
        e += torch.linalg.cross(rAB, rAC, dim=-1).unsqueeze(2) * (I[3] - I[1]).view(
            1, 4, 1
        )
        e += torch.linalg.cross(rAD, rAB, dim=-1).unsqueeze(2) * (I[2] - I[1]).view(
            1, 4, 1
        )
        e += torch.linalg.cross(rAC, rAD, dim=-1).unsqueeze(2) * (I[0] - I[1]).view(
            1, 4, 1
        )
        d += (
            (rAB * torch.linalg.cross(rAC, rAD, dim=-1))
            .sum(2)
            .view(xyz.shape[0], -1, 1, 1)
            * (-1 / N.pow(2))
            * dN
        )
        d += (1 / N) * e
        dsinchi = a * b + c * d
        dchi.append((1 - sinchiBACD.unsqueeze(2).pow(2)).pow(-0.5) * dsinchi)
    dchi = torch.stack(dchi).mean(0)
    dchi *= -1
    return dchi


# GEOMETRIES


def get_bond_geometry(xyz, bonds):
    b = get_b(xyz, bonds)
    return b


def get_drude_bond_geometry(xyz, drude_bonds):
    b = get_drude_b(xyz, drude_bonds)
    return b


def get_angle_geometry(xyz, angles):
    theta, cos_theta = get_theta(xyz, angles)
    bonds = angles[:, [1, 0, 1, 2]].view(-1, 2)
    b = get_b(xyz, bonds).view(xyz.shape[0], -1, 2)
    return (cos_theta, theta, b)


def get_dihedral_geometry(xyz, dihedrals):
    phi, cos_phi = get_phi(xyz, dihedrals)
    angles = torch.stack([dihedrals[:, :3], dihedrals[:, -3:]], dim=1).view(-1, 3)
    theta, cos_theta = get_theta(xyz, angles)
    cos_theta = cos_theta.view(xyz.shape[0], -1, 2)
    theta = theta.view(xyz.shape[0], -1, 2)
    bonds = dihedrals[:, [0, 1, 1, 2, 2, 3]].view(-1, 2)
    b = get_b(xyz, bonds).view(xyz.shape[0], -1, 3)
    return (cos_phi, phi, theta, b)


def get_improper_geometry(xyz, impropers):
    chi = get_chi(xyz, impropers)
    d = get_d(xyz, impropers)
    i, j, k, l = 0, 1, 2, 3
    angles = torch.stack(
        [impropers[:, [i, j, k]], impropers[:, [i, j, l]], impropers[:, [i, k, l]]],
        dim=1,
    ).view(-1, 3)
    theta, cos_theta = get_theta(xyz, angles)
    cos_theta = cos_theta.view(xyz.shape[0], -1, 3)
    theta = theta.view(xyz.shape[0], -1, 3)
    return (chi, cos_theta, theta, d)


# DERIVATIVES


def get_bond_derivatives(xyz, bonds):
    db = get_db(xyz, bonds)
    return db



def get_angle_derivatives(xyz, angles):
    dtheta = get_dtheta(xyz, angles)
    bonds = angles[:, [0, 1, 1, 2]].view(-1, 2)
    _db = get_db(xyz, bonds).view(xyz.shape[0], -1, 2, 2, 3)
    db = torch.zeros(xyz.shape[0], _db.shape[1], 2, 3, 3).to(xyz.device)
    db[:, :, 0, [0, 1]] += _db[:, :, 0]
    db[:, :, 1, [1, 2]] += _db[:, :, 1]
    return (dtheta, db)


def get_dihedral_derivatives(xyz, dihedrals):
    dphi = get_dphi(xyz, dihedrals)
    angles = torch.stack([dihedrals[:, :3], dihedrals[:, -3:]], dim=1).view(-1, 3)
    _dtheta = get_dtheta(xyz, angles).view(xyz.shape[0], -1, 2, 3, 3)
    dtheta = torch.zeros(xyz.shape[0], _dtheta.shape[1], 2, 4, 3).to(xyz.device)
    dtheta[:, :, 0, [0, 1, 2]] += _dtheta[:, :, 0]
    dtheta[:, :, 1, [1, 2, 3]] += _dtheta[:, :, 1]
    bonds = dihedrals[:, [0, 1, 1, 2, 2, 3]].view(-1, 2)
    _db = get_db(xyz, bonds).view(xyz.shape[0], -1, 3, 2, 3)
    db = torch.zeros(xyz.shape[0], _db.shape[1], 3, 4, 3).to(xyz.device)
    db[:, :, 0, [0, 1]] += _db[:, :, 0]
    db[:, :, 1, [1, 2]] += _db[:, :, 1]
    db[:, :, 2, [2, 3]] += _db[:, :, 2]
    return (dphi, dtheta, db)


def get_improper_derivatives(xyz, impropers):
    dchi = get_dchi(xyz, impropers)
    i, j, k, l = 0, 1, 2, 3
    angles = torch.stack(
        [impropers[:, [i, j, k]], impropers[:, [i, j, l]], impropers[:, [i, k, l]]],
        dim=1,
    ).view(-1, 3)
    _dtheta = get_dtheta(xyz, angles).view(xyz.shape[0], -1, 3, 3, 3)
    dtheta = torch.zeros(xyz.shape[0], _dtheta.shape[1], 3, 4, 3).to(xyz.device)
    dtheta[:, :, 0, [1, 0, 2]] += _dtheta[:, :, 0]
    dtheta[:, :, 1, [1, 0, 3]] += _dtheta[:, :, 1]
    dtheta[:, :, 2, [2, 0, 3]] += _dtheta[:, :, 2]
    return (dchi, dtheta)


############################### CREATE TOPS ###########################################################


def one_thread_product(neighbors):
    temp_list = []
    ind = neighbors[0]
    for x in neighbors[1]:
        if not isinstance(x, int):
            temp_list += [[]]
        else:
            if not isinstance(x, int):
                temp_list += list(itertools.product(x, [ind]))
            else:
                temp_list += list(itertools.product([x], [ind]))
    return temp_list


def build_bonds(graph, neighbors):
    n_threads = int(os.environ.get("OMP_NUM_THREADS", 1))

    if n_threads == 1:
        # serial impl
        bonds = [list(itertools.product(x, [ind])) for ind, x in enumerate(neighbors)]
        bonds = list(itertools.chain(*bonds))
        # bonds = torch.LongTensor([list(b) for b in bonds]) ?? this is here why?? torch does not care if you give it tuples
    else:
        # pool imp;
        # enum does not have global info so we zip it into iterator
        enum_ = range(len(neighbors))
        neigh_zip = zip(enum_, neighbors)

        with Pool(n_threads) as pool:
            bonds = pool.map(
                one_thread_product,
                neigh_zip,
                # chunksize=int(len(neighbors) / n_threads / 2),
                chunksize=1,
            )
        pool.close()
        bonds = list(itertools.chain(*bonds))
        # bonds = sum(bonds, [])
    bonds = torch.LongTensor(bonds)
    bonds = bonds[bonds[:, 1] > bonds[:, 0]]

    return bonds.view(-1, 2).to(graph.train_args.device)


def combine_and_build(i, neighbors):
    pairs = list(itertools.combinations(neighbors[i], 2))
    return [[pair[0], i, pair[1]] for pair in pairs]


def build_angles(graph, neighbors):
    n_threads = int(os.environ.get("OMP_NUM_THREADS", 1))

    if n_threads == 1:
        # serial impl
        # this is old leaving here for posterity
        # angles = [list(itertools.combinations(x, 2)) for x in neighbors]
        # angles = [
        #     [[pair[0]] + [i] + [pair[1]] for pair in pairs]
        #     for i, pairs in enumerate(angles)
        # ]
        angles = [
            [pair[0], i, pair[1]]
            for i in range(len(neighbors))
            for pair in itertools.combinations(neighbors[i], 2)
        ]
    else:
        with Pool(n_threads) as pool:
            # Prepare data to be mapped
            indices = list(range(len(neighbors)))
            # Perform combined combination and building in parallel
            angles = pool.starmap(
                combine_and_build,
                [(i, neighbors) for i in indices],
                # chunksize=(len(neighbors) // (n_threads * 2) or 1),
                chunksize=1,
            )
    angles = list(itertools.chain(*angles))
    angles = torch.LongTensor(angles)

    return angles.view(-1, 3).to(graph.train_args.device)


def worker(i, neighbors):
    result = []
    neighbors_i = set(neighbors[i])
    for j in neighbors[i]:
        neighbors_j = set(neighbors[j])
        k = neighbors_i - {j}
        l = neighbors_j - {i}
        if k and l:
            pairs = [(a, b) for a, b in itertools.product(k, l) if a < b]
            result.extend([[a, i, j, b] for a, b in pairs if len([a, i, j, b]) >= 4])
    return result


def build_dihedrals(graph, neighbors):
    n_threads = int(os.environ.get("OMP_NUM_THREADS", 1))

    if n_threads == 1:
        dihedrals = []
        for i in range(len(neighbors)):
            for j in neighbors[i]:
                k = set(neighbors[i]) - set([j])
                l = set(neighbors[j]) - set([i])
                if k and l:
                    pairs = list(
                        filter(lambda pair: pair[0] < pair[1], itertools.product(k, l))
                    )
                    for pair in pairs:
                        if len(pair) >= 2:
                            dihedrals += [[pair[0]] + [i] + [j] + [pair[1]]]
    else:
        dihedrals = []
        with Manager() as manager:
            shared_neighbors = manager.list(neighbors)
            with Pool(n_threads) as pool:
                results = pool.starmap(
                    worker,
                    [(i, shared_neighbors) for i in range(len(neighbors))],
                    # chunksize=(len(neighbors) // (n_threads * 2) or 1),
                    chunksize=1,
                )
            pool.close()

        # Flatten the list of lists
        for r in results:
            dihedrals.extend(r)

    dihedrals = torch.LongTensor(dihedrals)
    return dihedrals.view(-1, 4).to(graph.train_args.device)


def process_impropers(args):
    i, neighbor_list, bonds_set = args
    impropers = deque()

    if not isinstance(neighbor_list, list):
        return impropers  # Skip if not a list

    neighbor_count = len(neighbor_list)

    if neighbor_count == 3:
        pairs = itertools.combinations(neighbor_list, 2)
        if all(
            (pair not in bonds_set and (pair[1], pair[0]) not in bonds_set)
            for pair in pairs
        ):
            impropers.append([i] + neighbor_list)

    elif neighbor_count == 4:
        for combination in itertools.combinations(neighbor_list, 3):
            impropers.append([i] + list(combination))

    return list(impropers)


def build_impropers(graph, bonds, neighbors):
    # # old impl
    # neigh = copy.deepcopy(neighbors)
    # _impropers = copy.deepcopy(neighbors)
    # impropers = []
    # for i in range(len(_impropers)):
    #     if len(_impropers[i]) == 3:
    #         neighbors = _impropers[i]
    #         pairs = list(itertools.combinations(neighbors, 2))
    #         is_improper = True
    #         for pair in pairs:
    #             if list(pair) in list(bonds):
    #                 is_improper = False
    #                 break
    #         if is_improper:
    #             impropers.append([i] + _impropers[i])
    #     elif len(_impropers[i]) == 4:
    #         neighbors = _impropers[i]
    #         pairs = list(
    #             itertools.combinations(neighbors, 3)
    #         )  # get the 4 different impropers from 4-coordinated atoms
    #         for pair in pairs:
    #             impropers.append([i] + list(pair))
    # impropers = torch.LongTensor(impropers)
    # impropers = impropers.view(-1, 4)

    if isinstance(bonds, torch.Tensor):
        bonds_set = set(map(tuple, bonds.tolist()))
    else:
        bonds_set = set(map(tuple, bonds)) if bonds else set()

    n_threads = int(os.environ.get("OMP_NUM_THREADS", 1))

    if n_threads == 1:
        impropers = deque()  # Using deque for efficient append operations

        for i, neighbor_list in enumerate(neighbors):
            impropers.extend(process_impropers((i, neighbor_list, bonds_set)))

    else:
        with Pool(n_threads) as pool:
            data = [(i, neighbors[i], bonds_set) for i in range(len(neighbors))]
            results = pool.map(
                process_impropers,
                data,
                # chunksize=(len(neighbors) // (n_threads * 2) or 1),
                chunksize=1,
            )

        # Flatten the result list
        impropers = list(itertools.chain.from_iterable(results))

    impropers = torch.LongTensor(impropers).view(-1, 4)

    return impropers[
        :, [0, 1, 2, 3]
    ].to(
        graph.train_args.device
    )  # switch so that the center atom is the one-th index ("j" index of lammps improper documentation: https://docs.lammps.org/improper_class2.html)


def process_unique_combinations(args):
    angle, bonds_set, device = args
    comb = torch.combinations(angle, r=2).to(device)
    mask = torch.tensor(
        [tuple(item) not in bonds_set for item in comb.tolist()],
        dtype=torch.bool,
        device=device,
    )
    unique_comb = comb[mask]
    return unique_comb.tolist()


def build_pairs(graph, bonds, angles, dihedrals, dih_pair=False):
    atom_num = graph.atom_num
    device = graph.train_args.device

    # Initialize pair matrix
    pairs = torch.eye(atom_num, dtype=torch.long, device="cpu")

    # Convert bonds to set of tuples for efficient membership checking
    bonds_set = set(map(tuple, bonds.tolist()))

    # Initialize pair list with bonds
    pair_list = bonds.tolist()

    # Parallel processing of unique combinations from angles
    n_threads = int(os.environ.get("OMP_NUM_THREADS", 1))

    if n_threads > 1:
        with Pool(n_threads) as pool:
            data = [(angle, bonds_set, device) for angle in angles]
            pool_results = pool.map(
                process_unique_combinations,
                data,
                chunksize=(len(angles) // (n_threads * 2) or 1),
            )
        for result in pool_results:
            pair_list.extend(result)
    else:
        for angle in angles:
            pair_list.extend(process_unique_combinations((angle, bonds_set, device)))

    # Convert pair_list to tensor for scatter
    pair_tensor = torch.LongTensor(pair_list).to("cpu")

    # Flatten the pair indices for scattering
    ind_1, ind_2 = pair_tensor.t()
    flat_indices = ind_1 * atom_num + ind_2

    # Use scatter to fill the pairs matrix
    pairs_flat = pairs.view(-1).to("cpu")
    pairs_flat.scatter_(
        0,
        flat_indices,
        torch.ones(len(flat_indices), dtype=torch.long, device="cpu"),
    )
    pairs = pairs_flat.view(atom_num, atom_num)

    # Make pairs matrix symmetric
    pairs = pairs + pairs.t()

    # Find all zero pairs (non-existing links) and sort them
    pairs = (
        (pairs == 0)
        .nonzero(as_tuple=False)
        .view(-1, 2)
        .sort(dim=1)[0]
        .unique(dim=0)
        .view(-1, 2)
        .to(device)
    )
    # zero_pairs_sorted = zero_pairs.sort(dim=1)[0].unique(dim=0)

    # Convert to tensor and correct shape
    # pairs = zero_pairs_sorted.to(device)
    f_1_4 = None

    return pairs, f_1_4


############################### INDICES ###############################################################


def index_of(inp, source, max_index):
    """
    This is like np.where(), where you return the indices
    of different values in an array. Where np.where()
    can only find the index of one value at a time,
    this index_of will return a list where each value is the
    corresponding index of where each "inp" is in "source
    """
    # inp = list of atoms in cluster as their atomic num
    # source = list of unique atomic nums in inp, in increasing order
    # max_index = highest atomic number in source + 1
    X = torch.randint(0, 9999999999, (max_index,)).to(
        device=source.device
    )  # creates list with length highest-atomic-number-in-smiles+1 of random ints
    inp = X[
        inp
    ].sum(
        1
    )  # creates a list of length #atoms in cluster but with random ints for each element instead of atomic num
    source = X[
        source
    ].sum(
        1
    )  # list of length number-of-unique-elements containing the random int from above in place where atomic num used to be
    source, sorted_index, inverse = np.unique(
        source.tolist(), return_index=True, return_inverse=True
    )

    index = torch.cat([torch.tensor(source).to(device=inp.device), inp]).unique(
        sorted=True, return_inverse=True
    )[1][-len(inp) :]
    index = torch.tensor(sorted_index).to(device=inp.device)[index]
    return index


def get_bonds_in_angles(graph):
    """
    first we create bonds based on the component atoms inside of an angle,
    so the atoms in an angle are 0,1,2 and the bonds are 0-1 and 1-2.
    then, we compare with the listed bonds to return the correct index of the bonds created through the angles"""
    bonds = graph.angles[:, [0, 1, 1, 2]].view(-1, 2)
    ref_bonds = graph.bonds
    bonds = index_of(bonds, source=ref_bonds, max_index=len(bonds)).view(-1, 2)
    return bonds


def get_bonds_in_dihedrals(graph):
    bonds = graph.dihedrals[:, [0, 1, 1, 2, 2, 3]].view(-1, 2)
    ref_bonds = graph.bonds
    bonds = index_of(bonds, source=ref_bonds, max_index=len(bonds)).view(-1, 3)
    return bonds


def get_angles_in_dihedrals(graph):
    angles = graph.dihedrals[:, [0, 1, 2, 1, 2, 3]].view(-1, 3)
    ref_angles = graph.angles
    angles = index_of(angles, source=ref_angles, max_index=len(angles)).view(-1, 2)
    return angles


def get_bonds_in_impropers(graph):
    bonds = graph.impropers[:, [0, 1, 0, 2, 0, 3]].view(-1, 2)
    ref_bonds = graph.bonds
    bonds = index_of(bonds, source=ref_bonds, max_index=len(bonds)).view(-1, 3)
    return bonds


def get_angles_in_impropers(graph):
    angles = torch.stack(
        [
            graph.impropers[:, [1, 0, 2]],
            graph.impropers[:, [1, 0, 3]],
            graph.impropers[:, [2, 0, 3]],
        ],
        dim=1,
    ).view(-1, 3)
    ref_angles = graph.angles
    angles = index_of(angles, source=ref_angles, max_index=len(angles)).view(-1, 3)
    return angles
