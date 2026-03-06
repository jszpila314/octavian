"""

This file contains numba functions used to compute physical quantities in bulk.

Numba requires quite basic syntax meaning these functions are inherently quite readable.

"""

import numpy as np
from numba import njit, prange, boolean

@njit(parallel=True)
def compute_angular_momentum(pos_rel, vel_rel, mass, group_idx, n_groups):
    L = np.zeros((n_groups, 3))
    ktot_sum = np.zeros(n_groups)

    for i in prange(len(mass)):
        g = group_idx[i]
        rx, ry, rz = pos_rel[i, 0], pos_rel[i, 1], pos_rel[i, 2]
        vx, vy, vz = vel_rel[i, 0], vel_rel[i, 1], vel_rel[i, 2]
        m = mass[i]

        px, py, pz = m * vx, m * vy, m * vz
        L[g, 0] += ry * pz - rz * py
        L[g, 1] += rz * px - rx * pz
        L[g, 2] += rx * py - ry * px

        ktot_sum[g] += 0.5 * m * (vx**2 + vy**2 + vz**2)

    return L, ktot_sum

@njit(parallel=True)
def compute_rotation_quantities(pos_rel, vel_rel, mass, group_idx, L_group, n_groups):
    counter_rotating_mass = np.zeros(n_groups)
    krot_sum = np.zeros(n_groups)
    ktot_sum = np.zeros(n_groups)
    
    for i in prange(len(mass)):
        g = group_idx[i]
        rx, ry, rz = pos_rel[i, 0], pos_rel[i, 1], pos_rel[i, 2]
        vx, vy, vz = vel_rel[i, 0], vel_rel[i, 1], vel_rel[i, 2]
        m = mass[i]
        
        px, py, pz = m * vx, m * vy, m * vz
        Lx = ry * pz - rz * py
        Ly = rz * px - rx * pz
        Lz = rx * py - ry * px
        
        Lgx, Lgy, Lgz = L_group[g, 0], L_group[g, 1], L_group[g, 2]
        L_dot = Lx * Lgx + Ly * Lgy + Lz * Lgz
        
        if L_dot < 0:
            counter_rotating_mass[g] += m
        
        cx = ry * Lgz - rz * Lgy
        cy = rz * Lgx - rx * Lgz
        cz = rx * Lgy - ry * Lgx
        rz_cyl = np.sqrt(cx**2 + cy**2 + cz**2)
        
        ktot = 0.5 * m * (vx**2 + vy**2 + vz**2)
        ktot_sum[g] += ktot
        
        if rz_cyl > 0.0:
            krot_sum[g] += 0.5 * (L_dot / rz_cyl)**2 / m
    
    return counter_rotating_mass, krot_sum, ktot_sum

@njit
def compute_radial_quantiles(radii, mass, group_idx, n_groups, quantiles):
    group_mass_total = np.zeros(n_groups)
    for i in range(len(mass)):
        group_mass_total[group_idx[i]] += mass[i]
    
    cumulative = np.zeros(n_groups)
    result = np.full((n_groups, len(quantiles)), np.nan)
    found = np.zeros((n_groups, len(quantiles)), dtype=boolean)
    
    for i in range(len(mass)):
        g = group_idx[i]
        cumulative[g] += mass[i]
        frac = cumulative[g] / group_mass_total[g]
        
        for q in range(len(quantiles)):
            if not found[g, q] and frac >= quantiles[q]:
                result[g, q] = radii[i]
                found[g, q] = True
    
    return result

@njit
def compute_virial_quantities(radii, mass, group_idx, n_groups, rhocrit, factors):
    volume_factor = 4.0 / 3.0 * np.pi
    cumulative = np.zeros(n_groups)
    
    result_r = np.full((n_groups, len(factors)), np.nan)
    result_m = np.full((n_groups, len(factors)), np.nan)
    
    for i in range(len(mass)):
        g = group_idx[i]
        cumulative[g] += mass[i]
        r = radii[i]
        
        if r > 0:
            overdensity = cumulative[g] / (volume_factor * r**3) / rhocrit
            
            for f in range(len(factors)):
                if overdensity >= factors[f]:
                    result_r[g, f] = r
                    result_m[g, f] = cumulative[g]
    
    return result_r, result_m
