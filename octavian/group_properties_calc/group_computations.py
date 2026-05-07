"""

This file contains numba functions used to compute physical quantities in bulk.

Numba requires quite basic syntax meaning these functions are inherently quite readable.

"""

import numpy as np
from numba import njit, prange, boolean


"""

Physical Quantities

"""

@njit(parallel=True)
def compute_centre_of_mass(pos, vel, mass, group_idx, n_groups):
    com_pos = np.zeros((n_groups, 3))
    com_vel = np.zeros((n_groups, 3))
    total_mass = np.zeros(n_groups)
    
    for i in prange(len(mass)):
        g = group_idx[i]
        m = mass[i]
        total_mass[g] += m
        for d in range(3):
            com_pos[g, d] += pos[i, d] * m
            com_vel[g, d] += vel[i, d] * m
    
    for g in range(n_groups):
        if total_mass[g] > 0:
            for d in range(3):
                com_pos[g, d] /= total_mass[g]
                com_vel[g, d] /= total_mass[g]
    
    return com_pos, com_vel, total_mass

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
def compute_radial_quantiles(radius, mass, group_idx, n_groups, quantiles):

    counts = np.zeros(n_groups, dtype=np.int64)
    for i in range(len(mass)):
        counts[group_idx[i]] += 1

    start = np.zeros(n_groups, dtype=np.int64)
    for g in range(1, n_groups):
        start[g] = start[g-1] + counts[g-1]

    idx_sorted = np.empty(len(mass), dtype=np.int64)
    pos = np.zeros(n_groups, dtype=np.int64)
    for i in range(len(mass)):
        g = group_idx[i]
        idx_sorted[start[g] + pos[g]] = i
        pos[g] += 1

    result = np.full((n_groups, len(quantiles)), np.nan)
    for g in range(n_groups):
        s = start[g]
        e = s + counts[g]
        if s == e:
            continue
        indices = idx_sorted[s:e]
        r = radius[indices]
        m = mass[indices]
        order = np.argsort(r)
        r = r[order]
        m = m[order]
        cum = np.cumsum(m)
        frac = cum / cum[-1]
        for q in range(len(quantiles)):
            idx = np.searchsorted(frac, quantiles[q])
            if idx < len(r):
                result[g, q] = r[idx]

    return result


@njit(parallel=True)
def compute_radial_quantiles_and_rmax(radius, mass, group_idx, n_groups, quantiles):
    counts = np.zeros(n_groups, dtype=np.int64)
    for i in range(len(mass)):
        counts[group_idx[i]] += 1

    start = np.zeros(n_groups, dtype=np.int64)
    for g in range(1, n_groups):
        start[g] = start[g-1] + counts[g-1]

    idx_sorted = np.empty(len(mass), dtype=np.int64)
    pos = np.zeros(n_groups, dtype=np.int64)
    for i in range(len(mass)):
        g = group_idx[i]
        idx_sorted[start[g] + pos[g]] = i
        pos[g] += 1

    quantile_result = np.full((n_groups, len(quantiles)), np.nan)
    rmax = np.full(n_groups, np.nan)

    for g in prange(n_groups):
        s = start[g]
        e = s + counts[g]
        if s == e:
            continue
        indices = idx_sorted[s:e]
        r = radius[indices]
        m = mass[indices]
        order = np.argsort(r)
        r = r[order]
        m = m[order]
        cum = np.cumsum(m)
        total_mass = cum[-1]
        if total_mass <= 0:
            continue
        frac = cum / total_mass
        for q in range(len(quantiles)):
            idx = np.searchsorted(frac, quantiles[q])
            if idx < len(r):
                quantile_result[g, q] = r[idx]
        rmax[g] = r[-1]

    return quantile_result, rmax


@njit(parallel=True)
def compute_gas_scalar_sums(
    unique_groups,
    start,
    end,
    masses,
    metallicities,
    sfrs,
    temperatures,
    rhos,
    mass_HI,
    mass_H2,
    dust_masses,
    n_groups,
    nhlim,
):
    group_mass = np.zeros(n_groups)
    gas_HI = np.zeros(n_groups)
    gas_H2 = np.zeros(n_groups)
    dust = np.zeros(n_groups)
    ndust = np.zeros(n_groups, dtype=np.int64)
    sfr = np.zeros(n_groups)
    metal_mass = np.zeros(n_groups)
    metal_sfr = np.zeros(n_groups)
    temp_mass = np.zeros(n_groups)
    cgm_mass = np.zeros(n_groups)
    cgm_temp_mass = np.zeros(n_groups)
    cgm_temp_metal = np.zeros(n_groups)
    cgm_metal_mass = np.zeros(n_groups)

    for gi in prange(len(unique_groups)):
        group = unique_groups[gi]
        s = start[gi]
        e = end[gi]
        for i in range(s, e):
            m = masses[i]
            z = metallicities[i]
            sf = sfrs[i]
            temp = temperatures[i]
            rho = rhos[i]

            group_mass[group] += m
            gas_HI[group] += mass_HI[i]
            gas_H2[group] += mass_H2[i]
            sfr[group] += sf
            metal_mass[group] += z * m
            metal_sfr[group] += z * sf
            temp_mass[group] += temp * m

            if rho >= nhlim:
                dust[group] += dust_masses[i]
                if dust_masses[i] > 0:
                    ndust[group] += 1
            else:
                cgm_mass[group] += m
                cgm_temp_mass[group] += temp * m
                cgm_temp_metal[group] += temp * m * z
                cgm_metal_mass[group] += z * m

    return (
        group_mass,
        gas_HI,
        gas_H2,
        dust,
        ndust,
        sfr,
        metal_mass,
        metal_sfr,
        temp_mass,
        cgm_mass,
        cgm_temp_mass,
        cgm_temp_metal,
        cgm_metal_mass,
    )


@njit(parallel=True)
def compute_star_scalar_sums(
    unique_groups,
    start,
    end,
    masses,
    metallicities,
    ages,
    n_groups,
):
    total_mass = np.zeros(n_groups)
    metal_mass = np.zeros(n_groups)
    age_mass = np.zeros(n_groups)
    age_metal = np.zeros(n_groups)
    young_mass = np.zeros(n_groups)

    for gi in prange(len(unique_groups)):
        group = unique_groups[gi]
        s = start[gi]
        e = end[gi]
        for i in range(s, e):
            m = masses[i]
            z = metallicities[i]
            age = ages[i]
            total_mass[group] += m
            metal_mass[group] += z * m
            age_mass[group] += age * m
            age_metal[group] += age * m * z
            if age < 0.1:
                young_mass[group] += m

    return total_mass, metal_mass, age_mass, age_metal, young_mass


@njit
def compute_central_galaxy_flags(halo_positions, stellar_mass, galaxy_ids, n_halos):
    central = np.zeros(len(galaxy_ids), dtype=boolean)
    central_by_halo = np.full(n_halos, -1, dtype=np.int64)
    central_index = np.full(n_halos, -1, dtype=np.int64)
    max_mass = np.full(n_halos, -np.inf)

    for i in range(len(galaxy_ids)):
        h = halo_positions[i]
        if h < 0 or h >= n_halos:
            continue
        if stellar_mass[i] > max_mass[h]:
            max_mass[h] = stellar_mass[i]
            central_index[h] = i
            central_by_halo[h] = galaxy_ids[i]

    for h in range(n_halos):
        i = central_index[h]
        if i >= 0:
            central[i] = True

    return central, central_by_halo


@njit(parallel=True)
def compute_aperture_component_properties(
    neighbor_offsets,
    neighbor_indices,
    particle_masses,
    particle_codes,
    particle_velocities,
    include_matrix,
):
    n_galaxies = len(neighbor_offsets) - 1
    n_outputs = include_matrix.shape[0]

    output_mass = np.zeros((n_galaxies, n_outputs))
    output_count = np.zeros((n_galaxies, n_outputs), dtype=np.int64)
    mean_momentum = np.zeros((n_galaxies, n_outputs, 3))
    momentum_var = np.zeros((n_galaxies, n_outputs, 3))
    sigma = np.zeros((n_galaxies, n_outputs))

    for g in prange(n_galaxies):
        s = neighbor_offsets[g]
        e = neighbor_offsets[g + 1]

        for ni in range(s, e):
            p = neighbor_indices[ni]
            code = particle_codes[p]
            mass = particle_masses[p]
            if mass <= 0:
                continue
            for out in range(n_outputs):
                if include_matrix[out, code]:
                    output_mass[g, out] += mass
                    output_count[g, out] += 1
                    for d in range(3):
                        mean_momentum[g, out, d] += mass * particle_velocities[p, d]

        for out in range(n_outputs):
            count = output_count[g, out]
            if count > 0:
                for d in range(3):
                    mean_momentum[g, out, d] /= count

        for ni in range(s, e):
            p = neighbor_indices[ni]
            code = particle_codes[p]
            mass = particle_masses[p]
            if mass <= 0:
                continue
            for out in range(n_outputs):
                if include_matrix[out, code]:
                    for d in range(3):
                        delta = mass * particle_velocities[p, d] - mean_momentum[g, out, d]
                        momentum_var[g, out, d] += delta * delta

        for out in range(n_outputs):
            count = output_count[g, out]
            if count > 0 and output_mass[g, out] > 0:
                mean_mass = output_mass[g, out] / count
                sig2 = 0.0
                for d in range(3):
                    axis_sigma = np.sqrt(momentum_var[g, out, d] / count) / mean_mass
                    sig2 += axis_sigma * axis_sigma
                sigma[g, out] = np.sqrt(sig2 / 3.0)

    return output_mass, sigma


@njit
def compute_virial_quantities(radius, mass, group_idx, n_groups, rhocrit, factors):

    volume_factor = 4.0 / 3.0 * np.pi

    counts = np.zeros(n_groups, dtype=np.int64)
    for i in range(len(mass)):
        counts[group_idx[i]] += 1

    start = np.zeros(n_groups, dtype=np.int64)
    for g in range(1, n_groups):
        start[g] = start[g-1] + counts[g-1]

    idx_sorted = np.empty(len(mass), dtype=np.int64)
    pos = np.zeros(n_groups, dtype=np.int64)
    for i in range(len(mass)):
        g = group_idx[i]
        idx_sorted[start[g] + pos[g]] = i
        pos[g] += 1

    result_r = np.full((n_groups, len(factors)), np.nan)
    result_m = np.full((n_groups, len(factors)), np.nan)

    for g in range(n_groups):
        s = start[g]
        e = s + counts[g]
        if s == e:
            continue
        indices = idx_sorted[s:e]
        r = radius[indices]
        m = mass[indices]
        order = np.argsort(r)
        r = r[order]
        m = m[order]
        cumulative = np.cumsum(m)

        for i in range(len(r)):
            if r[i] > 0:
                overdensity = cumulative[i] / (volume_factor * r[i]**3) / rhocrit
                for f in range(len(factors)):
                    if overdensity >= factors[f]:
                        result_r[g, f] = r[i]
                        result_m[g, f] = cumulative[i]

    return result_r, result_m
