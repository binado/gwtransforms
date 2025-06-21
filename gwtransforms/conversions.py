from gwtransforms.backend import np


def component_masses_to_total_mass(mass_1, mass_2):
    return mass_1 + mass_2


def component_masses_to_mass_ratio(mass_1, mass_2):
    return mass_2 / mass_1


def primary_mass_and_mass_ratio_to_component_masses(mass_1, mass_ratio):
    return mass_1, mass_1 * mass_ratio


def component_masses_to_total_mass_and_mass_ratio(mass_1, mass_2):
    return component_masses_to_total_mass(
        mass_1, mass_2
    ), component_masses_to_mass_ratio(mass_1, mass_2)


def component_masses_to_symmetric_mass_ratio(mass_1, mass_2):
    return mass_1 * mass_2 / (mass_1 + mass_2) ** 2


def component_masses_to_chirp_mass(mass_1, mass_2):
    total_mass = component_masses_to_total_mass(mass_1, mass_2)
    symmetric_mass_ratio = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)
    return total_mass * symmetric_mass_ratio**0.6


def component_masses_to_chirp_mass_and_symmetric_mass_ratio(mass_1, mass_2):
    chirp_mass = component_masses_to_chirp_mass(mass_1, mass_2)
    symmetric_mass_ratio = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)
    return chirp_mass, symmetric_mass_ratio


def mass_ratio_to_symmetric_mass_ratio(mass_ratio):
    return mass_ratio / (1 + mass_ratio) ** 2


def symmetric_mass_ratio_to_mass_ratio(symmetric_mass_ratio):
    nu = symmetric_mass_ratio
    return -1 + 1 / (2 * nu) + np.sqrt(1 - 4 * nu) / (2 * nu)


def chirp_mass_and_symmetric_mass_ratio_to_total_mass_and_mass_ratio(
    chirp_mass, symmetric_mass_ratio
):
    total_mass = chirp_mass / symmetric_mass_ratio**0.6
    mass_ratio = symmetric_mass_ratio_to_mass_ratio(symmetric_mass_ratio)
    return total_mass, mass_ratio


def total_mass_and_mass_ratio_to_chirp_mass_and_symmetric_mass_ratio(
    total_mass, mass_ratio
):
    symmetric_mass_ratio = mass_ratio / (1 + mass_ratio) ** 2
    chirp_mass = total_mass * symmetric_mass_ratio**0.6
    return chirp_mass, symmetric_mass_ratio


def total_mass_and_mass_ratio_to_component_masses(total_mass, mass_ratio):
    return total_mass / (1 + mass_ratio), total_mass * mass_ratio / (1 + mass_ratio)


def chirp_mass_and_symmetric_mass_ratio_to_component_masses(
    chirp_mass, symmetric_mass_ratio
):
    total_mass, mass_ratio = (
        chirp_mass_and_symmetric_mass_ratio_to_total_mass_and_mass_ratio(
            chirp_mass, symmetric_mass_ratio
        )
    )
    return total_mass_and_mass_ratio_to_component_masses(total_mass, mass_ratio)
