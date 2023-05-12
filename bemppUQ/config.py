import numpy as np

config = {}

# Section A
config["polarization"] = np.array([1.0 + 1j, 2.0, -1.0 - 1.0 / 3.0 * 1j])
config["direction"] = np.array(
    [1.0 / np.sqrt(14), 2.0 / np.sqrt(14), 3.0 / np.sqrt(14)], dtype="float64"
)

# physical constants
config["eps_ext"] = 8.8542 * 10 ** (-12)
config["mu_ext"] = 1.2566 * 10 ** (-6)
config["c"] = 1 / np.sqrt(config["mu_ext"] * config["eps_ext"])

# parameters
# MIE
config["k_ext"] = 5.0
config["eps_rel"] = 1.9
config["mu_rel"] = 1.0

# Teflon
# config['k_ext'] = 1.047197551196598
# config['eps_rel'] = 2.1
# config['mu_rel'] = 1.

config["k_int"] = config["k_ext"] * np.sqrt(config["eps_rel"] * config["mu_rel"])
config["lambda"] = 2 * np.pi / config["k_ext"]
config["frequency"] = config["k_ext"] * config["c"] / 2.0 / np.pi

# Ferrite
# config['k_ext'] =
# config['eps_rel'] = 2.5
# config['mu_rel'] = 1.6


# for incident_z
config["polarization"] = np.array([0.0, 0.0, 1.0])
config["direction"] = np.array([1.0, 0.0, 0.0], dtype="float64")

# HPs
config["solver"] = "direct"  # or gmres
# options for gmres
config["restart"] = 10000
config["maxiter"] = 10000
config["tolerance"] = 1e-6

# options for assembly (1e-3 / hmat)
config["spaces"] = "maxwell_primal"  # or maxwell
config["osrc"] = True

# options for the Far Field at z=0
config["number_of_angles"] = 3601

config["angles"] = np.pi * np.linspace(0, 2, config["number_of_angles"])
config["far_field_points"] = np.array(
    [
        np.cos(config["angles"]),
        np.sin(config["angles"]),
        np.zeros(config["number_of_angles"]),
    ]
)