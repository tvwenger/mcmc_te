"""
model.py
Forward model Galactic electron temperature gradient.
Trey V. Wenger
"""

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd

import utils


def main(
    fname,
    prior_Rgal=5.0,
    prior_Te_Rgal_offset=[4350.0, 250.0],
    prior_Te_Rgal_slope=[375.0, 50.0],
    prior_Te_sigma=100.0,
):
    """
    Fit model to data using MCMC.

    Inputs:
        fname :: str
            Data filename
        prior_Rgal :: float
            Width of Galactocentric radius prior
        prior_Te_Rgal_offset :: Iterable[float]
            Mean and width of Te vs. Rgal offset (K)
        prior_Te_Rgal_slope :: Iterable[float]
            Mean and width of Te vs. Rgal slope (K/kpc)
        prior_Te_sigma :: float
            Width of intrinsic Te scatter (K)
    """
    data = pd.loadcsv(fname)

    with pm.Model(coords={"data": data.index}) as model:
        # Minimum Galactocentric radius (kpc)
        is_q3_q4 = np.cos(np.deg2rad(data["glong"])) < 0.0
        Rmin = utils.__R0 * np.abs(np.sin(np.deg2rad("glong")))
        Rmin[is_q3_q4] = utils.__R0

        # Galactocentric radius offset (kpc)
        Rgal_off_norm = pm.HalfNormal("Rgal_off_norm", sigma=1.0, dims="data")
        Rgal = pm.Deterministic("Rgal", Rmin + prior_Rgal * Rgal_off_norm, dims="data")

        # Electron temperature gradient offset (K)
        Te_Rgal_offset_norm = pm.Normal("Te_Rgal_offset_norm", mu=0.0, sigma=1.0)
        Te_Rgal_offset = pm.Deterministic(
            "Te_Rgal_offset",
            prior_Te_Rgal_offset[0] + prior_Te_Rgal_offset[1] * Te_Rgal_offset_norm,
        )

        # Electron temperature gradient slope (K)
        Te_Rgal_slope_norm = pm.Normal("Te_Rgal_slope_norm", mu=0.0, sigma=1.0)
        Te_Rgal_slope = pm.Deterministic(
            "Te_Rgal_slope",
            prior_Te_Rgal_slope[0] + prior_Te_Rgal_offset[1] * Te_Rgal_offset_norm,
        )


def model(data):
    """
    Generates an instance of the forward model applied to a given data point.

    Inputs:
        data :: pd.Series
            Data for a single HII region
    """
    # Quadrant 3 or 4
    # is_q3_q4 = np.cos(np.deg2rad(data["glong"])) < 0.0

    # Outer galaxy
    # sin_glong = np.sin(np.deg2rad(data["glong"]))
    # is_outer_galaxy = is_q3_q4 or np.sign(sin_glong) != np.sign(data["vlsr"])
    # kdar = ["F"] if is_q3_q4 else ["F", "N"]
    kdar = ["N", "F"]

    # line width in kHz
    data["fwhm_kHz"] = 1000.0 * data["line_freq"] * data["fwhm"] / c.c.to("km/s").value

    with pm.Model(coords={"kdar": kdar}) as model:
        # Kinematic distance ambiguity
        # if is_q3_q4:
        #    kdar_w = pm.Data("kdar_w", np.array([1.0]), dims="kdar")
        # else:
        #    kdar_w = pm.Dirichlet("kdar_w", a=[0.5, 0.5], dims="kdar")

        # Distance (kpc)
        distance = Maxwell(
            "distance",
            a=5.0,
            dims="kdar",
            # transform=pm.distributions.transforms.Ordered(),
            initval=np.array([5.0, 10.0]),
        )

        # Galactocentric radius (kpc)
        Rgal = pm.Deterministic(
            "Rgal",
            pt.sqrt(
                utils.__R0**2.0
                + distance**2.0
                - 2.0
                * utils.__R0
                * distance
                * np.abs(np.cos(np.deg2rad(data["glong"])))
            ),
            dims="kdar",
        )

        # KDAR weight
        kdar_w = pm.Dirichlet("kdar_w", a=[1.0, 1.0], dims="kdar")

        # LSR velocity (km/s)
        vlsr = utils.reid19_vlsr(data["glong"], data["glat"], Rgal)

        # LSR velocity likelihood
        _ = pm.NormalMixture(
            "vlsr", mu=vlsr, sigma=data["e_vlsr"], w=kdar_w, observed=data["vlsr"]
        )

        # Electron temperature (K)
        log10_te_norm = pm.Normal("log10_te_norm", mu=0.0, sigma=1.0, dims="kdar")
        log10_te = pm.Deterministic("log10_te", 3.5 + 0.5 * log10_te_norm, dims="kdar")
        te = 10.0**log10_te

        # Electron temperature likelihood
        if not np.isnan(data["te"]):
            _ = pm.NormalMixture(
                "te", mu=te, sigma=data["e_te"], w=kdar_w, observed=data["te"]
            )

        # Electron density
        log10_n_norm = pm.Normal("log10_n_norm", mu=0.0, sigma=1.0, dims="kdar")
        log10_n = pm.Deterministic("log10_n", 1.5 + 0.15 * log10_n_norm, dims="kdar")

        # Stromgren radius factor
        Rs_factor = pm.Beta("Rs_factor", alpha=3.0, beta=1.0, dims="kdar")
        Rs_angular = pt.clip(Rs_factor, 0.01, 1.0) * data["radius"]

        # Stromgren radius (pc)
        log10_Rs = pm.Deterministic(
            "log10_Rs",
            3.0 + pt.log10(distance) + pt.log10(Rs_angular) - np.log10(206265.0),
            dims="kdar",
        )

        # Emission measure (pc cm-6)
        log10_em = pm.Deterministic(
            "log10_em",
            log10_Rs + 2.0 * log10_n + np.log10(2.0),
            dims="kdar",
        )

        # Ionizing photon rate (s-1)
        _ = pm.Deterministic(
            "log10_q", 3.0 * log10_Rs + 2.0 * log10_n + 14.522, dims="kdar"
        )

        # RRL optical depth
        log10_tau_line = pm.Deterministic(
            "log10_tau_line",
            3.2833 - 2.5 * log10_te + log10_em - np.log10(data["fwhm_kHz"]),
            dims="kdar",
        )

        # Undiluted RRL brightness (mJy/beam)
        line_mu = (
            2.0
            * data["beam_area"]
            / 206265.0**2.0
            * (c.k_B / c.c**2.0).to("mJy MHz-2 K-1").value
            * data["line_freq"] ** 2.0
            * te
            * (1.0 - np.exp(-(10.0**log10_tau_line)))
        )

        # Beam dilution
        source_area = np.pi * Rs_angular**2.0 / (4.0 * np.log(2.0))
        beam_dilution = source_area / data["beam_area"]
        beam_dilution = pt.clip(beam_dilution, 0.0, 1.0)

        # Diluted RRL brightness (mJy/beam)
        line_mu = pm.Deterministic(
            "line_mu",
            line_mu * beam_dilution,
            dims="kdar",
        )
        # Line likelihood
        _ = pm.NormalMixture(
            "line",
            mu=line_mu,
            sigma=data["e_line"],
            w=kdar_w,
            observed=data["line"],
        )
    return model
