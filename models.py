import numpy as np
import mpmath as mp  # its part of sympy, offers high-precision floating-point arithmetic


class ERSCalc(object):
    """
    A 'calculator' class for stopping power and range of protons
    """
    alpha_cm_MeV = 0.0022
    p = 1.77

    @classmethod
    def range_cm(cls, energy_MeV):
        """
        Bragg-Kleeman rule for energy-range relationship
        Equation (8) in [1]
        """
        return cls.alpha_cm_MeV * energy_MeV ** cls.p

    @classmethod
    def stop_pow_MeV_cm(cls, resid_range_cm):
        """Bortfeld approximation for stopping power of protons"""
        return (resid_range_cm ** (1.0 / cls.p - 1.0)) / (cls.p * cls.alpha_cm_MeV ** (1.0 / cls.p))


class WilkensLET(object):
    """
    Analytical LET model implementation following
    [1] Wilkens JJ, Oelfke U. "Analytical linear energy transfer calculations for proton therapy"
    Med Phys. 2003 May;30(5):806-15. (DOI: 10.1118/1.1567852)
    """
    r_cm = 2e-4  # 2 um regularisation

    @staticmethod
    def parabolic_integral(nu, t):
        """
        evaluates precisely e^(-x^2/4)*D_(-nu)(x)
        """
        return np.array([float(mp.exp(-(v ** 2) / 4.0) * mp.pcfd(-nu, v)) for v in t])

    @staticmethod
    def exponent_square(x, y):
        """
        evaluates precisely e^(-(x+y)^2/8)
        """
        return np.array([float(mp.exp(-((xn + yn) ** 2) / 8.0)) for xn, yn in zip(x, y)])

    @classmethod
    def d_tilde(cls, nu, xi, zeta):
        """helper function defined by equation (10) in [1]"""
        part1 = cls.parabolic_integral(nu, xi)
        part2 = cls.parabolic_integral(nu, zeta)
        return part1 - part2

    @classmethod
    def szx(cls, energy_MeV, sigma_energy_MeV, z_cm):
        """
        calculates tuple sigma_cm, zeta, xi where
          - sigma_cm is total sigma, defined by equation (A3) in [1]
          - xi is a variable introduced for equation (A11) in [1]
          - zeta is a variable introduced for equation (A8) in [1]
        """

        # range
        range_cm = ERSCalc.range_cm(energy_MeV)

        # range straggling of monoenergetical protons, see Appendix [1]
        sigma_mono_cm = 0.012 * range_cm ** 0.935

        # range equivalent of energy straggline, equation (A2) in [1]
        sigma_r_cm = sigma_energy_MeV
        sigma_r_cm *= ERSCalc.alpha_cm_MeV
        sigma_r_cm *= ERSCalc.p
        sigma_r_cm *= (energy_MeV ** (ERSCalc.p - 1.0))

        # total sigma, equation (A3) in [1]
        sigma_cm = (sigma_mono_cm ** 2 + sigma_r_cm ** 2) ** 0.5

        # zeta variable introduced for equation (A8) in [1]
        zeta = (z_cm - range_cm) / sigma_cm

        # xi variable introduced for equation (A11) in [1]
        xi = (z_cm - range_cm - cls.r_cm) / sigma_cm

        return sigma_cm, zeta, xi

    @classmethod
    def let_d_MeV_cm(cls, energy_MeV, sigma_energy_MeV, z_cm):
        """dose averaged LET in [MeV/cm]
        """
        sigma_cm, zeta, xi = cls.szx(energy_MeV, sigma_energy_MeV, z_cm)

        # main part of <S>_z as in quation (10) or (A12) in [1]
        q = 1.0 + 1.0 / ERSCalc.p
        mean_s_z_part = (sigma_cm ** q) * float(mp.gamma(q)) * cls.d_tilde(q, xi, zeta)
        mean_s_z_part -= cls.r_cm * ((0.5 * cls.r_cm) ** (1.0 / ERSCalc.p)) * cls.exponent_square(zeta, xi)

        # main part of <S2>_z as in quation (10) or (A14) in [1]
        r = 2.0 / ERSCalc.p
        mean_s2_z_part = (sigma_cm ** r) * float(mp.gamma(r)) * cls.d_tilde(r, xi, zeta)
        mean_s2_z_part -= 2.0 * ((0.5 * cls.r_cm) ** r) * cls.exponent_square(zeta, xi)

        # factor part of <S2>_z divided by <S>_z
        const_factor_MeV_cm = 1.0 / (ERSCalc.p * (2.0 - ERSCalc.p) * ERSCalc.alpha_cm_MeV ** (1.0 / ERSCalc.p))

        # result
        result = const_factor_MeV_cm * mean_s2_z_part / mean_s_z_part

        # filling nonsense values (outside model domain) with np.nan
        # zeta > 0 is equivalent to z_cm > r_cm (points behind the range)
        result[zeta > 0] = np.nan

        return result

    @classmethod
    def let_t_MeV_cm(cls, energy_MeV, sigma_energy_MeV, z_cm):
        """track averaged LET
        """
        sigma_cm, zeta, xi = cls.szx(energy_MeV, sigma_energy_MeV, z_cm)

        # main part of <S>_z as in equation (10) or (A12) in [1]
        q = 1.0 + 1.0 / ERSCalc.p
        mean_s_z_part = (sigma_cm ** q) * float(mp.gamma(q)) * cls.d_tilde(q, xi, zeta)
        mean_s_z_part -= cls.r_cm * ((0.5 * cls.r_cm) ** (1.0 / ERSCalc.p)) * cls.exponent_square(zeta, xi)

        # main part of Q_z as in equation (10) or (A8) in [1]
        q_z_part = cls.parabolic_integral(1, zeta)

        # factor part of <S2>_z divided by <S>_z
        const_factor_MeV_cm = 1.0 / (sigma_cm * cls.r_cm * ERSCalc.alpha_cm_MeV ** (1.0 / ERSCalc.p))

        # result
        result = const_factor_MeV_cm * mean_s_z_part / q_z_part

        # filling nonsense values (outside model domain) with np.nan
        # zeta > 0 is equivalent to z_cm > r_cm (points behind the range)
        result[zeta > 0] = np.nan

        return result

    @classmethod
    def let_d_keV_um(cls, energy_MeV, sigma_energy_MeV, z_cm):
        """dose averaged LET in [keV/um]"""
        return cls.let_d_MeV_cm(energy_MeV, sigma_energy_MeV, z_cm) * 0.1

    @classmethod
    def let_t_keV_um(cls, energy_MeV, sigma_energy_MeV, z_cm):
        """track averaged LET in [keV/um]"""
        return cls.let_t_MeV_cm(energy_MeV, sigma_energy_MeV, z_cm) * 0.1
