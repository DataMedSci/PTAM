import numpy as np
import mpmath as mp  # its part of sympy, offers high-precision floating-point arithmetic


class ERSCalc(object):
    """
    A 'calculator' class for stopping power and range of protons
    """
    alpha_cm_MeV = 0.0022
    p = 1.77

    @classmethod
    def p_alpha_cm_MeV(cls):
        """
        Helper variable p*alpha^(1/p)
        """
        return cls.p * cls.alpha_cm_MeV ** (1.0 / cls.p)

    @classmethod
    def range_cm(cls, energy_MeV):
        """
        Bragg-Kleeman rule for energy-range relationship
        Equation (8) in [1]
        """
        return cls.alpha_cm_MeV * energy_MeV ** cls.p

    @classmethod
    def stop_pow_MeV_cm(cls, resid_range_cm):
        """
        Bortfeld approximation for stopping power of protons
        """
        return (resid_range_cm ** (1.0 / cls.p - 1.0)) / cls.p_alpha_cm_MeV


class ExactSpecialFunction(object):

    @staticmethod
    def parabolic_integral(nu, t):
        """
        evaluates exp(-x^2/4)*D_(-nu)(x) with precision defined by mpmath library
        (by default 15 decimal places or 53 bits)
        """
        return np.array([float(mp.exp(-(v ** 2) / 4.0) * mp.pcfd(-nu, v)) for v in t])

    @staticmethod
    def exponent_square(x, y):
        """
        evaluates exp(-(x+y)^2/8) with precision defined by mpmath library
        (by default 15 decimal places or 53 bits)
        """
        return np.array([float(mp.exp(-((xn + yn) ** 2) / 8.0)) for xn, yn in zip(x, y)])


class RangeStraggling(object):

    def __init__(self, energy_MeV):
        self.energy_MeV = energy_MeV
        self.range_cm = ERSCalc.range_cm(energy_MeV)

    def _sigma_cm(self, sigma_energy_MeV):
        # range straggling of monoenergetical protons, see Appendix [1]
        sigma_mono_cm = 0.012 * self.range_cm ** 0.935

        # range equivalent of energy straggline, equation (A2) in [1]
        sigma_r_cm = sigma_energy_MeV
        sigma_r_cm *= ERSCalc.alpha_cm_MeV
        sigma_r_cm *= ERSCalc.p
        sigma_r_cm *= (self.energy_MeV ** (ERSCalc.p - 1.0))

        # total sigma, equation (A3) in [1]
        sigma_cm = (sigma_mono_cm ** 2 + sigma_r_cm ** 2) ** 0.5

        return sigma_cm

    @staticmethod
    def sigma_cm(energy_MeV, sigma_energy_MeV):
        """
        TODO
        """
        return RangeStraggling(energy_MeV)._sigma_cm(sigma_energy_MeV)


class GeneralModel(object):
    r_cm = 2e-4  # 2 um regularisation

    def __init__(self, energy_MeV, sigma_energy_MeV, z_cm):
        self.energy_MeV = energy_MeV
        self.range_cm = ERSCalc.range_cm(energy_MeV)

        self.sigma_cm = RangeStraggling.sigma_cm(energy_MeV, sigma_energy_MeV)

        # zeta variable introduced for equation (A8) in [1]
        self.zeta = (z_cm - self.range_cm) / self.sigma_cm

        # xi variable introduced for equation (A11) in [1]
        self.xi = (z_cm - self.range_cm - self.r_cm) / self.sigma_cm


class BortfeldModel(GeneralModel):
    rho_g_cm3 = 1.0

    beta_cm = 0.012  # 1/cm

    gamma = 0.6

    def dose_MeV_g(self, fluence_cm2, eps=0.1):
        """dose in [MeV/g]"""

        A = self.sigma_cm ** (1.0 / ERSCalc.p)
        A *= mp.gamma(1.0 / ERSCalc.p)
        A /= mp.sqrt(2.0 * np.pi)
        A /= self.rho_g_cm3
        A /= ERSCalc.p_alpha_cm_MeV()
        A /= (1.0 + self.beta_cm * self.range_cm)

        B = self.beta_cm / ERSCalc.p + self.gamma * self.beta_cm + eps / self.range_cm

        result = fluence_cm2 * A

        bracket_part1 = ExactSpecialFunction.parabolic_integral(1.0 / ERSCalc.p, self.zeta) / self.sigma_cm
        bracket_part2 = B * ExactSpecialFunction.parabolic_integral(1.0 + (1.0 / ERSCalc.p), self.zeta)

        result *= (bracket_part1 + bracket_part2)

        return result

    @classmethod
    def dose_Gy(cls, fluence_cm2, energy_MeV, sigma_energy_MeV, z_cm, eps=0.1):
        """dose in [Gy]"""

        model = BortfeldDose(energy_MeV, sigma_energy_MeV, z_cm)

        dose_MeV_g = model.dose_MeV_g(fluence_cm2, eps)

        return 1.6021766e-10 * dose_MeV_g


class WilkensModel(GeneralModel):
    """
    Analytical LET model implementation following
    [1] Wilkens JJ, Oelfke U. "Analytical linear energy transfer calculations for proton therapy"
    Med Phys. 2003 May;30(5):806-15. (DOI: 10.1118/1.1567852)
    """

    def _d_tilde(self, nu):
        """helper function defined by equation (10) in [1]"""
        part1 = ExactSpecialFunction.parabolic_integral(nu, self.xi)
        part2 = ExactSpecialFunction.parabolic_integral(nu, self.zeta)
        return part1 - part2

    def let_d_MeV_cm(self):
        """
        dose averaged LET in [MeV/cm]
        """
        # main part of <S>_z as in quation (10) or (A12) in [1]
        q = 1.0 + 1.0 / ERSCalc.p
        mean_s_z_part = (self.sigma_cm ** q) * float(mp.gamma(q)) * self._d_tilde(q)

        mean_s_z_part -= self.r_cm * ((0.5 * self.r_cm) ** (1.0 / ERSCalc.p)) * ExactSpecialFunction.exponent_square(
            self.zeta, self.xi)

        # main part of <S2>_z as in quation (10) or (A14) in [1]
        r = 2.0 / ERSCalc.p
        mean_s2_z_part = (self.sigma_cm ** r) * float(mp.gamma(r)) * self.d_tilde(r)
        mean_s2_z_part -= 2.0 * ((0.5 * self.r_cm) ** r) * ExactSpecialFunction.exponent_square(self.zeta, self.xi)

        # factor part of <S2>_z divided by <S>_z
        const_factor_MeV_cm = 1.0 / (ERSCalc.p_alpha_cm_MeV() * (2.0 - ERSCalc.p))

        # result
        result = const_factor_MeV_cm * mean_s2_z_part / mean_s_z_part

        # filling nonsense values (outside model domain) with np.nan
        # zeta > 0 is equivalent to z_cm > r_cm (points behind the range)
        result[self.zeta > 0] = np.nan

        return result

    def let_t_MeV_cm(self):
        """
        track averaged LET in [MeV/cm]
        """

        # main part of <S>_z as in equation (10) or (A12) in [1]
        q = 1.0 + 1.0 / ERSCalc.p
        mean_s_z_part = (self.sigma_cm ** q) * float(mp.gamma(q)) * self._d_tilde(q)
        mean_s_z_part -= self.r_cm * ((0.5 * self.r_cm) ** (1.0 / ERSCalc.p)) * ExactSpecialFunction.exponent_square(
            self.zeta, self.xi)

        # main part of Q_z as in equation (10) or (A8) in [1]
        q_z_part = ExactSpecialFunction.parabolic_integral(1, self.zeta)

        # factor part of <S2>_z divided by <S>_z
        const_factor_MeV_cm = 1.0 / (self.sigma_cm * self.r_cm * ERSCalc.alpha_cm_MeV ** (1.0 / ERSCalc.p))

        # result
        result = const_factor_MeV_cm * mean_s_z_part / q_z_part

        # filling nonsense values (outside model domain) with np.nan
        # zeta > 0 is equivalent to z_cm > r_cm (points behind the range)
        result[self.zeta > 0] = np.nan

        return result

    @classmethod
    def let_d_keV_um(cls, energy_MeV, sigma_energy_MeV, z_cm):
        """
        dose averaged LET in [keV/um]
        """
        model = WilkensLET(energy_MeV, sigma_energy_MeV, z_cm)
        return model.let_d_MeV_cm() * 0.1

    @classmethod
    def let_t_keV_um(cls, energy_MeV, sigma_energy_MeV, z_cm):
        """
        track averaged LET in [keV/um]
        """
        model = WilkensLET(energy_MeV, sigma_energy_MeV, z_cm)
        return model.let_t_MeV_cm() * 0.1
