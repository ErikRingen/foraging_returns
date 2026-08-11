"""Tests for prior widening used by the wide-prior sensitivity refit."""

import pytest
from foraging_model.priors import get_priors, scale_priors


class TestScalePriors:
    def test_normal_sigma_scaled(self):
        base, wide = get_priors(), scale_priors(2.0)
        for name in ('m0', 'k0', 'b0', 'effort_age', 'eta_mu0'):
            assert wide[name]['sigma'] == pytest.approx(2 * base[name]['sigma'])
            assert wide[name]['mu'] == base[name]['mu']

    def test_gamma_mean_preserved_sd_doubled(self):
        base, wide = get_priors(), scale_priors(2.0)
        b, w = base['shape'], wide['shape']
        assert w['alpha'] / w['beta'] == pytest.approx(b['alpha'] / b['beta'])
        assert (w['alpha'] ** 0.5 / w['beta']
                ) == pytest.approx(2 * b['alpha'] ** 0.5 / b['beta'])

    def test_exponential_scale_doubled(self):
        base, wide = get_priors(), scale_priors(2.0)
        assert wide['sigma_re']['lam'] == pytest.approx(base['sigma_re']['lam'] / 2)

    def test_gender_zsn_sigma_doubled(self):
        base, wide = get_priors(), scale_priors(2.0)
        assert wide['sigma_gender_skill']['sigma'] == pytest.approx(
            2 * base['sigma_gender_skill']['sigma'])

    def test_lkj_eta_unchanged(self):
        assert scale_priors(2.0)['lkj_eta'] == get_priors()['lkj_eta']

    def test_factor_one_is_identity(self):
        assert scale_priors(1.0) == get_priors()


class TestScopedScaling:
    def test_koster_only_widens_only_koster_priors(self):
        from foraging_model.priors import KOSTER_DERIVED
        base, wide = get_priors(), scale_priors(2.0, only=KOSTER_DERIVED)
        for name in KOSTER_DERIVED:
            assert wide[name]['sigma'] == pytest.approx(2 * base[name]['sigma'])
        for name in ('effort_age', 'shape', 'sigma_re', 'sigma_gender_skill',
                     'intercept_mu', 'intercept_success'):
            assert wide[name] == base[name]
