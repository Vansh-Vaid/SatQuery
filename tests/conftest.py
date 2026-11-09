from __future__ import annotations

from datetime import date

import pytest

from satquery.demo import synth as S


@pytest.fixture(scope="session")
def labels():
    lab = S.base_layout(128, 128)
    return lab, S.evolve(lab)


@pytest.fixture(scope="session")
def s2(labels):
    return S.render_s2(labels[0], seed=1, acquired=date(2024, 6, 1), name="s2_t1.tif")


@pytest.fixture(scope="session")
def s2_t2(labels):
    return S.render_s2(labels[1], seed=2, gain=1.05, acquired=date(2025, 6, 5), name="s2_t2.tif")


@pytest.fixture(scope="session")
def s2_cloudy(labels):
    return S.render_s2(labels[0], seed=3, cloud=S.cloud_mask(128, 128), acquired=date(2024, 6, 1), name="s2_c.tif")


@pytest.fixture(scope="session")
def s1(labels):
    return S.render_s1(labels[0], seed=4, acquired=date(2024, 6, 2), name="s1_t1.tif")


@pytest.fixture(scope="session")
def s1_t2(labels):
    return S.render_s1(labels[1], seed=5, acquired=date(2025, 6, 6), name="s1_t2.tif")
