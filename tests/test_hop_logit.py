"""Tests for the logit metrics of `hop_logit.py`."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/scripts/f10"))
from research.scripts.f10.hop_all import claim  # noqa: E402
from research.scripts.f10.hop_logit import js_divergence, kl_divergence, pooled_margin, softcap  # noqa: E402


def test_softcap_is_off_without_a_cap_and_bounded_with_a_cap():
    x = torch.tensor([1.0, 500.0])
    assert torch.equal(softcap(x, None), x)
    assert softcap(x, 30.0).max() <= 30.0
    assert abs(float(softcap(x, 30.0)[0]) - 30 * math.tanh(1 / 30)) < 1e-5


def test_js_is_0_for_equal_inputs_and_ln2_for_disjoint_inputs():
    p, q = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    assert abs(float(js_divergence(p, p))) < 1e-6
    assert abs(float(js_divergence(p, q)) - math.log(2)) < 1e-5


def test_kl_of_a_known_pair():
    p, q = torch.tensor([0.5, 0.5]), torch.tensor([0.25, 0.75])
    assert abs(float(kl_divergence(p, q)) - (0.5 * math.log(2) + 0.5 * math.log(2 / 3))) < 1e-5


def test_a_claim_blocks_another_worker_until_it_is_stale(tmp_path):
    path = tmp_path / "shard-00000"
    assert claim(path, "a")
    assert not claim(path, "b")
    assert claim(path, "a")  # a requeued job takes its own claim again
    assert claim(path, "b", stale=0.0)
    assert path.read_text() == "b"


def test_pooled_margin_counts_the_forms_with_a_space():
    # Yes, yes, " Yes", " yes", then the no forms. Most mass sits on the forms with a space.
    x = torch.log(torch.tensor([0.01, 0.01, 0.01, 0.57, 0.01, 0.01, 0.01, 0.37]))
    assert abs(float(pooled_margin(x)) - math.log(0.60 / 0.40)) < 1e-5


def test_pooled_margin_uses_both_cases_and_ignores_a_shift():
    x = torch.log(torch.tensor([0.1, 0.3, 0.2, 0.4]))
    assert abs(float(pooled_margin(x)) - math.log(0.4 / 0.6)) < 1e-5
    assert abs(float(pooled_margin(x + 7.0)) - float(pooled_margin(x))) < 1e-5


def test_split_gives_98_ntas_and_is_fixed():
    import pandas as pd
    from research.scripts.f10.analyze_logit import split_ntas

    ntas = pd.Series([f"N{i:03d}" for i in range(197) for _ in range(3)])
    a, b = split_ntas(ntas), split_ntas(ntas)
    assert a.sum() == 98 * 3 and (a == b).all()
    assert ntas[a].nunique() + ntas[~a].nunique() == 197 and not set(ntas[a]) & set(ntas[~a])


def test_cluster_r_gives_the_pearson_r_and_a_lower_t_for_clusters():
    import numpy as np
    from research.scripts.f10.analyze_logit import cluster_r

    rng = np.random.default_rng(1)
    u = np.repeat(rng.normal(size=60), 3)
    x, y = u + 0.2 * rng.normal(size=180), u + 0.2 * rng.normal(size=180)
    r, t, _ = cluster_r(x, y, np.repeat(np.arange(60), 3))
    assert abs(r - np.corrcoef(x, y)[0, 1]) < 1e-9
    assert 0 < t < r * np.sqrt(178) / np.sqrt(1 - r * r) * 3


def test_holm_of_a_known_case():
    import numpy as np
    from research.scripts.f10.analyze_logit import holm

    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
