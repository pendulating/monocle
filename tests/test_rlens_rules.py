"""LRP rules for the lens backward pass. See research/monocle/rlens_rules.py.

The rules must keep the forward value EXACT and change the backward. Every
test here holds one half of that contract. No gemma weights are needed: the
rules are tensor functions, and the installer is tested on a fake model.
"""
import pytest
import torch
from torch import nn

from research.monocle import rlens_rules as rr


def plain_norm(self, hidden_states):
    """The stock gemma-4 `_norm`, for a value and gradient comparison."""
    mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
    return hidden_states * torch.pow(mean_squared, -0.5)


class Eps:
    eps = 1e-6


# ---------------------------------------------------------------------------
# The half rule
# ---------------------------------------------------------------------------
class TestHalfRule:
    def test_the_value_is_exact(self):
        a = torch.randn(5, 7)
        b = torch.randn(5, 7)
        assert torch.equal(rr.half_product(a, b), a * b)

    def test_the_value_is_exact_in_bfloat16(self):
        a = torch.randn(64, 64).bfloat16()
        b = torch.randn(64, 64).bfloat16()
        assert torch.equal(rr.half_product(a, b), a * b)

    def test_each_branch_gets_half(self):
        a = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        b = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        rr.half_product(a, b).sum().backward()
        assert torch.allclose(a.grad, 0.5 * b.detach())
        assert torch.allclose(b.grad, 0.5 * a.detach())

    def test_a_plain_product_gives_the_whole_relevance(self):
        a = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        b = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        (a * b).sum().backward()
        assert torch.allclose(a.grad, b.detach())


# ---------------------------------------------------------------------------
# The identity rule
# ---------------------------------------------------------------------------
class TestIdentityRule:
    def test_the_value_is_exact(self):
        x = torch.randn(9, 11)
        y = nn.functional.gelu(x, approximate="tanh")
        slope = rr.identity_slope(x, y, 0.5)
        assert torch.equal(rr.straight_through(x, y, slope), y)

    def test_the_backward_is_the_slope(self):
        x = torch.randn(6, dtype=torch.float64, requires_grad=True)
        y = x * 3.0
        slope = torch.full_like(x, 0.25)
        rr.straight_through(x, y, slope).sum().backward()
        assert torch.allclose(x.grad, slope)

    def test_the_slope_is_the_ratio(self):
        x = torch.tensor([2.0, -4.0, 0.5])
        y = torch.tensor([1.0, -1.0, 2.0])
        got = rr.identity_slope(x, y, 99.0)
        assert torch.allclose(got, torch.tensor([0.5, 0.25, 4.0]))

    def test_the_guard_holds_at_zero(self):
        x = torch.tensor([0.0, 1e-9, 1.0])
        y = torch.tensor([0.0, 5e-10, 0.5])
        got = rr.identity_slope(x, y, 0.5)
        assert torch.allclose(got, torch.tensor([0.5, 0.5, 0.5]))

    def test_the_slope_never_carries_a_gradient(self):
        x = torch.randn(5, requires_grad=True)
        y = nn.functional.gelu(x)
        assert not rr.identity_slope(x, y, 0.5).requires_grad

    @pytest.mark.parametrize("act,want", [
        (lambda t: nn.functional.gelu(t, approximate="tanh"), 0.5),
        (torch.nn.functional.silu, 0.5),
        (torch.nn.functional.relu, 0.5),
        (torch.tanh, 1.0),
    ])
    def test_the_slope_at_zero(self, act, want):
        assert rr.slope_at_zero(act) == pytest.approx(want, abs=1e-5)

    def test_it_differs_from_the_true_derivative(self):
        x = torch.tensor([-2.0], dtype=torch.float64, requires_grad=True)
        y = nn.functional.gelu(x, approximate="tanh")
        y.sum().backward()
        true = float(x.grad)
        ratio = float(y.detach() / x.detach())
        assert abs(true - ratio) > 0.1


# ---------------------------------------------------------------------------
# The norm rule
# ---------------------------------------------------------------------------
class TestNormRule:
    def test_the_value_is_exact(self):
        h = torch.randn(2, 3, 16)
        assert torch.equal(rr.lrp_norm(Eps(), h), plain_norm(Eps(), h))

    def test_the_denominator_is_constant(self):
        h = torch.randn(1, 2, 8, dtype=torch.float64, requires_grad=True)
        rr.lrp_norm(Eps(), h).sum().backward()
        scale = torch.pow(h.pow(2).mean(-1, keepdim=True) + Eps.eps, -0.5)
        assert torch.allclose(h.grad, scale.expand_as(h).detach())

    def test_the_stock_gradient_is_different(self):
        h = torch.randn(1, 2, 8, dtype=torch.float64, requires_grad=True)
        plain_norm(Eps(), h).sum().backward()
        stock = h.grad.clone()
        h.grad = None
        rr.lrp_norm(Eps(), h).sum().backward()
        assert not torch.allclose(stock, h.grad)


# ---------------------------------------------------------------------------
# The softmax rule
# ---------------------------------------------------------------------------
class TestSoftmaxRule:
    def test_the_value_is_exact(self):
        x = torch.randn(3, 5)
        ruled = rr.make_softmax(nn.functional.softmax)
        assert torch.equal(ruled(x, dim=-1),
                           nn.functional.softmax(x, dim=-1))

    def test_it_cuts_the_gradient(self):
        x = torch.randn(3, 5, requires_grad=True)
        ruled = rr.make_softmax(nn.functional.softmax)
        assert not ruled(x, dim=-1).requires_grad

    def test_it_counts_its_calls(self):
        report = {"softmax_calls": 0}
        ruled = rr.make_softmax(nn.functional.softmax, report)
        ruled(torch.randn(2, 3), dim=-1)
        ruled(torch.randn(2, 3), dim=-1)
        assert report["softmax_calls"] == 2

    def test_it_passes_a_dtype_through(self):
        x = torch.randn(3, 5).bfloat16()
        ruled = rr.make_softmax(nn.functional.softmax)
        assert ruled(x, dim=-1, dtype=torch.float32).dtype == torch.float32


# ---------------------------------------------------------------------------
# A fake model, to test the installer
# ---------------------------------------------------------------------------
class FakeRMSNorm(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, hidden_states):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states):
        return self._norm(hidden_states) * self.weight


class WrongRMSNorm(FakeRMSNorm):
    """A norm the rule does NOT reproduce. It stands for a later gemma."""

    def _norm(self, hidden_states):
        centered = hidden_states - hidden_states.mean(-1, keepdim=True)
        mean_squared = centered.pow(2).mean(-1, keepdim=True) + self.eps
        return centered * torch.pow(mean_squared, -0.5)


class FakeMLP(nn.Module):
    def __init__(self, dim=8, inner=16):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inner, bias=False)
        self.up_proj = nn.Linear(dim, inner, bias=False)
        self.down_proj = nn.Linear(inner, dim, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x))
                              * self.up_proj(x))


class PlainMLP(nn.Module):
    """No GeGLU parts, so the installer must skip it."""

    def __init__(self, dim=8):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(x)


class Config:
    def __init__(self, impl="eager", sub=None):
        self._attn_implementation = impl
        if sub is not None:
            self.text_config = sub


class FakeModel(nn.Module):
    def __init__(self, norm_cls=FakeRMSNorm, impl="eager", n=2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.ModuleDict({"norm": norm_cls(), "mlp": FakeMLP()})
             for _ in range(n)])
        self.spare = PlainMLP()
        self.config = Config(impl)

    def forward(self, x):
        for block in self.blocks:
            x = x + block["mlp"](block["norm"](x))
        return x


@pytest.fixture
def model():
    torch.manual_seed(0)
    return FakeModel()


# ---------------------------------------------------------------------------
# Where the rules go
# ---------------------------------------------------------------------------
class TestDiscovery:
    def test_it_finds_the_norm_class(self, model):
        assert rr.classes_with_suffix(model, "RMSNorm") == [FakeRMSNorm]

    def test_it_finds_both_mlp_classes(self, model):
        got = rr.classes_with_suffix(model, "MLP")
        assert set(got) == {FakeMLP, PlainMLP}

    def test_it_returns_each_class_once(self, model):
        assert len(rr.classes_with_suffix(model, "RMSNorm")) == 1

    def test_one_instance_matches_the_exact_class(self, model):
        assert type(rr.one_instance(model, FakeMLP)) is FakeMLP

    def test_one_instance_raises_when_absent(self, model):
        with pytest.raises(LookupError):
            rr.one_instance(model, WrongRMSNorm)


class TestAttentionKinds:
    def test_a_string(self):
        assert rr.attention_kinds(Config("eager")) == {"eager"}

    def test_a_nested_config(self):
        cfg = Config("eager", sub=Config("sdpa"))
        assert rr.attention_kinds(cfg) == {"eager", "sdpa"}

    def test_a_dict(self):
        cfg = Config({"text": "eager", "vision": "sdpa"})
        assert rr.attention_kinds(cfg) == {"eager", "sdpa"}

    def test_nothing_named(self):
        assert rr.attention_kinds(Config(None)) == set()


# ---------------------------------------------------------------------------
# The context manager
# ---------------------------------------------------------------------------
class TestInstall:
    def test_it_reports_what_it_patched(self, model):
        with rr.lrp_rules(model) as report:
            assert report["norm"] == 1
            assert report["mlp"] == 1
            assert report["softmax"] is True
            assert report["zero_slope"] == [0.5]

    def test_it_skips_a_module_with_no_geglu(self, model):
        before = PlainMLP.forward
        with rr.lrp_rules(model):
            assert PlainMLP.forward is before

    def test_it_restores_everything(self, model):
        norm, mlp = FakeRMSNorm._norm, FakeMLP.forward
        softmax = nn.functional.softmax
        with rr.lrp_rules(model):
            assert FakeRMSNorm._norm is not norm
            assert FakeMLP.forward is not mlp
            assert nn.functional.softmax is not softmax
        assert FakeRMSNorm._norm is norm
        assert FakeMLP.forward is mlp
        assert nn.functional.softmax is softmax

    def test_it_restores_after_an_error(self, model):
        norm = FakeRMSNorm._norm
        with pytest.raises(ValueError):
            with rr.lrp_rules(model):
                raise ValueError("boom")
        assert FakeRMSNorm._norm is norm

    @pytest.mark.parametrize(
        "impl", ["sdpa", "flash_attention_2", "flex_attention"])
    def test_it_refuses_a_fused_attention(self, impl):
        with pytest.raises(RuntimeError, match="eager"):
            with rr.lrp_rules(FakeModel(impl=impl)):
                pass

    def test_it_allows_a_config_that_names_nothing(self):
        with rr.lrp_rules(FakeModel(impl=None)) as report:
            assert report["softmax"] is True

    def test_it_counts_the_softmax_calls_of_a_forward(self, model):
        with rr.lrp_rules(model) as report:
            assert report["softmax_calls"] == 0
            nn.functional.softmax(torch.randn(2, 3), dim=-1)
            assert report["softmax_calls"] == 1

    def test_it_allows_a_non_eager_model_without_the_softmax_rule(self):
        with rr.lrp_rules(FakeModel(impl="sdpa"), softmax=False) as report:
            assert report["softmax"] is False

    def test_it_refuses_a_norm_the_rule_does_not_reproduce(self):
        target = FakeModel(norm_cls=WrongRMSNorm)
        with pytest.raises(RuntimeError, match="forward value"):
            with rr.lrp_rules(target):
                pass

    def test_a_refusal_leaves_nothing_patched(self):
        target = FakeModel(norm_cls=WrongRMSNorm)
        norm, softmax = WrongRMSNorm._norm, nn.functional.softmax
        with pytest.raises(RuntimeError):
            with rr.lrp_rules(target):
                pass
        assert WrongRMSNorm._norm is norm
        assert nn.functional.softmax is softmax

    @pytest.mark.parametrize("flag,site", [
        ("norm", "norm"), ("identity", "mlp"), ("half", "mlp"),
    ])
    def test_a_flag_turns_a_rule_off(self, model, flag, site):
        with rr.lrp_rules(model, **{flag: False}) as report:
            if flag == "norm":
                assert report["norm"] == 0
            else:
                assert report["mlp"] == 1


class TestInstalledBehaviour:
    def test_the_forward_does_not_move(self, model):
        x = torch.randn(1, 3, 8)
        with torch.no_grad():
            before = model(x)
            with rr.lrp_rules(model):
                after = model(x)
        assert torch.equal(before, after)

    def test_the_backward_does_move(self, model):
        x = torch.randn(1, 3, 8, requires_grad=True)
        model(x).sum().backward()
        plain = x.grad.clone()
        x.grad = None
        with rr.lrp_rules(model):
            model(x).sum().backward()
        assert not torch.allclose(plain, x.grad)

    def test_every_rule_alone_moves_the_backward(self, model):
        x = torch.randn(1, 3, 8, requires_grad=True)
        model(x).sum().backward()
        plain = x.grad.clone()
        for flag in ("norm", "identity", "half"):
            off = {"norm": False, "identity": False, "half": False,
                   "softmax": False}
            off[flag] = True
            x.grad = None
            with rr.lrp_rules(model, **off):
                model(x).sum().backward()
            assert not torch.allclose(plain, x.grad), flag


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------
class TestMatchScale:
    def test_it_matches_the_frobenius_norm(self):
        j = {6: torch.randn(4, 4), 20: torch.randn(4, 4)}
        r = {6: j[6] * 0.004, 20: j[20] * 0.5}
        got, scales = rr.match_scale(r, j)
        for layer in j:
            assert float(got[layer].norm()) == pytest.approx(
                float(j[layer].norm()), rel=1e-5)
        assert scales[6] == pytest.approx(250.0, rel=1e-3)

    def test_no_ranking_moves(self):
        j = {6: torch.randn(4, 4)}
        r = {6: torch.randn(4, 4)}
        h = torch.randn(4)
        before = (r[6] @ h).argsort()
        got, _ = rr.match_scale(r, j)
        assert torch.equal((got[6] @ h).argsort(), before)

    def test_it_refuses_a_missing_layer(self):
        with pytest.raises(KeyError):
            rr.match_scale({6: torch.randn(2, 2)}, {12: torch.randn(2, 2)})

    def test_it_refuses_an_empty_fit(self):
        with pytest.raises(ValueError, match="zeros"):
            rr.match_scale({6: torch.zeros(2, 2)}, {6: torch.randn(2, 2)})
