"""The clamp of the F10.3 coordinate swap (CPU)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/f10"))
from name_coord_swap import added_targets, clamp_hook, deleted_targets, matched_basis, swapped_targets  # noqa: E402


def clamp(h, V, P, t):
    return clamp_hook(V, P, t, torch.arange(len(h)), [])(None, None, h[None])[0]


def setup():
    torch.manual_seed(0)
    V = torch.randn(32, 2)
    V = V / V.norm(dim=0)
    return V, torch.linalg.pinv(V), torch.randn(5, 32)


def test_swap_exchanges_the_coordinates():
    V, P, h = setup()
    t = swapped_targets(h, P, 1.0)
    new = clamp(h, V, P, t)
    c, c_new = h @ P.T, new @ P.T
    assert torch.allclose(c_new[:, 0], c[:, 1], atol=1e-4) and torch.allclose(c_new[:, 1], c[:, 0], atol=1e-4)


def test_swap_keeps_the_orthogonal_part():
    V, P, h = setup()
    new = clamp(h, V, P, swapped_targets(h, P, 1.0))
    Q = torch.eye(32) - V @ P
    assert torch.allclose(new @ Q.T, h @ Q.T, atol=1e-4)


def test_clean_targets_are_the_identity():
    V, P, h = setup()
    assert torch.allclose(clamp(h, V, P, h @ P.T), h, atol=1e-4)


def test_targets_come_from_the_clean_pass():
    V, P, h = setup()
    t = swapped_targets(h, P, 2.0)
    assert torch.allclose(clamp(h + 3 * V[:, 0], V, P, t) @ P.T, t, atol=1e-4)


def test_delete_clears_the_name_and_keeps_the_other():
    V, P, h = setup()
    c = h @ P.T
    new = clamp(h, V, P, deleted_targets(h, P)) @ P.T
    assert torch.allclose(new[:, 0], torch.zeros(5), atol=1e-4) and torch.allclose(new[:, 1], c[:, 1], atol=1e-4)


def test_add_copies_the_name_coordinate():
    V, P, h = setup()
    c = h @ P.T
    new = clamp(h, V, P, added_targets(h, P)) @ P.T
    assert torch.allclose(new[:, 0], c[:, 0], atol=1e-4) and torch.allclose(new[:, 1], c[:, 0], atol=1e-4)


def test_matched_basis_keeps_the_edit_size():
    V, P, h = setup()
    R = matched_basis(V, torch.Generator().manual_seed(1))
    delta = swapped_targets(h, P, 1.0) - h @ P.T
    assert torch.allclose((delta @ R.T).norm(dim=1), (delta @ V.T).norm(dim=1), rtol=1e-4)
    assert abs(float(R[:, 0] @ V[:, 0])) < 0.9


def test_replay_keeps_the_size_of_the_logged_edit_on_a_moved_residual():
    V, P, h = setup()
    t = swapped_targets(h, P, 1.0)
    pos = torch.arange(len(h))
    log: list = []
    clamp_hook(V, P, t, pos, [], log=log)(None, None, h[None])
    R = matched_basis(V, torch.Generator().manual_seed(2))
    moved = h + 0.5 * log[0] @ V.T
    assert (t - moved @ P.T).norm() < 0.75 * log[0].norm()  # a fresh clamp here would be smaller
    edited = clamp_hook(V, P, t, pos, [], along=R, replay=log[0])(None, None, moved[None])[0]
    assert torch.allclose((edited - moved).norm(dim=1), (log[0] @ V.T).norm(dim=1), rtol=1e-3)
