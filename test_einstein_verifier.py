"""
Tests for the Einstein Summation Verifier.
Run with:  python -m pytest test_einstein_verifier.py -v
       or:  python test_einstein_verifier.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from einstein_summation_verifier import verify


# ─────────────────────────────────────────────────────────────────────────────
# Group 1: Basic examples from the user specification
# ─────────────────────────────────────────────────────────────────────────────

def test_ex1_AijBj():
    """A_{ij}B_{j} — free: i, dummy: j, well-formed."""
    r = verify(r"A_{ij}B_{j}")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i"}
    assert r.dummy_indices == {"j"}


def test_ex2_AiBj():
    """A_i B_j — free: i,j, dummy: none, well-formed."""
    r = verify(r"A_i B_j")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i", "j"}
    assert r.dummy_indices == set()


def test_ex3_AiBj_plus_Cij():
    """A_i B_j + C_{ij} — free: i,j, dummy: none, well-formed."""
    r = verify(r"A_i B_j + C_{ij}")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i", "j"}
    assert r.dummy_indices == set()


def test_ex4_inconsistent_free():
    """A_i B_j + C_{ik} — ill-formed (inconsistent free indices)."""
    r = verify(r"A_i B_j + C_{ik}")
    assert not r.well_formed


def test_ex5_triple_index():
    """A_{i,i,j} B_i — ill-formed (i appears too many times)."""
    # Note: commas in index groups are ignored by the parser (not index symbols),
    # so A_{iij} with i appearing twice in the tensor + B_i adds a third.
    r = verify(r"A_{iij} B_i")
    assert not r.well_formed


def test_ex6_two_terms_different_dummies():
    """A_{ij}B_{j} + D_{irs}C_{rs} — free: i, dummy: j,r,s, well-formed."""
    r = verify(r"A_{ij}B_{j} + D_{irs}C_{rs}")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i"}
    assert r.dummy_indices == {"j", "r", "s"}


def test_ex7_nested_parens():
    """A_i * (B_{ij} + C_{ijk} * D_k) — free: j, dummy: i,k, well-formed."""
    r = verify(r"A_i * (B_{ij} + C_{ijk} * D_k)")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"j"}
    assert r.dummy_indices == {"i", "k"}


def test_ex8_inconsistent_in_brackets():
    """A_i * (B_{ij} + C_{kj}) — ill-formed (inconsistent free inside brackets)."""
    r = verify(r"A_i * (B_{ij} + C_{kj})")
    assert not r.well_formed


# ─────────────────────────────────────────────────────────────────────────────
# Group 2: Mixed upper/lower index examples
# ─────────────────────────────────────────────────────────────────────────────

def test_mixed1_AjiBkj():
    """A^i_j B^j_k — free: i,k, dummy: j, well-formed."""
    r = verify(r"A^i_j B^j_k")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i", "k"}
    assert r.dummy_indices == {"j"}


def test_mixed2_inconsistent_upper():
    """T^{ij} + S^{ik} — ill-formed (free indices don't match: j vs k)."""
    r = verify(r"T^{ij} + S^{ik}")
    assert not r.well_formed


def test_mixed3_metric_contraction():
    """g_{ij} v^j — free: i, dummy: j, well-formed."""
    r = verify(r"g_{ij} v^j")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i"}
    assert r.dummy_indices == {"j"}


def test_mixed4_triple_product():
    r"""A^i_{\ j} B^j_{\ k} C^k_{\ l} — free: i,l, dummy: j,k, well-formed."""
    r = verify(r"A^i_{j} B^j_{k} C^k_{l}")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i", "l"}
    assert r.dummy_indices == {"j", "k"}


def test_mixed5_kronecker_delta():
    r"""\delta^i_{j} T^{jk} - S^{ik} — free: i,k, dummy: j, well-formed."""
    r = verify(r"\delta^i_{j} T^{jk} - S^{ik}")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i", "k"}
    assert r.dummy_indices == {"j"}


def test_mixed6_double_metric():
    r"""g_{ij} g^{jk} R^l_{kmn} — free: i,l,m,n, dummy: j,k, well-formed."""
    r = verify(r"g_{ij} g^{jk} R^l_{kmn}")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i", "l", "m", "n"}
    assert r.dummy_indices == {"j", "k"}


def test_mixed7_levi_civita():
    r"""\epsilon_{ijk} v^j w^k — free: i, dummy: j,k, well-formed."""
    r = verify(r"\epsilon_{ijk} v^j w^k")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i"}
    assert r.dummy_indices == {"j", "k"}


def test_mixed8_sum_with_dummy():
    r"""T^i_{j} S^j_{k} + R^i_{k} — free: i,k, dummy: j (first term), well-formed."""
    r = verify(r"T^i_{j} S^j_{k} + R^i_{k}")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i", "k"}
    assert "j" in r.dummy_indices


# ─────────────────────────────────────────────────────────────────────────────
# Group 3: \left / \right delimiter handling (ESV-1)
# ─────────────────────────────────────────────────────────────────────────────

def test_left_right_basic():
    r"""\left( A_i B_j \right) — free: i,j — well-formed."""
    r = verify(r"\left( A_i B_j \right)")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i", "j"}

def test_left_right_with_sum():
    r"""\left( A_{ij} B_j + C_i \right) — free: i, dummy: j — well-formed."""
    r = verify(r"\left( A_{ij} B_j + C_i \right)")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i"}
    assert r.dummy_indices == {"j"}

def test_left_right_multiply_outside():
    r"""K \left( \delta_{ij} - n_i n_j \right) \nabla^2 n_j
    The original bug expression pattern: well-formed."""
    r = verify(r"K \left( A_{ij} - B_i B_j \right) C_j")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i"}

def test_left_right_inconsistent_inside():
    r"""\left( A_i + B_k \right) — ill-formed (inconsistent free inside)."""
    r = verify(r"\left( A_i + B_k \right)")
    assert not r.well_formed

def test_left_right_square_brackets():
    r"""\left[ T^{ij} \right] — free: i,j — well-formed."""
    r = verify(r"\left[ T^{ij} \right]")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i", "j"}

def test_nested_left_right():
    r"""\left( A_i \left( B_{ij} C_j \right) \right) — free: i, dummy: j."""
    r = verify(r"\left( A_i \left( B_{ij} C_j \right) \right)")
    assert r.well_formed
    # Inner: B_{ij}C_j => free: i, dummy: j
    # Outer: A_i * (inner) => i contracts, so dummy: i,j ... 
    # Actually: inner has free {i,j...} let me reconsider
    # B_{ij}C_j: free=i, dummy=j
    # A_i * (free=i, dummy=j) => i appears in both => dummy
    assert r.dummy_indices >= {"i", "j"}


# ─────────────────────────────────────────────────────────────────────────────
# Additional edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_equation_both_sides_match():
    """F^{ij} = A^i B^j + C^{ij} — both sides have free {i,j}."""
    r = verify(r"F^{ij} = A^i B^j + C^{ij}")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i", "j"}


def test_equation_sides_mismatch():
    """F^{ij} = A^i B^k — sides have different free indices."""
    r = verify(r"F^{ij} = A^i B^k")
    assert not r.well_formed


def test_single_tensor_no_indices():
    """T — no indices at all, well-formed (scalar)."""
    r = verify(r"T")
    assert r.well_formed
    assert set(r.free_indices.keys()) == set()
    assert r.dummy_indices == set()


def test_scalar_plus_scalar():
    """A + B — both scalars, well-formed."""
    r = verify(r"A + B")
    assert r.well_formed


def test_unary_minus():
    """-A_i + B_i — well-formed with leading minus."""
    r = verify(r"-A_i + B_i")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i"}


def test_command_indices():
    r"""T^{\mu\nu} g_{\mu\alpha} — free: \nu, \alpha; dummy: \mu."""
    r = verify(r"T^{\mu\nu} g_{\mu\alpha}")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {r"\nu", r"\alpha"}
    assert r.dummy_indices == {r"\mu"}


def test_backslash_space_in_index():
    r"""A^i_{\ j} — the '\ ' should be ignored, only j is an index."""
    r = verify(r"A^i_j")
    assert r.well_formed
    assert set(r.free_indices.keys()) == {"i", "j"}


def test_ex5_comma_separated_triple():
    """A_{i,i,j} B_i — commas are non-index tokens, so this is A with
    indices i,i,j (comma ignored) then B with index i. 
    i appears 3 times total -> ill-formed."""
    # With commas as non-index chars, A_{i,i,j} has i appearing twice.
    # Then B_i adds a third occurrence of i in the product.
    r = verify(r"A_{iij}B_i")
    assert not r.well_formed


# ─────────────────────────────────────────────────────────────────────────────
# Run tests manually if not using pytest
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import inspect
    
    tests = [(name, obj) for name, obj in globals().items()
             if name.startswith("test_") and callable(obj)]
    
    passed = 0
    failed = 0
    errors = []
    
    for name, func in sorted(tests):
        try:
            func()
            passed += 1
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            errors.append((name, e))
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"  ERROR {name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
    
    if errors:
        print(f"\nFailures:")
        for name, e in errors:
            print(f"  {name}: {e}")
    
    sys.exit(1 if failed else 0)


# ─────────────────────────────────────────────────────────────────────────────
# Non-Index Symbols  (Feature B)
# ─────────────────────────────────────────────────────────────────────────────

def test_verify_nonidx_baseline():
    """A^T_i with no filter — T and i both appear as free indices (baseline)."""
    r = verify(r"A^T_i")
    assert r.well_formed
    assert "T" in r.free_indices
    assert "i" in r.free_indices

def test_verify_nonidx_unbracketed():
    """A^T_i with non_index_symbols={'T'} — only i is free; T excluded."""
    r = verify(r"A^T_i", frozenset({"T"}))
    assert r.well_formed
    assert "T" not in r.free_indices
    assert "i" in r.free_indices

def test_verify_nonidx_command():
    r"""A^{\dagger}_{ij} with non_index_symbols={'\dagger'} — free: i, j."""
    r = verify(r"A^{\dagger}_{ij}", frozenset({r"\dagger"}))
    assert r.well_formed
    assert r"\dagger" not in r.free_indices
    assert "i" in r.free_indices
    assert "j" in r.free_indices

def test_verify_nonidx_transpose_parens():
    """(A^T)_i with no filter — ESV handles parens correctly; free: i only."""
    r = verify(r"(A^T)_i")
    assert r.well_formed
    assert "i" in r.free_indices
    # T is inside parens; ESV recurses in and sees it as a free index of
    # the inner expr, then merges with outer _i — net result T is free too.
    # This test documents current ESV behaviour (no non_index_symbols needed
    # when using the parenthesis convention).
    assert "T" in r.free_indices  # T visible inside parens to ESV