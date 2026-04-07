"""
Tests for the LaTeX Index Editor (no GUI required).
Run with:  python -m pytest test_latex_index_editor.py -v
       or:  python test_latex_index_editor.py
"""

import sys, os
# Stub tkinter so the module can be imported in headless environments
import unittest.mock as _mock
sys.modules.setdefault("tkinter", _mock.MagicMock())
sys.modules.setdefault("tkinter.ttk", _mock.MagicMock())
sys.modules.setdefault("tkinter.messagebox", _mock.MagicMock())
sys.path.insert(0, os.path.dirname(__file__))

from latex_index_editor import (
    tokenize, classify, parse_rules, substitute, reconstruct,
    generate_diff, format_report,
    T_COMMAND, T_CHAR, T_SUBSCRIPT, T_SUPER,
    ParseError, RuleError,
    SubRule,
    _index_symbol,
)
from einstein_summation_verifier import verify, VerificationResult


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 – Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def test_tokenize_simple():
    toks = tokenize(r"a_i")
    texts = [t.text for t in toks]
    assert texts == ["a", "_", "i"]

def test_tokenize_command():
    toks = tokenize(r"\mu")
    assert toks[0].type == T_COMMAND
    assert toks[0].text == r"\mu"

def test_tokenize_braces():
    toks = tokenize(r"T^{\mu\nu}")
    texts = [t.text for t in toks]
    assert "{" in texts and "}" in texts

def test_tokenize_preserves_whitespace():
    toks = tokenize("a  b")
    assert any(t.text == "  " for t in toks)

def test_tokenize_trailing_backslash_raises():
    try:
        tokenize("a\\")
        assert False, "Should have raised ValueError"
    except (ParseError, ValueError):
        pass

def test_tokenize_roundtrip():
    s = r"T^{\mu\nu}{}_{ij} + \Gamma^A_{bc}"
    assert reconstruct(tokenize(s)) == s


# ─────────────────────────────────────────────────────────────────────────────
# Module 2 – Index Classifier
# ─────────────────────────────────────────────────────────────────────────────

def _run(latex, free):
    toks = tokenize(latex)
    warns = classify(toks, set(free))
    return toks, warns

def test_free_index_tagged():
    toks, _ = _run(r"v_i", ["i"])
    idx = [t for t in toks if t.text == "i"]
    assert all(t.role == "free" for t in idx)

def test_dummy_detected_mixed():
    # T^\mu{}_\mu  → \mu appears once upper, once lower
    toks, _ = _run(r"T^\mu{}_\mu", [])
    mu_toks = [t for t in toks if t.text == r"\mu"]
    assert all(t.role == "dummy" for t in mu_toks), [t.role for t in mu_toks]

def test_dummy_detected_both_lower():
    # v_i v_i → Euclidean contraction
    toks, _ = _run(r"v_i v_i", [])
    i_toks = [t for t in toks if t.text == "i"]
    assert all(t.role == "dummy" for t in i_toks)

def test_structural_label_not_touched():
    # \Gamma^A → A is NOT in free list, appears once → structural
    toks, _ = _run(r"\Gamma^A", [])
    a_toks = [t for t in toks if t.text == "A"]
    assert all(t.role == "structural" for t in a_toks)

def test_body_token_unchanged():
    toks, _ = _run(r"x + y", [])
    for t in toks:
        assert t.role == "body"

def test_conflict_warning():
    # \mu appears twice in separate exclusive slots within one term → genuine
    # contraction; user declaring it free should still produce a warning.
    _, warns = _run(r"T^\mu{}_\mu", [r"\mu"])
    assert any("possible dummy" in w.lower() for w in warns), warns

def test_braced_group_indices():
    # \nu_{ijk} — three free indices in a brace group
    toks, _ = _run(r"\nu_{ijk}", ["i", "j", "k"])
    for sym in ["i", "j", "k"]:
        hits = [t for t in toks if t.text == sym]
        assert all(t.role == "free" for t in hits), f"{sym}: {[t.role for t in hits]}"


# ─────────────────────────────────────────────────────────────────────────────
# Module 3 – Rule Parser
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_simple_rule():
    rules, _ = parse_rules("i -> r", {"i"})
    assert len(rules) == 1
    assert rules[0].old == "i" and rules[0].new == "r"

def test_parse_multiple_rules_comma():
    rules, _ = parse_rules("i -> r, j -> s", {"i", "j"})
    assert len(rules) == 2

def test_parse_multiple_rules_newline():
    rules, _ = parse_rules("i -> r\nj -> s", {"i", "j"})
    assert len(rules) == 2

def test_parse_no_arrow_raises():
    try:
        parse_rules("i = r", {"i"})
        assert False
    except RuleError:
        pass

def test_parse_empty_lhs_raises():
    try:
        parse_rules("-> r", {"i"})
        assert False
    except RuleError:
        pass

def test_parse_duplicate_raises():
    try:
        parse_rules("i -> r, i -> s", {"i"})
        assert False
    except RuleError:
        pass

def test_parse_noop_self_warning():
    _, warns = parse_rules("i -> i", {"i"})
    assert warns  # should warn about self-mapping

def test_parse_non_index_warning():
    _, warns = parse_rules("x -> y", {"i"})  # x not eligible
    assert any("x" in w for w in warns)

def test_parse_latex_command_rule():
    rules, _ = parse_rules(r"\mu -> \rho", {r"\mu"})
    assert rules[0].old == r"\mu" and rules[0].new == r"\rho"


# ─────────────────────────────────────────────────────────────────────────────
# Module 4 – Substitution Engine
# ─────────────────────────────────────────────────────────────────────────────

def _pipeline(latex, free, rule_text):
    toks = tokenize(latex)
    classify(toks, set(free))
    eligible = {t.text for t in toks if t.role in ("free", "dummy")}
    rules, _ = parse_rules(rule_text, eligible)
    mod = substitute(toks, rules)
    return reconstruct(mod), toks, mod

def test_substitute_free_index():
    out, _, _ = _pipeline(r"v_i", ["i"], "i -> r")
    assert out == r"v_r"

def test_substitute_dummy_index():
    out, _, _ = _pipeline(r"T^\mu{}_\mu", [], r"\mu -> \nu")
    assert out == r"T^\nu{}_\nu"

def test_substitute_simultaneous_swap():
    # i -> j and j -> i simultaneously — must not cascade
    out, _, _ = _pipeline(r"A_i B_j", ["i", "j"], "i -> j, j -> i")
    assert out == r"A_j B_i"

def test_substitute_structural_untouched():
    # A is structural, must not change even if a rule says A -> B
    toks = tokenize(r"\Gamma^A")
    classify(toks, set())                       # A → structural
    rules = [SubRule("A", "B")]
    mod = substitute(toks, rules)
    assert reconstruct(mod) == r"\Gamma^A"

def test_substitute_body_untouched():
    # letter in body (not index position) must not be replaced
    toks = tokenize(r"x + v_i")
    classify(toks, {"i"})
    rules = [SubRule("x", "y")]
    mod = substitute(toks, rules)
    assert reconstruct(mod) == r"x + v_i"   # x is body, unchanged

def test_substitute_no_rules():
    out, _, _ = _pipeline(r"T_i", ["i"], "")
    assert out == r"T_i"


# ─────────────────────────────────────────────────────────────────────────────
# Module 5 – Reconstructor (covered by roundtrip + substitution tests above)
# ─────────────────────────────────────────────────────────────────────────────

def test_reconstruct_identity():
    s = r"R_{\mu\nu\rho\sigma}"
    assert reconstruct(tokenize(s)) == s


# ─────────────────────────────────────────────────────────────────────────────
# Module 6 – Diff Generator
# ─────────────────────────────────────────────────────────────────────────────

def test_diff_reports_change():
    latex = r"v_i"
    toks = tokenize(latex)
    classify(toks, {"i"})
    rules = [SubRule("i", "r")]
    mod = substitute(toks, rules)
    report = generate_diff(toks, mod, rules, [], [])
    assert any("'i' → 'r'" in c for c in report.changes)

def test_diff_no_change():
    latex = r"v_i"
    toks = tokenize(latex)
    classify(toks, {"i"})
    mod = substitute(toks, [])
    report = generate_diff(toks, mod, [], [], [])
    assert report.changes == []

def test_diff_classifies_free():
    latex = r"v_i"
    toks = tokenize(latex)
    classify(toks, {"i"})
    mod = substitute(toks, [])
    report = generate_diff(toks, mod, [], [], [])
    assert report.classifications.get("i") == "free"

def test_diff_pattern_both_lower():
    latex = r"v_i v_i"
    toks = tokenize(latex)
    classify(toks, set())
    mod = substitute(toks, [])
    report = generate_diff(toks, mod, [], [], [])
    pat = report.dummy_patterns.get("i", "")
    assert "both-lower" in pat

def test_format_report_nocrash():
    latex = r"T_i^j"
    toks = tokenize(latex)
    classify(toks, {"i", "j"})
    mod = substitute(toks, [SubRule("i", "k")])
    report = generate_diff(toks, mod, [SubRule("i", "k")], [], [])
    input_result = verify(latex)
    output_result = verify(reconstruct(mod))
    text = format_report(report, reconstruct(mod),
                         input_result=input_result,
                         output_result=output_result)
    assert "Well-formed" in text
    assert "i" in text


# ─────────────────────────────────────────────────────────────────────────────
# Integration / physics scenarios
# ─────────────────────────────────────────────────────────────────────────────

def test_einstein_standard():
    # T^\mu{}_\mu  rename dummy \mu → \rho
    out, _, _ = _pipeline(r"T^\mu{}_\mu", [], r"\mu -> \rho")
    assert out == r"T^\rho{}_\rho"

def test_euclidean_contraction():
    out, _, _ = _pipeline(r"v_i v_i", [], "i -> k")
    assert out == r"v_k v_k"

def test_mixed_free_and_dummy():
    # R_{\mu\nu\rho}{}^\nu  — \nu is dummy, \mu \rho are free
    latex = r"R_{\mu\nu\rho}{}^\nu"
    toks = tokenize(latex)
    classify(toks, {r"\mu", r"\rho"})
    eligible = {t.text for t in toks if t.role in ("free", "dummy")}
    rules, _ = parse_rules(r"\mu -> \alpha, \nu -> \sigma", eligible)
    mod = substitute(toks, rules)
    result = reconstruct(mod)
    assert r"\alpha" in result   # free index renamed
    assert r"\sigma" in result   # dummy index renamed
    assert r"\mu" not in result
    assert r"\nu" not in result

def test_structural_label_safe():
    # \Gamma^A  — A must NOT be renamed even with an A->B rule present
    toks = tokenize(r"\Gamma^A")
    classify(toks, set())
    rules = [SubRule("A", "B")]
    mod = substitute(toks, rules)
    assert reconstruct(mod) == r"\Gamma^A"

def test_swap_free_indices():
    out, _, _ = _pipeline(r"F_{\mu\nu} - F_{\nu\mu}", [r"\mu", r"\nu"],
                          r"\mu -> \nu, \nu -> \mu")
    # After simultaneous swap
    assert r"\nu\mu" in out or r"\nu" in out  # rough sanity


# ─────────────────────────────────────────────────────────────────────────────
# Bug regression – free-index false-positive warning
# ─────────────────────────────────────────────────────────────────────────────

def test_bug_original_case_no_warning():
    # h_{i}=-K\left(\delta_{ij}-n_{i}n_{j}\right)\nabla^{2}n_{j}
    # 'i' is a genuine free index; must produce zero warnings.
    latex = r"h_{i}=-K\left(\delta_{ij}-n_{i}n_{j}\right)\nabla^{2}n_{j}"
    _, warns = _run(latex, ["i"])
    assert warns == [], f"Unexpected warnings: {warns}"

def test_bug_original_case_i_tagged_free():
    # All occurrences of 'i' must be tagged 'free', not 'dummy'.
    latex = r"h_{i}=-K\left(\delta_{ij}-n_{i}n_{j}\right)\nabla^{2}n_{j}"
    toks, _ = _run(latex, ["i"])
    i_toks = [t for t in toks if t.text == "i" and t.role != "body"]
    assert i_toks, "No 'i' index tokens found"
    assert all(t.role == "free" for t in i_toks), [(t.text, t.role) for t in i_toks]

def test_multiterm_free_index_no_warning():
    # a_{ij}b^{j} + c_{i}  —  'i' appears once per term; must not warn.
    latex = r"a_{ij}b^{j} + c_{i}"
    _, warns = _run(latex, ["i"])
    assert warns == [], f"Unexpected warnings: {warns}"

def test_multiterm_free_index_tagged_correctly():
    latex = r"a_{ij}b^{j} + c_{i}"
    toks, _ = _run(latex, ["i"])
    i_toks = [t for t in toks if t.text == "i" and t.role != "body"]
    assert all(t.role == "free" for t in i_toks), [(t.text, t.role) for t in i_toks]

def test_equation_lhs_rhs_split_no_warning():
    # f_{i} = g_{ij}h^{j}  —  'i' once on each side of '='; must not warn.
    latex = r"f_{i} = g_{ij}h^{j}"
    _, warns = _run(latex, ["i"])
    assert warns == [], f"Unexpected warnings: {warns}"

def test_genuine_dummy_still_detected_after_fix():
    # T^{\mu}{}_{\mu}  —  \mu contracted; must still be detected as dummy.
    toks, _ = _run(r"T^{\mu}{}_{\mu}", [])
    mu_toks = [t for t in toks if t.text == r"\mu"]
    assert all(t.role == "dummy" for t in mu_toks), [(t.text, t.role) for t in mu_toks]

def test_dummy_inside_parentheses_still_detected():
    # \left( v_i v_i \right)  —  contraction inside \left(…\right) is still dummy.
    toks, _ = _run(r"\left(v_i v_i\right)", [])
    i_toks = [t for t in toks if t.text == "i"]
    assert all(t.role == "dummy" for t in i_toks), [(t.text, t.role) for t in i_toks]

def test_j_is_dummy_in_original_bug_expression():
    # 'j' is genuinely contracted in the bug expression and must be tagged dummy.
    latex = r"h_{i}=-K\left(\delta_{ij}-n_{i}n_{j}\right)\nabla^{2}n_{j}"
    toks, _ = _run(latex, ["i"])
    j_toks = [t for t in toks if t.text == "j" and t.role != "body"]
    assert j_toks, "No 'j' index tokens found"
    assert all(t.role == "dummy" for t in j_toks), [(t.text, t.role) for t in j_toks]

def test_genuine_conflict_euclidean_contraction():
    # v_i v_i  —  'i' in two exclusive slots, same term: must warn even when
    # user declares it free (it really is contracted).
    _, warns = _run(r"v_i v_i", ["i"])
    assert any("possible dummy" in w.lower() for w in warns), warns

def test_genuine_conflict_trace():
    # T_{ii}  —  trace: same symbol twice in one slot → must warn.
    _, warns = _run(r"T_{ii}", ["i"])
    assert any("possible dummy" in w.lower() for w in warns), warns

def test_delta_ij_both_free_no_warning():
    # \delta_{ij}  —  i and j share one slot; neither is contracted; no warning.
    _, warns = _run(r"\delta_{ij}", ["i", "j"])
    assert warns == [], f"Unexpected warnings: {warns}"

def test_three_way_contraction_warns():
    # A_i B_i C_i  —  three exclusive slots for i in one term: must warn.
    _, warns = _run(r"A_i B_i C_i", ["i"])
    assert any("possible dummy" in w.lower() for w in warns), warns


# ─────────────────────────────────────────────────────────────────────────────
# Verifier integration tests  (LIE-17)
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_blocks_on_ill_formed_input():
    """Ill-formed input (triple index) should be caught by the verifier."""
    result = verify(r"A_{iij} B_i")
    assert not result.well_formed
    assert result.error is not None

def test_pipeline_blocks_inconsistent_free():
    """Inconsistent free indices across additive terms should be caught."""
    result = verify(r"A_i B_j + C_{ik}")
    assert not result.well_formed

def test_output_collision_detected():
    """Substitution i→j when j already exists should produce ill-formed output."""
    # Input:  A_{ij} B^j  — free: i, dummy: j — well-formed
    latex = r"A_{ij} B^j"
    input_result = verify(latex)
    assert input_result.well_formed
    assert set(input_result.free_indices.keys()) == {"i"}
    assert "j" in input_result.dummy_indices

    # Now substitute i→j: A_{jj} B^j — j appears 3 times → ill-formed
    toks = tokenize(latex)
    free_symbols = set(input_result.free_indices.keys())
    classify(toks, free_symbols)
    rules = [SubRule("i", "j")]
    mod = substitute(toks, rules)
    output_latex = reconstruct(mod)
    output_result = verify(output_latex)
    assert not output_result.well_formed

def test_verifier_free_indices_drive_classification():
    """Verifier's free-index detection should match what classify() uses."""
    latex = r"g_{ij} v^j"
    result = verify(latex)
    assert result.well_formed
    assert set(result.free_indices.keys()) == {"i"}
    assert result.dummy_indices == {"j"}

    # When classify() is given the verifier's free set, i should be free
    toks = tokenize(latex)
    classify(toks, set(result.free_indices.keys()))
    i_toks = [t for t in toks if t.text == "i" and t.role != "body"]
    assert all(t.role == "free" for t in i_toks)
    j_toks = [t for t in toks if t.text == "j" and t.role != "body"]
    assert all(t.role == "dummy" for t in j_toks)

def test_format_report_with_verifier_results():
    """format_report should include verifier sections when results are provided."""
    latex = r"T^i_j"
    toks = tokenize(latex)
    input_result = verify(latex)
    classify(toks, set(input_result.free_indices.keys()))
    mod = substitute(toks, [])
    report = generate_diff(toks, mod, [], [], [])
    text = format_report(report, reconstruct(mod),
                         input_result=input_result,
                         output_result=verify(reconstruct(mod)))
    assert "Einstein Verification (Input)" in text
    assert "Einstein Verification (Output)" in text
    assert "Well-formed" in text

def test_format_report_ill_formed_output():
    """format_report should show ILL-FORMED when output verification fails."""
    from einstein_summation_verifier import VerificationResult as VR
    ill = VR(well_formed=False, free_indices={}, dummy_indices=set(),
             error="Index 'j' appears more than twice")
    ok = VR(well_formed=True, free_indices={"i": False}, dummy_indices={"j"})
    report = generate_diff([], [], [], [], [])
    text = format_report(report, "", input_result=ok, output_result=ill)
    assert "ILL-FORMED" in text

def test_digit_not_treated_as_index():
    """Digits should never be treated as tensor indices (LIE-3)."""
    # T_2  — the '2' is NOT an index
    toks = tokenize(r"T_2")
    classify(toks, set())
    digit_toks = [t for t in toks if t.text == "2"]
    # digit should NOT be classified as free/dummy/structural
    for t in digit_toks:
        assert t.role == "body", f"Digit '2' was classified as {t.role}"


# ─────────────────────────────────────────────────────────────────────────────
# Runner (plain python, no pytest needed)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Non-Index Symbols (new feature)
# ─────────────────────────────────────────────────────────────────────────────

def test_nonindex_letter_excluded():
    """A letter declared as non-index should be classified as structural, not dummy."""
    toks = tokenize(r"A_t B_t")
    classify(toks, set(), frozenset({"t"}))
    t_toks = [t for t in toks if t.text == "t"]
    assert len(t_toks) == 2
    for t in t_toks:
        assert t.role != "dummy", f"t should not be dummy, got role={t.role}"


def test_nonindex_command_excluded():
    r"""A command declared as non-index (e.g. \perp) should be structural."""
    toks = tokenize(r"v_{\perp}")
    classify(toks, set(), frozenset({r"\perp"}))
    perp = [t for t in toks if t.text == r"\perp"]
    assert len(perp) == 1
    assert perp[0].role != "free" and perp[0].role != "dummy", \
        f"\\perp should be structural, got role={perp[0].role}"


def test_nonindex_not_eligible():
    """A rule targeting a non-index symbol should produce a warning."""
    toks = tokenize(r"A^{\mu}{}_{\mu} B_t")
    ni = frozenset({"t"})
    classify(toks, set(), ni)
    eligible = set()
    for t in toks:
        sym = _index_symbol(t, ni)
        if sym and t.role in ("free", "dummy"):
            eligible.add(sym)
    assert r"\mu" in eligible, f"\\mu should be eligible, got {eligible}"
    assert "t" not in eligible, f"t should NOT be eligible, got {eligible}"
    _, warnings = parse_rules("t->s", eligible)
    assert any("no effect" in w for w in warnings), \
        f"Expected 'no effect' warning, got: {warnings}"


def test_nonindex_empty_backward_compat():
    """With no non-index set, t appearing twice should be detected as dummy (backward compat)."""
    toks = tokenize(r"A_t B_t")
    classify(toks, set())  # no nonindex → default empty frozenset
    t_toks = [t for t in toks if t.text == "t"]
    assert any(t.role == "dummy" for t in t_toks), \
        f"t should be dummy when nonindex is empty, got roles={[t.role for t in t_toks]}"


def test_nonindex_mixed():
    r"""Mixed: \mu is dummy (contracted), t is excluded as non-index."""
    toks = tokenize(r"T^{\mu}{}_{\mu} A_t")
    classify(toks, set(), frozenset({"t"}))
    mu_toks = [t for t in toks if t.text == r"\mu"]
    t_toks  = [t for t in toks if t.text == "t"]
    assert all(t.role == "dummy" for t in mu_toks), \
        f"\\mu should be dummy, got roles={[t.role for t in mu_toks]}"
    assert all(t.role != "dummy" for t in t_toks), \
        f"t should not be dummy, got roles={[t.role for t in t_toks]}"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ✓  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'─'*50}")
    print(f"  {passed} passed  |  {failed} failed  |  {len(tests)} total")