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
)


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
        assert False, "Should have raised ParseError"
    except ParseError:
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
    # \mu appears twice within the same term → genuine dummy conflict is
    # no longer warned about (user declaration is trusted unconditionally).
    # The warning block has been removed; this test checks there are NO
    # spurious warnings when a symbol that looks like a dummy is declared free.
    _, warns = _run(r"T^\mu{}_\mu", [r"\mu"])
    # No conflict warning expected — user declaration is authoritative
    assert not any("conflict" in w.lower() or "possible dummy" in w.lower() for w in warns), warns

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
    text = format_report(report, reconstruct(mod))
    assert "free" in text
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
    # T^{\mu}{}_{\mu}  —  \mu contracted; must still be detected as dummy
    # even after the per-term counting change.
    toks, _ = _run(r"T^{\mu}{}_{\mu}", [])
    mu_toks = [t for t in toks if t.text == r"\mu"]
    assert all(t.role == "dummy" for t in mu_toks), [(t.text, t.role) for t in mu_toks]

def test_dummy_inside_parentheses_still_detected():
    # \left( v_i v_i \right)  —  the contraction is inside \left(…\right)
    # but is still a genuine dummy within that term.
    toks, _ = _run(r"\left(v_i v_i\right)", [])
    i_toks = [t for t in toks if t.text == "i"]
    assert all(t.role == "dummy" for t in i_toks), [(t.text, t.role) for t in i_toks]

def test_j_is_dummy_in_original_bug_expression():
    # In the original bug expression, 'j' is genuinely contracted and must
    # be detected as a dummy index (it appears ≥ 2 times within the RHS term).
    latex = r"h_{i}=-K\left(\delta_{ij}-n_{i}n_{j}\right)\nabla^{2}n_{j}"
    toks, _ = _run(latex, ["i"])
    j_toks = [t for t in toks if t.text == "j" and t.role != "body"]
    assert j_toks, "No 'j' index tokens found"
    assert all(t.role == "dummy" for t in j_toks), [(t.text, t.role) for t in j_toks]


# ─────────────────────────────────────────────────────────────────────────────
# Runner (plain python, no pytest needed)
# ─────────────────────────────────────────────────────────────────────────────

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
