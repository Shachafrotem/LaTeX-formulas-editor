"""
LaTeX Index Editor
==================
A lightweight desktop tool for reliable index substitutions in LaTeX equations.

Modules implemented:
  1. Tokenizer          – imported from einstein_summation_verifier
  2. Index Classifier   – tags tokens as free / dummy / structural / body
  3. Rule Parser        – parses  "old -> new"  substitution rules
  4. Substitution Engine – simultaneous (non-sequential) token replacement
  5. Reconstructor      – token stream → LaTeX string
  6. Diff / Preview     – human-readable change summary
  7. GUI                – tkinter interface (no external deps)
  8. Error Handler      – typed errors → plain-English messages

The einstein_summation_verifier is the authoritative source for tokenization
and index classification.  Free indices are auto-detected; the user no longer
declares them manually.
"""

from __future__ import annotations
import re
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
from typing import Optional

from einstein_summation_verifier import (
    tokenize, Token,
    T_COMMAND, T_LBRACE, T_RBRACE, T_LBRACKET, T_RBRACKET,
    T_SUBSCRIPT, T_SUPER, T_SPACE, T_CHAR,
    verify, VerificationResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 8 – Error Handler  (defined first; used by every other module)
# ─────────────────────────────────────────────────────────────────────────────

class LIEError(Exception):
    """Base for all LaTeX Index Editor errors."""
    pass

class ParseError(LIEError):
    pass

class RuleError(LIEError):
    pass

class TargetWarning(LIEError):
    pass

class NoOpWarning(LIEError):
    pass

class DummyConflictWarning(LIEError):
    pass


def friendly_message(exc: Exception) -> str:
    """Translate a typed exception into a plain-English user message."""
    if isinstance(exc, ParseError):
        return f"Parse error: {exc}"
    if isinstance(exc, RuleError):
        return f"Rule error: {exc}"
    if isinstance(exc, TargetWarning):
        return f"Warning: {exc}"
    if isinstance(exc, NoOpWarning):
        return f"Warning (no-op): {exc}"
    if isinstance(exc, DummyConflictWarning):
        return f"Conflict warning: {exc}"
    return f"Unexpected error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 – LaTeX Tokenizer  (imported from einstein_summation_verifier)
# ─────────────────────────────────────────────────────────────────────────────
# Token types, the Token dataclass, and the tokenize() function are all
# imported from einstein_summation_verifier at the top of this file.


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 – Index Classifier
# ─────────────────────────────────────────────────────────────────────────────

def _index_symbol(tok: Token) -> Optional[str]:
    """Return the *symbol string* for a token that could be an index, else None."""
    if tok.type == T_CHAR and re.match(r"[A-Za-z]", tok.text):
        return tok.text
    if tok.type == T_COMMAND:
        return tok.text          # e.g. \mu, \nu
    return None


def _collect_candidates_with_slots(
    tokens: list[Token],
) -> tuple[list[int], dict[int, int]]:
    """
    Walk the token stream and return every token that sits in an index
    position (immediately after _ or ^, possibly inside braces), together
    with a slot-membership map.

    Returns:
        candidate_positions – list of token indices in index positions.
            Side-effect: sets .upper on each candidate token (True = ^,
            False = _).
        slot_map – {token_index: slot_id}.  Every token inside the same
            _{…} or ^{…} group shares the same slot_id (a monotonically
            increasing integer).  This lets callers distinguish
            "two separate slots" (e.g. v_i v_i → slot 1 and slot 2) from
            "two symbols in one slot" (e.g. delta_{ij} -> slot 1).
    """
    candidate_positions: list[int] = []
    slot_map: dict[int, int] = {}
    slot_id = 0
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]
        if tok.type in (T_SUBSCRIPT, T_SUPER):
            is_upper = (tok.type == T_SUPER)
            slot_id += 1
            i += 1
            # skip whitespace
            while i < n and tokens[i].type == T_SPACE:
                i += 1
            if i >= n:
                break

            if tokens[i].type == T_LBRACE:
                # collect everything inside the braces
                i += 1  # skip {
                while i < n and tokens[i].type == T_SPACE:
                    i += 1
                while i < n and tokens[i].type != T_RBRACE:
                    if _index_symbol(tokens[i]) is not None:
                        tokens[i].upper = is_upper
                        candidate_positions.append(i)
                        slot_map[i] = slot_id
                    i += 1
                i += 1  # skip }
            else:
                # single-token index
                if _index_symbol(tokens[i]) is not None:
                    tokens[i].upper = is_upper
                    candidate_positions.append(i)
                    slot_map[i] = slot_id
                i += 1
        else:
            i += 1

    return candidate_positions, slot_map


def _collect_candidates(tokens: list[Token]) -> list[int]:
    """
    Convenience wrapper around _collect_candidates_with_slots that returns
    only the candidate position list (for callers that don't need slot info).
    """
    candidate_positions, _ = _collect_candidates_with_slots(tokens)
    return candidate_positions


def _split_candidates_by_term(
    tokens: list[Token], candidate_positions: list[int]
) -> list[list[int]]:
    """
    Partition *candidate_positions* into groups, one group per top-level
    additive term.

    A term boundary is any '+', '-', or '=' T_CHAR token encountered while
    the nesting depth is zero.  Depth is tracked across:
      T_LBRACE / T_RBRACE        { }
      T_LBRACKET / T_RBRACKET    [ ]
      T_CHAR '(' / ')'           ( )
      T_COMMAND '\\left' / '\\right'

    Note: \\left and \\right are counted directly (depth +=/-= 1) so that
    the bracket character that follows (e.g. '(' after \\left) is NOT also
    counted — that would double-count the depth change.

    Empty groups (produced by a leading '-' or consecutive boundaries) are
    discarded from the output.
    """
    # Build a set for O(1) candidate lookup
    candidate_set = set(candidate_positions)

    depth = 0
    current_group: list[int] = []
    groups: list[list[int]] = []

    for i, tok in enumerate(tokens):
        # ── depth tracking ────────────────────────────────────────────────
        if tok.type == T_LBRACE or tok.type == T_LBRACKET:
            depth += 1
        elif tok.type == T_RBRACE or tok.type == T_RBRACKET:
            depth -= 1
        elif tok.type == T_CHAR and tok.text == "(":
            depth += 1
        elif tok.type == T_CHAR and tok.text == ")":
            depth -= 1
        elif tok.type == T_COMMAND and tok.text == r"\left":
            depth += 1
        elif tok.type == T_COMMAND and tok.text == r"\right":
            depth -= 1

        # ── term boundary at depth zero ───────────────────────────────────
        elif depth == 0 and tok.type == T_CHAR and tok.text in ("+", "-", "="):
            groups.append(current_group)
            current_group = []
            continue

        # ── accumulate candidates into current group ──────────────────────
        if i in candidate_set:
            current_group.append(i)

    # Flush the final group
    groups.append(current_group)

    # Discard empty groups (e.g. from a leading unary minus)
    return [g for g in groups if g]


def classify(tokens: list[Token], free_symbols: set[str]) -> list[str]:
    """
    Tag every token in *tokens* with a .role:
      'free'       – declared by the user as a free index
      'dummy'      – detected as a contracted (repeated) index
      'structural' – syntactic index position but not a tensor index
      'body'       – not in any index position

    Returns a list of warnings (strings).
    """
    warnings: list[str] = []

    candidate_positions, slot_map = _collect_candidates_with_slots(tokens)

    # ── Per-term dummy detection ──────────────────────────────────────────────
    # Split candidates into groups, one per top-level additive term (terms are
    # separated by +, -, or = at nesting depth zero).  A symbol is a dummy
    # index only if it appears ≥ 2 times within the *same* term — counting
    # globally is wrong because a free index legitimately appears once per term
    # in every summand of a multi-term expression.
    term_groups = _split_candidates_by_term(tokens, candidate_positions)
    detected_dummy: set[str] = set()
    for group in term_groups:
        term_count: dict[str, int] = {}
        for pos in group:
            sym = _index_symbol(tokens[pos])
            if sym:
                term_count[sym] = term_count.get(sym, 0) + 1
        for sym, cnt in term_count.items():
            if cnt >= 2:
                detected_dummy.add(sym)

    # ── Slot-aware conflict check for user-declared free symbols ─────────────
    # The simple per-term count is not enough to distinguish a genuinely free
    # index from a mistakenly declared one, because a multi-index slot like
    # \delta_{ij} causes both i and j to appear in the same term without being
    # contracted.  We use slot structure to make a more precise decision:
    #
    # A user-declared free symbol is flagged as a conflict if, within any
    # single top-level term, it occupies ≥ 2 *exclusive* slots — where an
    # exclusive slot is one whose entire symbol content is just that one symbol
    # (no companion indices in the same slot).  This correctly catches:
    #   v_i v_i      — two exclusive slots in one term → warn
    #   T_{ii}       — one slot but the symbol appears twice in it → warn (trace)
    #   A_i B_i C_i  — three exclusive slots → warn
    # and correctly clears:
    #   \delta_{ij}  — i shares its slot with j → slot not exclusive → no warn
    #   h_{i}=-K(\delta_{ij}-n_{i}n_{j})\nabla^2 n_{j}
    #                — on the RHS, i appears in \delta_{ij} (not exclusive)
    #                  and n_{i} (exclusive), giving only 1 exclusive slot
    #                  in that term → no warn
    #   a_{ij}b^j + c_{i}  — i has 1 exclusive slot per term → no warn

    for sym in sorted(free_symbols):
        for group in term_groups:
            # Collect the set of symbols in each slot for this term
            slot_syms: dict[int, list[str]] = {}
            for pos in group:
                sid = slot_map.get(pos)
                s = _index_symbol(tokens[pos])
                if sid is not None and s:
                    slot_syms.setdefault(sid, []).append(s)

            # Trace: sym appears twice inside the same slot (e.g. T_{ii})
            trace = any(
                syms.count(sym) >= 2
                for syms in slot_syms.values()
                if sym in syms
            )
            # Exclusive slots: slots whose only distinct symbol is sym
            exclusive_count = sum(
                1 for syms in slot_syms.values()
                if sym in syms and len(set(syms)) == 1
            )

            if trace or exclusive_count >= 2:
                total = sum(1 for pos in group if _index_symbol(tokens[pos]) == sym)
                warnings.append(
                    f"'{sym}' is declared as a free index but appears {total} time(s) "
                    f"in exclusive index slots within the same term "
                    f"(possible dummy index). Please verify."
                )
                break  # one warning per symbol is enough

    # Tag candidate tokens
    for pos in candidate_positions:
        sym = _index_symbol(tokens[pos])
        if sym is None:
            continue
        if sym in free_symbols:
            tokens[pos].role = "free"
        elif sym in detected_dummy:
            tokens[pos].role = "dummy"
        else:
            tokens[pos].role = "structural"

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 – Rule Parser
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubRule:
    old: str
    new: str


def parse_rules(rule_text: str, eligible_symbols: set[str]) -> tuple[list[SubRule], list[str]]:
    """
    Parse lines / comma-separated  "old -> new"  rules.
    Returns (rules, warnings).
    Raises RuleError on malformed input.
    """
    rules: list[SubRule] = []
    warnings: list[str] = []
    seen_old: set[str] = set()

    parts = re.split(r"[,\n]+", rule_text)
    for raw in parts:
        raw = raw.strip()
        if not raw:
            continue
        if "->" not in raw:
            raise RuleError(f"Rule '{raw}' is missing '->'. Expected format: old -> new")
        lhs, _, rhs = raw.partition("->")
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not lhs:
            raise RuleError(f"Rule '{raw}': left-hand side is empty.")
        if not rhs:
            raise RuleError(f"Rule '{raw}': right-hand side is empty.")
        if lhs == rhs:
            warnings.append(f"Rule '{lhs} -> {rhs}' maps a symbol to itself (no-op).")
            continue
        if lhs in seen_old:
            raise RuleError(f"Duplicate rule for '{lhs}'.")
        seen_old.add(lhs)

        if eligible_symbols and lhs not in eligible_symbols:
            warnings.append(
                f"'{lhs}' is not classified as a free or dummy index — "
                f"rule '{lhs} -> {rhs}' will have no effect."
            )
        rules.append(SubRule(old=lhs, new=rhs))

    return rules, warnings


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 – Simultaneous Substitution Engine
# ─────────────────────────────────────────────────────────────────────────────

def substitute(tokens: list[Token], rules: list[SubRule]) -> list[Token]:
    """
    Return a *new* list of Token objects with all substitutions applied
    simultaneously.  Only tokens with role 'free' or 'dummy' are touched.
    """
    lookup = {r.old: r.new for r in rules}
    result: list[Token] = []

    for tok in tokens:
        if tok.role in ("free", "dummy"):
            sym = _index_symbol(tok)
            if sym and sym in lookup:
                new_tok = Token(
                    type=tok.type,
                    text=lookup[sym],
                    role=tok.role,
                    upper=tok.upper,
                )
                result.append(new_tok)
                continue
        result.append(tok)   # pass-through (shallow copy keeps original)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 – LaTeX Reconstructor
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct(tokens: list[Token]) -> str:
    """Concatenate token texts to produce the output LaTeX string."""
    return "".join(t.text for t in tokens)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 – Diff / Preview Generator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiffReport:
    classifications: dict[str, str]   # symbol → role
    dummy_patterns:  dict[str, str]   # dummy symbol → pattern description
    changes:         list[str]        # human-readable change lines
    warnings:        list[str]        # all collected warnings


def generate_diff(
    original: list[Token],
    modified: list[Token],
    rules: list[SubRule],
    classifier_warnings: list[str],
    rule_warnings: list[str],
) -> DiffReport:
    """
    Produce a DiffReport comparing *original* and *modified* token streams.
    """
    # Build classification table from the original tagged stream
    classifications: dict[str, str] = {}
    dummy_upper_count: dict[str, int] = {}
    dummy_lower_count: dict[str, int] = {}

    for tok in original:
        sym = _index_symbol(tok)
        if sym and tok.role != "body":
            classifications[sym] = tok.role
            if tok.role == "dummy":
                if tok.upper:
                    dummy_upper_count[sym] = dummy_upper_count.get(sym, 0) + 1
                else:
                    dummy_lower_count[sym] = dummy_lower_count.get(sym, 0) + 1

    # Describe contraction patterns for dummy indices
    dummy_patterns: dict[str, str] = {}
    for sym in classifications:
        if classifications[sym] == "dummy":
            up  = dummy_upper_count.get(sym, 0)
            dn  = dummy_lower_count.get(sym, 0)
            if up > 0 and dn > 0:
                pat = f"mixed ({up}↑ {dn}↓)"
            elif up > 0:
                pat = f"both-upper ({up}↑)"
            else:
                pat = f"both-lower ({dn}↓)"
            dummy_patterns[sym] = pat

    # Build change list by comparing token texts
    changes: list[str] = []
    lookup = {r.old: r.new for r in rules}
    for pos, (orig, mod) in enumerate(zip(original, modified)):
        if orig.text != mod.text:
            ctx = ""
            # try to find the nearest preceding command for context
            for j in range(pos - 1, max(pos - 6, -1), -1):
                if original[j].type == T_COMMAND:
                    ctx = f" (near {original[j].text})"
                    break
            pos_label = "subscript" if orig.upper is False else ("superscript" if orig.upper else "position")
            changes.append(f"  token {pos}: '{orig.text}' → '{mod.text}'  [{pos_label}{ctx}]")

    # LIE-16: Keep only rule-parser warnings; classifier "possible dummy"
    # warnings are now redundant with the verifier's authoritative output.
    all_warnings = rule_warnings
    return DiffReport(
        classifications=classifications,
        dummy_patterns=dummy_patterns,
        changes=changes,
        warnings=all_warnings,
    )


def format_report(
    report: DiffReport,
    output_latex: str,
    *,
    input_result: Optional[VerificationResult] = None,
    output_result: Optional[VerificationResult] = None,
) -> str:
    """Render a DiffReport to a human-readable string for the GUI panel."""
    lines: list[str] = []

    # ── Einstein Verification (Input) ──  [LIE-13]
    lines.append("─── Einstein Verification (Input) ───")
    if input_result is not None:
        if input_result.well_formed:
            if input_result.free_indices:
                free_parts = []
                for name in sorted(input_result.free_indices):
                    pos = "upper" if input_result.free_indices[name] else "lower"
                    free_parts.append(f"{name} ({pos})")
                lines.append(f"  Free indices:  {', '.join(free_parts)}")
            else:
                lines.append("  Free indices:  (none — scalar expression)")
            if input_result.dummy_indices:
                lines.append(f"  Dummy indices: {', '.join(sorted(input_result.dummy_indices))}")
            else:
                lines.append("  Dummy indices: (none)")
            lines.append("  Well-formed ✓")
        else:
            lines.append(f"  ILL-FORMED ✗")
            lines.append(f"  {input_result.error}")
    else:
        lines.append("  (not verified)")

    # ── Substitutions Applied ──  [LIE-15]
    lines.append("")
    lines.append("─── Substitutions Applied ───")
    if report.changes:
        lines.extend(report.changes)
    else:
        lines.append("  (no changes made)")

    # ── Einstein Verification (Output) ──  [LIE-14]
    if output_result is not None:
        lines.append("")
        lines.append("─── Einstein Verification (Output) ───")
        if output_result.well_formed:
            if output_result.free_indices:
                free_parts = []
                for name in sorted(output_result.free_indices):
                    pos = "upper" if output_result.free_indices[name] else "lower"
                    free_parts.append(f"{name} ({pos})")
                lines.append(f"  Free indices:  {', '.join(free_parts)}")
            else:
                lines.append("  Free indices:  (none — scalar expression)")
            if output_result.dummy_indices:
                lines.append(f"  Dummy indices: {', '.join(sorted(output_result.dummy_indices))}")
            else:
                lines.append("  Dummy indices: (none)")
            lines.append("  Well-formed ✓")
        else:
            lines.append("  ILL-FORMED ✗")
            lines.append(f"  {output_result.error}")

    # ── Warnings ──  [LIE-16]: keep rule-parser warnings only
    if report.warnings:
        lines.append("")
        lines.append("─── Warnings ───")
        for w in report.warnings:
            lines.append(f"  ⚠  {w}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7 – GUI
# ─────────────────────────────────────────────────────────────────────────────

FONT_MONO  = ("Courier New", 11)
FONT_LABEL = ("Segoe UI", 10) if tk.TkVersion else ("TkDefaultFont", 10)
FONT_TITLE = ("Segoe UI", 13, "bold")
ACCENT     = "#2563EB"   # blue
BG_DARK    = "#1E293B"
BG_PANEL   = "#0F172A"
BG_FIELD   = "#1E293B"
FG_MAIN    = "#E2E8F0"
FG_DIM     = "#94A3B8"
FG_ACCENT  = "#60A5FA"
FG_WARN    = "#FBBF24"
FG_OK      = "#34D399"
BORDER     = "#334155"


class ToolTip:
    """Simple tooltip widget."""
    def __init__(self, widget: tk.Widget, text: str):
        self._widget = widget
        self._text   = text
        self._tip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            self._tip, text=self._text, justify="left",
            background="#1E3A5F", foreground="#E2E8F0",
            font=("Segoe UI", 9), relief="flat", padx=6, pady=4,
            wraplength=340,
        )
        lbl.pack()

    def _hide(self, _event=None):
        if self._tip:
            self._tip.destroy()
            self._tip = None


class LaTeXIndexEditorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("LaTeX Index Editor")
        root.configure(bg=BG_DARK)
        root.resizable(True, True)
        root.minsize(370, 700)

        self._build_ui()
        self._dock_right()

    def _dock_right(self):
        """Position the window flush against the right edge of the screen."""
        self.root.update_idletasks()
        win_w = 400
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = screen_w - win_w
        self.root.geometry(f"{win_w}x{screen_h}+{x}+0")

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root

        # ── Title bar ──
        title_frame = tk.Frame(root, bg=BG_PANEL, pady=6)
        title_frame.pack(fill="x", side="top")
        tk.Label(
            title_frame, text="LaTeX Index Editor",
            font=("Segoe UI", 12, "bold"),
            bg=BG_PANEL, fg=FG_ACCENT,
        ).pack(side="left", padx=12)

        # ── Scrollable main area ──
        # Use a canvas + frame so the whole UI scrolls if the window is short
        outer = tk.Frame(root, bg=BG_DARK)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=BG_DARK, highlightthickness=0)
        scrollbar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        self._main_frame = tk.Frame(canvas, bg=BG_DARK)

        self._main_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas_window = canvas.create_window((0, 0), window=self._main_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Make the inner frame stretch to the canvas width
        def _on_canvas_resize(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", _on_canvas_resize)

        # Mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        col = self._main_frame  # single-column parent

        # ── (1) Input formula ──
        self._input_box = self._labeled_textbox(
            col, "(1) Input Formula (paste your LaTeX equation)",
            height=4,
        )

        # ── (2) Replacement rules ──
        self._rules_box = self._labeled_textbox(
            col,
            "(2) Replacement Rules\n(one per line or comma-separated, e.g.  i->r)",
            height=3,
        )

        # ── Buttons row: Apply + Clear side by side ──
        btn_frame = tk.Frame(col, bg=BG_DARK)
        btn_frame.pack(fill="x", pady=(8, 2), padx=8)
        self._apply_btn = tk.Button(
            btn_frame, text="Apply Substitution",
            command=self._run,
            bg=ACCENT, fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat", padx=14, pady=5,
            activebackground="#1D4ED8", activeforeground="white",
            cursor="hand2",
        )
        self._apply_btn.pack(side="left")

        self._clear_btn = tk.Button(
            btn_frame, text="Clear All",
            command=self._clear_all,
            bg=BG_PANEL, fg=FG_DIM,
            font=("Segoe UI", 10),
            relief="flat", padx=14, pady=5,
            cursor="hand2",
        )
        self._clear_btn.pack(side="left", padx=(8, 0))

        # ── Error / status bar ──
        self._status_var = tk.StringVar(value="")
        self._status_lbl = tk.Label(
            col, textvariable=self._status_var,
            bg=BG_DARK, fg=FG_WARN,
            font=("Segoe UI", 9), justify="left",
            anchor="w", wraplength=340,
        )
        self._status_lbl.pack(fill="x", padx=8, pady=(2, 4))

        # ── (3) Output formula with Copy button right-aligned above ──
        copy_row = tk.Frame(col, bg=BG_DARK)
        copy_row.pack(fill="x", padx=8, pady=(4, 0))
        copy_btn = tk.Button(
            copy_row, text="Copy",
            command=self._copy_output,
            bg=BG_PANEL, fg=FG_ACCENT,
            font=("Segoe UI", 9), relief="flat", padx=8, pady=2,
            cursor="hand2",
        )
        copy_btn.pack(side="right")

        self._output_box = tk.Text(
            col, height=4,
            font=FONT_MONO, bg=BG_FIELD, fg=FG_OK,
            insertbackground=FG_MAIN, relief="flat",
            padx=8, pady=6, state="disabled",
            highlightbackground=BORDER, highlightthickness=1,
        )
        self._output_box.pack(fill="both", expand=False, padx=8, pady=(2, 0))

        # ── (4) Verification & Change Summary ──
        self._section_label(col, "(4) Verification & Change Summary").pack(
            anchor="w", padx=8, pady=(10, 0))
        self._summary_box = tk.Text(
            col, height=14,
            font=("Courier New", 9), bg=BG_FIELD, fg=FG_DIM,
            insertbackground=FG_MAIN, relief="flat",
            padx=8, pady=6, state="disabled",
            highlightbackground=BORDER, highlightthickness=1,
        )
        self._summary_box.pack(fill="both", expand=True, padx=8, pady=(2, 8))

    # ── Helper widget builders ────────────────────────────────────────────────

    def _section_label(self, parent, text: str) -> tk.Label:
        return tk.Label(parent, text=text,
                        bg=BG_DARK, fg=FG_DIM,
                        font=("Segoe UI", 9, "bold"), anchor="w")

    def _labeled_textbox(self, parent, label: str, height: int) -> tk.Text:
        self._section_label(parent, label).pack(fill="x", pady=(8, 0))
        box = tk.Text(
            parent, height=height,
            font=FONT_MONO, bg=BG_FIELD, fg=FG_MAIN,
            insertbackground=FG_MAIN, relief="flat",
            padx=8, pady=6,
            highlightbackground=BORDER, highlightthickness=1,
        )
        box.pack(fill="both", expand=False, pady=(2, 0))
        return box

    def _entry(self, parent) -> tk.Entry:
        e = tk.Entry(
            parent, font=FONT_MONO,
            bg=BG_FIELD, fg=FG_MAIN,
            insertbackground=FG_MAIN, relief="flat",
            highlightbackground=BORDER, highlightthickness=1,
        )
        e.pack(fill="x", pady=(2, 0), ipady=5, padx=1)
        return e

    # ── Actions ──────────────────────────────────────────────────────────────

    def _clear_status(self):
        self._status_var.set("")
        self._status_lbl.config(fg=FG_WARN)

    def _set_status(self, msg: str, ok: bool = False):
        self._status_var.set(msg)
        self._status_lbl.config(fg=FG_OK if ok else FG_WARN)

    def _set_output(self, text: str):
        self._output_box.config(state="normal")
        self._output_box.delete("1.0", "end")
        self._output_box.insert("1.0", text)
        self._output_box.config(state="disabled")

    def _set_summary(self, text: str):
        self._summary_box.config(state="normal")
        self._summary_box.delete("1.0", "end")
        self._summary_box.insert("1.0", text)
        self._summary_box.config(state="disabled")

    def _copy_output(self):
        text = self._output_box.get("1.0", "end").strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._set_status("Output copied to clipboard.", ok=True)

    def _clear_all(self):
        for box in (self._input_box, self._rules_box):
            box.delete("1.0", "end")
        self._set_output("")
        self._set_summary("")
        self._clear_status()

    def _run(self):
        """Main pipeline: run all modules and update the UI."""
        self._clear_status()

        latex_input = self._input_box.get("1.0", "end").strip()
        rules_text  = self._rules_box.get("1.0", "end").strip()

        if not latex_input:
            self._set_status("Please enter a LaTeX formula in the Input Formula field.")
            return

        try:
            # ── Module 1: Tokenize ──
            tokens = tokenize(latex_input)

            # ── Verify input (ground truth) ──  [LIE-4]
            input_result = verify(latex_input)
            if not input_result.well_formed:
                self._set_status(f"Input ill-formed: {input_result.error}")
                self._set_summary(format_report(
                    DiffReport({}, {}, [], []),
                    "",
                    input_result=input_result,
                    output_result=None,
                ))
                return

            # ── Free symbols from verifier (authoritative) ──  [LIE-4]
            free_symbols: set[str] = set(input_result.free_indices.keys())

            # ── Module 2: Classify ──
            classifier_warnings = classify(tokens, free_symbols)

            # Compute eligible symbols (free + dummy)
            eligible: set[str] = set()
            for tok in tokens:
                sym = _index_symbol(tok)
                if sym and tok.role in ("free", "dummy"):
                    eligible.add(sym)

            # ── Module 3: Parse rules ──
            rules: list[SubRule] = []
            rule_warnings: list[str] = []
            if rules_text:
                rules, rule_warnings = parse_rules(rules_text, eligible)

            # ── Module 4: Substitute ──
            modified_tokens = substitute(tokens, rules)

            # ── Module 5: Reconstruct ──
            output_latex = reconstruct(modified_tokens)

            # ── Verify output ──  [LIE-10]
            output_result = verify(output_latex)

            # ── Module 6: Diff / Preview ──  [LIE-11]
            report = generate_diff(
                tokens, modified_tokens, rules,
                classifier_warnings, rule_warnings,
            )
            summary_text = format_report(
                report, output_latex,
                input_result=input_result,
                output_result=output_result,
            )

            # ── Update UI ──
            self._set_output(output_latex)
            self._set_summary(summary_text)

            if not output_result.well_formed:
                self._set_status(
                    f"Output is ILL-FORMED after substitution: {output_result.error}",
                    ok=False,
                )
            elif report.warnings:
                self._set_status(
                    f"{len(report.warnings)} warning(s) — see summary panel.", ok=False
                )
            else:
                n_changes = len(report.changes)
                self._set_status(
                    f"Done — {n_changes} token(s) substituted.", ok=True
                )

        except LIEError as exc:
            self._set_status(friendly_message(exc))
        except Exception as exc:
            self._set_status(f"Unexpected error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = LaTeXIndexEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()