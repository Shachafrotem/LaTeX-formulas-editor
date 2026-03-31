"""
LaTeX Index Editor
==================
A lightweight desktop tool for reliable index substitutions in LaTeX equations.

Modules implemented:
  1. Tokenizer          – breaks LaTeX into a flat token stream
  2. Index Classifier   – tags tokens as free / dummy / structural / body
  3. Rule Parser        – parses  "old -> new"  substitution rules
  4. Substitution Engine – simultaneous (non-sequential) token replacement
  5. Reconstructor      – token stream → LaTeX string
  6. Diff / Preview     – human-readable change summary
  7. GUI                – tkinter interface (no external deps)
  8. Error Handler      – typed errors → plain-English messages
"""

from __future__ import annotations
import re
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
from typing import Optional


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
# MODULE 1 – LaTeX Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

# Token types
T_COMMAND   = "command"      # \alpha, \mu, \frac, …
T_LBRACE    = "lbrace"       # {
T_RBRACE    = "rbrace"       # }
T_LBRACKET  = "lbracket"     # [
T_RBRACKET  = "rbracket"     # ]
T_SUBSCRIPT = "subscript"    # _
T_SUPER     = "superscript"  # ^
T_SPACE     = "space"        # whitespace run
T_CHAR      = "char"         # any other single character


@dataclass
class Token:
    type:  str
    text:  str
    # Set by the classifier
    role:  str = "body"           # body | free | dummy | structural
    upper: Optional[bool] = None  # True = ^, False = _, None = not an index


def tokenize(latex: str) -> list[Token]:
    """
    Break *latex* into a flat list of Token objects.
    Raises ParseError on illegal input.
    """
    tokens: list[Token] = []
    i = 0
    n = len(latex)
    while i < n:
        ch = latex[i]

        # LaTeX command  \word  or  \.  (single special char after backslash)
        if ch == "\\":
            if i + 1 >= n:
                raise ParseError("Trailing backslash at end of input.")
            nxt = latex[i + 1]
            if nxt.isalpha():
                j = i + 1
                while j < n and latex[j].isalpha():
                    j += 1
                tokens.append(Token(T_COMMAND, latex[i:j]))
                i = j
            else:
                tokens.append(Token(T_COMMAND, latex[i:i+2]))
                i += 2

        elif ch == "{":
            tokens.append(Token(T_LBRACE, ch)); i += 1
        elif ch == "}":
            tokens.append(Token(T_RBRACE, ch)); i += 1
        elif ch == "[":
            tokens.append(Token(T_LBRACKET, ch)); i += 1
        elif ch == "]":
            tokens.append(Token(T_RBRACKET, ch)); i += 1
        elif ch == "_":
            tokens.append(Token(T_SUBSCRIPT, ch)); i += 1
        elif ch == "^":
            tokens.append(Token(T_SUPER, ch)); i += 1
        elif ch in (" ", "\t", "\n", "\r"):
            j = i
            while j < n and latex[j] in (" ", "\t", "\n", "\r"):
                j += 1
            tokens.append(Token(T_SPACE, latex[i:j]))
            i = j
        else:
            tokens.append(Token(T_CHAR, ch)); i += 1

    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 – Index Classifier
# ─────────────────────────────────────────────────────────────────────────────

def _index_symbol(tok: Token) -> Optional[str]:
    """Return the *symbol string* for a token that could be an index, else None."""
    if tok.type == T_CHAR and re.match(r"[A-Za-z0-9]", tok.text):
        return tok.text
    if tok.type == T_COMMAND:
        return tok.text          # e.g. \mu, \nu
    return None


def _collect_candidates(tokens: list[Token]) -> list[int]:
    """
    Walk the token stream and return the *indices* (into `tokens`) of every
    token that immediately follows a _ or ^, possibly through braces.

    The token at that position also has its .upper attribute set (True / False).
    """
    candidate_positions: list[int] = []
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]
        if tok.type in (T_SUBSCRIPT, T_SUPER):
            is_upper = (tok.type == T_SUPER)
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
                    i += 1
                i += 1  # skip }
            else:
                # single-token index
                if _index_symbol(tokens[i]) is not None:
                    tokens[i].upper = is_upper
                    candidate_positions.append(i)
                i += 1
        else:
            i += 1

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

    Returns a sorted list of warnings (strings).
    """
    warnings: list[str] = []

    candidate_positions = _collect_candidates(tokens)

    # Split candidates into per-term groups and detect dummies per term.
    # A symbol is a dummy index if it appears ≥ 2 times within the same
    # top-level term.  Counting globally (across the whole expression) is
    # wrong because a free index legitimately appears once per term in every
    # additive summand, giving a large global count that does not indicate
    # contraction.
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

    # User-declared free symbols are trusted unconditionally.
    # The former conflict warning (fired when a declared free symbol also
    # appeared in detected_dummy) has been removed because realistic physics
    # expressions such as  h_{i} = -K(δ_{ij} - n_{i}n_{j})∇²n_{j}  produce
    # unavoidable false positives: the count-based heuristic cannot distinguish
    # "two different free indices sharing a tensor slot" from "an index
    # contracted with itself".  The user's explicit declaration is authoritative.

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

    all_warnings = classifier_warnings + rule_warnings
    return DiffReport(
        classifications=classifications,
        dummy_patterns=dummy_patterns,
        changes=changes,
        warnings=all_warnings,
    )


def format_report(report: DiffReport, output_latex: str) -> str:
    """Render a DiffReport to a human-readable string for the GUI panel."""
    lines: list[str] = []

    lines.append("─── Index Classification ───")
    if report.classifications:
        for sym, role in sorted(report.classifications.items()):
            pat = report.dummy_patterns.get(sym, "")
            pat_str = f"  [{pat}]" if pat else ""
            lines.append(f"  {sym:>8s}  →  {role}{pat_str}")
    else:
        lines.append("  (no index symbols found)")

    lines.append("")
    lines.append("─── Substitutions Applied ───")
    if report.changes:
        lines.extend(report.changes)
    else:
        lines.append("  (no changes made)")

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
        root.minsize(780, 580)

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root

        # ── Title bar ──
        title_frame = tk.Frame(root, bg=BG_PANEL, pady=10)
        title_frame.pack(fill="x", side="top")
        tk.Label(
            title_frame, text="LaTeX Index Editor",
            font=("Segoe UI", 15, "bold"),
            bg=BG_PANEL, fg=FG_ACCENT,
        ).pack(side="left", padx=16)
        tk.Label(
            title_frame,
            text="Reliable tensor-index substitution in LaTeX equations",
            font=("Segoe UI", 9), bg=BG_PANEL, fg=FG_DIM,
        ).pack(side="left", padx=4)

        # ── Main layout: left column (inputs) + right column (outputs) ──
        main = tk.Frame(root, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=12, pady=8)

        left  = tk.Frame(main, bg=BG_DARK)
        right = tk.Frame(main, bg=BG_DARK)
        left.pack(side="left",  fill="both", expand=True, padx=(0, 6))
        right.pack(side="right", fill="both", expand=True, padx=(6, 0))

        # ── Input formula ──
        self._input_box = self._labeled_textbox(
            left, "① Input Formula  (paste your LaTeX equation)",
            height=5,
        )

        # ── Free index declaration ──
        fi_frame = tk.Frame(left, bg=BG_DARK)
        fi_frame.pack(fill="x", pady=(8, 0))
        lbl = self._section_label(fi_frame, "② Free Indices  (comma-separated, e.g.  i, j, k)")
        lbl.pack(side="left")
        help_btn = tk.Label(fi_frame, text=" ?", fg=FG_ACCENT, bg=BG_DARK,
                            font=("Segoe UI", 10, "bold"), cursor="hand2")
        help_btn.pack(side="left")
        ToolTip(help_btn,
            "Free indices are the un-contracted (open) indices of the expression — "
            "typically the ones that remain after all Einstein summations have been "
            "evaluated.  Only symbols you list here will be tagged as free indices "
            "and substituted.  Dummy (contracted) indices are detected automatically "
            "by counting repetitions in index positions.")
        self._free_entry = self._entry(left)

        # ── Replacement rules ──
        self._rules_box = self._labeled_textbox(
            left,
            "③ Replacement Rules  (one per line or comma-separated, e.g.  i -> r)",
            height=4,
        )

        # ── Apply button ──
        btn_frame = tk.Frame(left, bg=BG_DARK)
        btn_frame.pack(fill="x", pady=10)
        self._apply_btn = tk.Button(
            btn_frame, text="▶  Apply Substitution",
            command=self._run,
            bg=ACCENT, fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat", padx=16, pady=6,
            activebackground="#1D4ED8", activeforeground="white",
            cursor="hand2",
        )
        self._apply_btn.pack(side="left")

        self._clear_btn = tk.Button(
            btn_frame, text="✕  Clear All",
            command=self._clear_all,
            bg=BG_PANEL, fg=FG_DIM,
            font=("Segoe UI", 10),
            relief="flat", padx=10, pady=6,
            cursor="hand2",
        )
        self._clear_btn.pack(side="left", padx=(8, 0))

        # ── Error / status bar ──
        self._status_var = tk.StringVar(value="")
        self._status_lbl = tk.Label(
            left, textvariable=self._status_var,
            bg=BG_DARK, fg=FG_WARN,
            font=("Segoe UI", 9), justify="left", wraplength=360,
            anchor="w",
        )
        self._status_lbl.pack(fill="x", pady=(0, 4))

        # ── Output formula ──
        out_hdr = tk.Frame(right, bg=BG_DARK)
        out_hdr.pack(fill="x")
        self._section_label(out_hdr, "④ Output Formula").pack(side="left")
        copy_btn = tk.Button(
            out_hdr, text="⎘ Copy",
            command=self._copy_output,
            bg=BG_PANEL, fg=FG_ACCENT,
            font=("Segoe UI", 9), relief="flat", padx=8, pady=2,
            cursor="hand2",
        )
        copy_btn.pack(side="right")

        self._output_box = tk.Text(
            right, height=5,
            font=FONT_MONO, bg=BG_FIELD, fg=FG_OK,
            insertbackground=FG_MAIN, relief="flat",
            padx=8, pady=6, state="disabled",
            highlightbackground=BORDER, highlightthickness=1,
        )
        self._output_box.pack(fill="both", expand=False, pady=(2, 0))

        # ── Classification & change summary ──
        self._section_label(right, "⑤ Classification & Change Summary").pack(
            anchor="w", pady=(10, 0))
        self._summary_box = tk.Text(
            right, height=18,
            font=("Courier New", 10), bg=BG_FIELD, fg=FG_DIM,
            insertbackground=FG_MAIN, relief="flat",
            padx=8, pady=6, state="disabled",
            highlightbackground=BORDER, highlightthickness=1,
        )
        self._summary_box.pack(fill="both", expand=True, pady=(2, 0))

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
        self._free_entry.delete(0, "end")
        self._set_output("")
        self._set_summary("")
        self._clear_status()

    def _run(self):
        """Main pipeline: run all modules and update the UI."""
        self._clear_status()

        latex_input = self._input_box.get("1.0", "end").strip()
        free_text   = self._free_entry.get().strip()
        rules_text  = self._rules_box.get("1.0", "end").strip()

        if not latex_input:
            self._set_status("Please enter a LaTeX formula in the Input Formula field.")
            return

        try:
            # ── Module 1: Tokenize ──
            tokens = tokenize(latex_input)

            # ── Parse free index list ──
            free_symbols: set[str] = set()
            if free_text:
                for sym in re.split(r"[,\s]+", free_text):
                    sym = sym.strip()
                    if sym:
                        free_symbols.add(sym)

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

            # ── Module 6: Diff / Preview ──
            report = generate_diff(
                tokens, modified_tokens, rules,
                classifier_warnings, rule_warnings,
            )
            summary_text = format_report(report, output_latex)

            # ── Update UI ──
            self._set_output(output_latex)
            self._set_summary(summary_text)

            if report.warnings:
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