"""
Einstein Summation Verifier
============================
Receives a LaTeX tensor-calculus expression and:
  1. Identifies which indices are free and which are dummy.
  2. Verifies that the expression is well-formed under Einstein summation rules.

Rules enforced:
  - Each index appears at most twice per term (once = free, twice = dummy).
  - All additive terms share the same set of free indices (name + position).
  - Both sides of an equation ('=') have the same free indices.

This module reuses the tokenizer from latex_index_editor.py and builds a
recursive-descent parser on top to handle nested parentheses and
multiplicative grouping.
"""

from __future__ import annotations
import re
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

# ─── Import the tokenizer from the existing codebase ──────────────────────────
# We vendor the minimal pieces we need so this script is self-contained.

# Token types (mirrored from latex_index_editor)
T_COMMAND   = "command"
T_LBRACE    = "lbrace"
T_RBRACE    = "rbrace"
T_LBRACKET  = "lbracket"
T_RBRACKET  = "rbracket"
T_SUBSCRIPT = "subscript"
T_SUPER     = "superscript"
T_SPACE     = "space"
T_CHAR      = "char"


@dataclass
class Token:
    type:  str
    text:  str
    role:  str = "body"
    upper: Optional[bool] = None


def tokenize(latex: str) -> list[Token]:
    """Break *latex* into a flat list of Token objects."""
    tokens: list[Token] = []
    i = 0
    n = len(latex)
    while i < n:
        ch = latex[i]
        if ch == "\\":
            if i + 1 >= n:
                raise ValueError("Trailing backslash at end of input.")
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


# ─── Index representation ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Index:
    """An index occurrence: its name and whether it is upper (^) or lower (_)."""
    name: str
    is_upper: bool

    def __repr__(self):
        pos = "^" if self.is_upper else "_"
        return f"{pos}{self.name}"


# ─── Expression tree ──────────────────────────────────────────────────────────
# We parse the token stream into a lightweight AST:
#   Equation  = Expr '=' Expr
#   Expr      = Term (('+' | '-') Term)*
#   Term      = Factor (implicit-multiply Factor)*
#   Factor    = '(' Expr ')' | Atom
#   Atom      = tensor-symbol with its subscript/superscript indices


@dataclass
class IndexInfo:
    """Result of analysing indices in a (sub)expression."""
    free:  dict[str, bool]    # index_name -> is_upper  (free indices)
    dummy: set[str]           # dummy index names
    error: Optional[str] = None  # if set, expression is ill-formed


def _merge_product(left: IndexInfo, right: IndexInfo) -> IndexInfo:
    """
    Merge index info for a product of two factors.
    
    When two factors are multiplied:
    - An index that appears once in left AND once in right becomes dummy.
    - An index appearing twice already (dummy) in either stays dummy.
    - An index appearing once in only one factor stays free.
    - An index that would appear 3+ times is an error.
    """
    if left.error:
        return left
    if right.error:
        return right

    # Collect all index occurrences with their positions
    # We need to count total occurrences across both sides
    all_free = {}       # name -> is_upper (for indices appearing exactly once total)
    new_dummy = set(left.dummy) | set(right.dummy)

    # Check for conflicts: a dummy in one side appearing in the other is an error
    for name in left.dummy:
        if name in right.free or name in right.dummy:
            return IndexInfo({}, set(),
                error=f"Index '{name}' appears more than twice "
                      f"(already a dummy index, then appears again).")
    for name in right.dummy:
        if name in left.free:
            return IndexInfo({}, set(),
                error=f"Index '{name}' appears more than twice "
                      f"(already a dummy index, then appears again).")

    # Now merge free indices from both sides
    for name, is_upper in left.free.items():
        if name in right.free:
            # Appears in both -> becomes dummy
            new_dummy.add(name)
        else:
            all_free[name] = is_upper

    for name, is_upper in right.free.items():
        if name not in left.free:
            all_free[name] = is_upper

    return IndexInfo(all_free, new_dummy)


def _merge_sum(terms: list[IndexInfo]) -> IndexInfo:
    """
    Merge index info for a sum of terms.
    
    All terms must have the same free indices (name AND position).
    Dummy indices are collected from all terms.
    """
    if not terms:
        return IndexInfo({}, set())

    # Check each term for errors first
    for t in terms:
        if t.error:
            return t

    # All terms must have same free indices
    reference = terms[0]
    all_dummy = set()
    for t in terms:
        all_dummy |= t.dummy

    for i, t in enumerate(terms[1:], 1):
        if t.free != reference.free:
            # Build a readable description of the mismatch
            ref_desc = _format_free(reference.free)
            cur_desc = _format_free(t.free)
            return IndexInfo({}, set(),
                error=f"Inconsistent free indices across additive terms: "
                      f"term 1 has {ref_desc}, but term {i+1} has {cur_desc}.")

    return IndexInfo(dict(reference.free), all_dummy)


def _format_free(free: dict[str, bool]) -> str:
    if not free:
        return "{}"
    parts = []
    for name in sorted(free):
        pos = "upper" if free[name] else "lower"
        parts.append(f"{name}({pos})")
    return "{" + ", ".join(parts) + "}"


# ─── Parser ───────────────────────────────────────────────────────────────────
# Recursive descent over the token stream.

class Parser:
    """
    Parse a token stream into index information, respecting:
    - Parenthetical grouping: (...)
    - Additive terms: +, -
    - Implicit multiplication: juxtaposition of atoms/factors
    - Equation sides: =
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Optional[Token]:
        """Look at the current token without consuming it (skips whitespace)."""
        p = self.pos
        while p < len(self.tokens) and self.tokens[p].type == T_SPACE:
            p += 1
        if p < len(self.tokens):
            return self.tokens[p]
        return None

    def _advance(self) -> Optional[Token]:
        """Consume and return the next non-whitespace token."""
        while self.pos < len(self.tokens) and self.tokens[self.pos].type == T_SPACE:
            self.pos += 1
        if self.pos < len(self.tokens):
            tok = self.tokens[self.pos]
            self.pos += 1
            return tok
        return None

    def _skip_whitespace(self):
        while self.pos < len(self.tokens) and self.tokens[self.pos].type == T_SPACE:
            self.pos += 1

    def parse_full(self) -> IndexInfo:
        """
        Parse the entire expression. If there's an '=' sign, treat it as
        an equation and verify both sides match.
        """
        # Split at top-level '=' signs
        sides = self._parse_equation()
        if len(sides) == 1:
            return sides[0]

        # Equation: all sides must have same free indices
        for s in sides:
            if s.error:
                return s

        reference = sides[0]
        for i, s in enumerate(sides[1:], 1):
            if s.free != reference.free:
                ref_desc = _format_free(reference.free)
                cur_desc = _format_free(s.free)
                return IndexInfo({}, set(),
                    error=f"Equation sides have different free indices: "
                          f"LHS has {ref_desc}, RHS has {cur_desc}.")

        # Collect all dummy indices from all sides
        all_dummy = set()
        for s in sides:
            all_dummy |= s.dummy
        return IndexInfo(dict(reference.free), all_dummy)

    def _parse_equation(self) -> list[IndexInfo]:
        """Parse expression, splitting at top-level '='."""
        sides = [self._parse_expr()]
        while True:
            tok = self._peek()
            if tok and tok.type == T_CHAR and tok.text == "=":
                self._advance()  # consume '='
                sides.append(self._parse_expr())
            else:
                break
        return sides

    def _parse_expr(self) -> IndexInfo:
        """
        Expr = Term (('+' | '-') Term)*
        """
        terms = [self._parse_term()]

        while True:
            tok = self._peek()
            if tok and tok.type == T_CHAR and tok.text in ("+", "-"):
                self._advance()  # consume '+' or '-'
                terms.append(self._parse_term())
            else:
                break

        return _merge_sum(terms)

    def _parse_term(self) -> IndexInfo:
        """
        Term = Factor (implicit-multiply Factor)*
        
        Implicit multiplication: two factors next to each other with no
        '+'/'-'/'='/')' between them.  We also handle explicit '*'.
        """
        result = self._parse_factor()

        while True:
            tok = self._peek()
            if tok is None:
                break
            # Stop at additive operators, '=', or closing parens / \right
            if tok.type == T_CHAR and tok.text in ("+", "-", "=", ")"):
                break
            if tok.type == T_COMMAND and tok.text == r"\right":
                break
            # Explicit multiplication sign: consume and continue
            if tok.type == T_CHAR and tok.text == "*":
                self._advance()
                factor = self._parse_factor()
                result = _merge_product(result, factor)
                continue
            # If next thing is the start of a factor, implicit multiply
            if self._is_factor_start(tok):
                factor = self._parse_factor()
                result = _merge_product(result, factor)
            else:
                break

        return result

    def _is_factor_start(self, tok: Token) -> bool:
        """Check if a token can start a new factor (atom or '(' or '\\left')."""
        if tok.type == T_CHAR and tok.text == "(":
            return True
        if tok.type == T_COMMAND and tok.text == r"\left":
            return True
        if tok.type == T_CHAR and tok.text.isalpha():
            return True
        if tok.type == T_COMMAND:
            return True
        if tok.type == T_LBRACE:
            # Could be start of a group
            return True
        return False

    def _parse_factor(self) -> IndexInfo:
        r"""
        Factor = '(' Expr ')' [indices]
               | '\left' delim Expr '\right' delim [indices]
               | Atom
        """
        tok = self._peek()
        if tok is None:
            return IndexInfo({}, set())

        # Plain parentheses: ( Expr )
        if tok.type == T_CHAR and tok.text == "(":
            self._advance()  # consume '('
            inner = self._parse_expr()
            # Consume closing ')'
            close = self._peek()
            if close and close.type == T_CHAR and close.text == ")":
                self._advance()
            # After a parenthesized group, there might be indices on the group
            indices = self._collect_trailing_indices()
            if indices:
                atom_info = self._make_index_info(indices)
                return _merge_product(inner, atom_info)
            return inner

        # \left( ... \right)  (or \left[ ... \right] etc.)
        if tok.type == T_COMMAND and tok.text == r"\left":
            self._advance()  # consume \left
            # Consume the delimiter character after \left (e.g. '(' '[' '.' etc.)
            delim = self._peek()
            if delim is not None:
                self._advance()
            inner = self._parse_expr()
            # Consume \right
            close = self._peek()
            if close and close.type == T_COMMAND and close.text == r"\right":
                self._advance()
                # Consume the delimiter after \right
                delim2 = self._peek()
                if delim2 is not None and delim2.type == T_CHAR:
                    self._advance()
            # Trailing indices on the group
            indices = self._collect_trailing_indices()
            if indices:
                atom_info = self._make_index_info(indices)
                return _merge_product(inner, atom_info)
            return inner

        return self._parse_atom()

    def _parse_atom(self) -> IndexInfo:
        r"""
        Atom = symbol [sub/super indices]*
        
        A symbol is a single letter, a command (\alpha, \Gamma, etc.),
        or a digit sequence.  After it we collect any subscript/superscript
        index groups.
        """
        tok = self._peek()
        if tok is None:
            return IndexInfo({}, set())

        # Consume the base symbol
        if tok.type == T_CHAR and tok.text.isalpha():
            self._advance()
        elif tok.type == T_COMMAND:
            # \left and \right are structural delimiters, not tensor symbols.
            # If we reach here it means the factor handler didn't catch them
            # (e.g. mismatched delimiters).  Skip without collecting indices.
            if tok.text in (r"\left", r"\right"):
                self._advance()
                return IndexInfo({}, set())
            self._advance()
        elif tok.type == T_LBRACE:
            # Brace group as a factor — skip through matching braces
            self._skip_brace_group()
        else:
            # Not a recognizable atom start — skip one token
            self._advance()
            return IndexInfo({}, set())

        # Collect trailing indices
        indices = self._collect_trailing_indices()
        return self._make_index_info(indices)

    def _skip_brace_group(self):
        """Skip a { ... } group including nested braces."""
        tok = self._advance()  # consume '{'
        depth = 1
        while depth > 0:
            tok = self._advance()
            if tok is None:
                break
            if tok.type == T_LBRACE:
                depth += 1
            elif tok.type == T_RBRACE:
                depth -= 1

    def _collect_trailing_indices(self) -> list[Index]:
        """
        Collect all subscript/superscript index groups following the current
        position.  Returns a list of Index objects.
        """
        indices: list[Index] = []

        while True:
            self._skip_whitespace()
            if self.pos >= len(self.tokens):
                break
            tok = self.tokens[self.pos]

            if tok.type not in (T_SUBSCRIPT, T_SUPER):
                break

            is_upper = (tok.type == T_SUPER)
            self.pos += 1  # consume _ or ^
            self._skip_whitespace()

            if self.pos >= len(self.tokens):
                break

            tok = self.tokens[self.pos]
            if tok.type == T_LBRACE:
                # Multi-index group: collect everything inside { ... }
                self.pos += 1  # consume {
                self._skip_whitespace()
                while self.pos < len(self.tokens) and self.tokens[self.pos].type != T_RBRACE:
                    t = self.tokens[self.pos]
                    sym = self._index_symbol(t)
                    if sym is not None:
                        indices.append(Index(sym, is_upper))
                    self.pos += 1
                if self.pos < len(self.tokens):
                    self.pos += 1  # consume }
            else:
                # Single-token index
                sym = self._index_symbol(tok)
                if sym is not None:
                    indices.append(Index(sym, is_upper))
                self.pos += 1

        return indices

    def _index_symbol(self, tok: Token) -> Optional[str]:
        """Return the symbol string if this token can be an index, else None."""
        if tok.type == T_CHAR and re.match(r"[A-Za-z]", tok.text):
            return tok.text
        if tok.type == T_COMMAND:
            return tok.text
        # Skip commas, spaces, digits-in-indices, backslash-space, etc.
        return None

    def _make_index_info(self, indices: list[Index]) -> IndexInfo:
        """
        Given a list of indices on a single tensor, determine which are
        free (appear once) and which are dummy (appear twice).
        Three or more occurrences is an error.
        """
        count: dict[str, int] = {}
        position: dict[str, bool] = {}  # name -> is_upper (first occurrence)

        for idx in indices:
            count[idx.name] = count.get(idx.name, 0) + 1
            if idx.name not in position:
                position[idx.name] = idx.is_upper

        free = {}
        dummy = set()
        for name, cnt in count.items():
            if cnt == 1:
                free[name] = position[name]
            elif cnt == 2:
                dummy.add(name)
            else:
                return IndexInfo({}, set(),
                    error=f"Index '{name}' appears {cnt} times on a single "
                          f"tensor (maximum is 2).")

        return IndexInfo(free, dummy)


# ─── Main verification function ───────────────────────────────────────────────

@dataclass
class VerificationResult:
    well_formed: bool
    free_indices: dict[str, bool]   # name -> is_upper
    dummy_indices: set[str]
    error: Optional[str] = None

    def __str__(self):
        if not self.well_formed:
            return f"ILL-FORMED: {self.error}"

        free_str = ", ".join(sorted(self.free_indices.keys())) or "(none)"
        dummy_str = ", ".join(sorted(self.dummy_indices)) or "(none)"
        return f"Well-formed. Free indices: {free_str}. Dummy indices: {dummy_str}."


def verify(latex: str) -> VerificationResult:
    """
    Verify that a LaTeX tensor expression is well-formed under Einstein
    summation convention.
    
    Returns a VerificationResult with:
      - well_formed: bool
      - free_indices: dict mapping name -> is_upper
      - dummy_indices: set of names
      - error: description if ill-formed
    """
    tokens = tokenize(latex)
    parser = Parser(tokens)
    info = parser.parse_full()

    if info.error:
        return VerificationResult(
            well_formed=False,
            free_indices={},
            dummy_indices=set(),
            error=info.error,
        )

    return VerificationResult(
        well_formed=True,
        free_indices=info.free,
        dummy_indices=info.dummy,
    )


# ─── CLI interface ────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        expr = " ".join(sys.argv[1:])
    else:
        expr = input("Enter LaTeX expression: ")

    result = verify(expr)
    print(result)


if __name__ == "__main__":
    main()
