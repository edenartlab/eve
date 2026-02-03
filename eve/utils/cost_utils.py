from __future__ import annotations

from typing import Any, Dict, List

from pyparsing import (
    Forward,
    Group,
    Keyword,
    Literal,
    ParserElement,
    QuotedString,
    Suppress,
    Word,
    ZeroOrMore,
    alphanums,
    alphas,
    infixNotation,
    oneOf,
    opAssoc,
    pyparsing_common,
)

# Enable memoisation to speed up recursive parsing.  This has a global effect but does not interfere with other parsers in normal usage.
ParserElement.enablePackrat()


def _sum_lengths(array: List[Any], property_path: str) -> int:
    """Sum the lengths of a property across all items in an array.

    Parameters
    ----------
    array : List[Any]
        A list of objects (dicts or objects with attributes).
    property_path : str
        A dot-separated path to the property whose length to sum.
        E.g., "text" or "nested.field"

    Returns
    -------
    int
        The sum of len(item[property]) for all items.
    """
    total = 0
    for item in array:
        val = item
        for prop in property_path.split("."):
            if isinstance(val, dict):
                val = val.get(prop, "")
            else:
                val = getattr(val, prop, "")
        if val is not None:
            total += len(val)
    return total


def _build_expression_parser(variables: Dict[str, Any]) -> ParserElement:
    """Construct a pyparsing parser for evaluating JS‑style expressions.

    The returned parser will substitute values from ``variables`` when
    encountering identifiers.  Evaluation is performed by parse actions
    attached to the grammar elements; no further processing is necessary
    once parsing succeeds.

    Parameters
    ----------
    variables : Dict[str, Any]
        A mapping of variable names to Python values used when
        evaluating the expression.

    Returns
    -------
    ParserElement
        A parser configured to evaluate an expression and return the
        resulting Python value.
    """
    # Define number literals.  Use copy() because the default instances
    # attach parse actions that return numbers in lists; copy() preserves
    # these semantics while allowing us to attach our own parse action.
    integer = pyparsing_common.signed_integer.copy().setParseAction(
        lambda t: [int(t[0])]
    )
    real = pyparsing_common.fnumber.copy().setParseAction(lambda t: [float(t[0])])
    number = real | integer

    # Define string literals (single or double quoted).  These return
    # Python str values when parsed.  The escChar parameter ensures
    # backslash escapes are processed correctly.
    single_quoted = QuotedString("'", escChar="\\").setParseAction(
        lambda t: [str(t[0])]
    )
    double_quoted = QuotedString('"', escChar="\\").setParseAction(
        lambda t: [str(t[0])]
    )
    string = single_quoted | double_quoted

    # Define booleans and null/None.  Keywords are case‑insensitive.
    true_literal = (Keyword("true", caseless=True) | Keyword("True")).setParseAction(
        lambda: [True]
    )
    false_literal = (Keyword("false", caseless=True) | Keyword("False")).setParseAction(
        lambda: [False]
    )
    null_literal = (Keyword("null", caseless=True) | Keyword("None")).setParseAction(
        lambda: [None]
    )

    # Forward declarations for recursive grammar elements.
    expr: Forward = Forward()

    # Identifiers: variable names consisting of letters, digits and
    # underscores.  When encountered, look up the value in ``variables``.
    ident_name = Word(alphas + "_", alphanums + "_")

    # Simple identifier lookup (returns the value from variables)
    ident = ident_name.copy().setParseAction(lambda t: [variables.get(t[0], None)])

    # Function arguments: simple values only (idents or strings) to avoid recursion
    func_arg = ident | string
    func_args = Group(func_arg + ZeroOrMore(Suppress(",") + func_arg))

    # Function call: funcName(arg1, arg2, ...)
    # We support sumLengths(array, "property") built-in
    func_call = ident_name + Suppress("(") + func_args + Suppress(")")

    def func_call_action(tokens):
        func_name = tokens[0]
        args = list(tokens[1]) if len(tokens) > 1 else []
        if func_name == "sumLengths":
            if len(args) != 2:
                raise ValueError(
                    f"sumLengths requires 2 arguments: array and property, got {len(args)}: {args}"
                )
            return _sum_lengths(args[0], args[1])
        raise ValueError(f"Unknown function: {func_name}")

    func_call.setParseAction(func_call_action)

    # Atomic operands: numbers, strings, booleans, null/None, function calls, or identifiers.
    atom = (
        number
        | string
        | true_literal
        | false_literal
        | null_literal
        | func_call
        | ident
    )

    # Parenthesised expressions.  Suppress the parentheses so they do not
    # clutter the parse result.  The enclosed expression is parsed
    # recursively by referencing ``expr``.
    _base_operand = atom | (Suppress("(") + expr + Suppress(")"))

    # Support postfix accessors: .property, [index], and .length
    # These can be chained: segments[0].text.length
    dot_access = Suppress(".") + ident_name
    bracket_access = Suppress("[") + expr + Suppress("]")
    length_suffix = Suppress(".") + Keyword("length")

    accessor = dot_access | bracket_access | length_suffix

    def postfix_action(tokens):
        val = tokens[0]
        for i in range(1, len(tokens)):
            accessor_token = tokens[i]
            if accessor_token == "length":
                try:
                    val = len(val)
                except TypeError as e:
                    raise ValueError(
                        f"Cannot take .length of value of type {type(val).__name__}"
                    ) from e
            elif isinstance(accessor_token, int):
                # Array index access
                try:
                    val = val[accessor_token]
                except (IndexError, KeyError, TypeError) as e:
                    raise ValueError(
                        f"Cannot access index {accessor_token}: {e}"
                    ) from e
            else:
                # Property access
                prop = str(accessor_token)
                if isinstance(val, dict):
                    val = val.get(prop)
                else:
                    val = getattr(val, prop, None)
        return val

    operand = (_base_operand + ZeroOrMore(accessor)).setParseAction(postfix_action)

    # Define evaluation functions for unary and binary operators.  The
    # parse actions receive a nested list structure representing the
    # operator and its operand(s) and must return a single evaluated
    # Python value.
    def unary_eval(tokens):
        op = tokens[0][0]
        val = tokens[0][1]
        if op == "-":
            return -val
        if op == "+":
            return +val
        if op in ("!", "not"):
            return not val
        raise ValueError(f"Unknown unary operator: {op}")

    def binary_eval(tokens):
        values = tokens[0]
        result = values[0]
        for i in range(1, len(values), 2):
            op = values[i]
            right = values[i + 1]
            if op == "+":
                result = result + right
            elif op == "-":
                result = result - right
            elif op == "*":
                result = result * right
            elif op == "/":
                result = result / right
            elif op == "%":
                result = result % right
            elif op == "<":
                result = result < right
            elif op == "<=":
                result = result <= right
            elif op == ">":
                result = result > right
            elif op == ">=":
                result = result >= right
            elif op in ("==", "==="):
                # JavaScript's == and === are both treated as Python ==.
                result = result == right
            elif op in ("!=", "!=="):
                result = result != right
            elif op in ("and", "&&"):
                result = result and right
            elif op in ("or", "||"):
                result = result or right
            else:
                raise ValueError(f"Unknown binary operator: {op}")
        return result

    # Use infixNotation (also known as operatorPrecedence) to declare
    # operator precedence and associativity.  Operators are listed from
    # highest precedence to lowest.  For each operator level we provide
    # the parse action that combines the operands.
    cond_expr = infixNotation(
        operand,
        [
            (oneOf("! not"), 1, opAssoc.RIGHT, unary_eval),
            (oneOf("* / %"), 2, opAssoc.LEFT, binary_eval),
            (oneOf("+ -"), 2, opAssoc.LEFT, binary_eval),
            (oneOf("< <= > >="), 2, opAssoc.LEFT, binary_eval),
            (oneOf("== != === !=="), 2, opAssoc.LEFT, binary_eval),
            ((Literal("and") | Literal("&&")), 2, opAssoc.LEFT, binary_eval),
            ((Literal("or") | Literal("||")), 2, opAssoc.LEFT, binary_eval),
        ],
    )

    # Ternary expression: cond_expr ? expr : expr
    ternary = cond_expr + Suppress("?") + expr + Suppress(":") + expr

    def ternary_action(tokens):
        # ``tokens`` is a list: [cond_val, true_val, false_val]
        cond_val, true_val, false_val = tokens
        return true_val if cond_val else false_val

    ternary.setParseAction(ternary_action)

    # The full expression can be a ternary or a conditional expression.
    expr <<= ternary | cond_expr
    return expr


def _coerce_value(value: Any) -> Any:
    """Auto-coerce string values that look like numbers or booleans."""
    if not isinstance(value, str):
        return value
    # Check for boolean strings
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    # Try integer first
    try:
        return int(value)
    except ValueError:
        pass
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    # Return as-is if not numeric
    return value


def eval_cost(expression: str, **variables: Any) -> Any:
    """Evaluate a JavaScript‑style expression in Python.

    The expression may contain nested ternary operators (``a ? b : c``),
    comparison operators (``==``, ``!=``, ``<``, ``<=``,
    ``>``, ``>=``), logical operators (``&&``, ``||``, ``!``/``not``),
    arithmetic operations (``+``, ``-``, ``*``, ``/``, ``%``) and
    parentheses for grouping.  JavaScript boolean literals
    (``true``/``false``), the ``null`` literal (interpreted as Python
    ``None``) and string literals (single or double quoted) are
    recognised.  Variable names consisting of letters, digits and
    underscores are looked up in the keyword arguments provided to this
    function.

    Examples
    --------
    >>> expr = '(output == "video" ? (((quality == "pro" ? 40 : 20) + (sound_effects ? 5 : 0)) * duration) : (5 * n_samples))'
    >>> eval_cost(expr, duration=10, quality="pro", sound_effects=None, n_samples=5, output="video")
    400

    The same expression evaluated with a different ``output`` value:

    >>> eval_cost(expr, duration=10, quality="pro", sound_effects=True, n_samples=5, output="audio")
    25

    Parameters
    ----------
    expression : str
        A string containing the expression written using JavaScript
        syntax.  The expression must be a valid expression (no
        statements) and may include nested ternaries.
    **variables : Any
        Keyword arguments mapping variable names used in ``expression`` to
        their Python values.

    Returns
    -------
    Any
        The value of the evaluated expression.  Numeric results that
        happen to be integers will be returned as Python ``int`` values;
        other numeric results are returned as ``float``.  Strings,
        booleans and ``None`` values are returned unchanged.
    """
    # Auto-coerce string values that look like numbers or booleans
    variables = {k: _coerce_value(v) for k, v in variables.items()}
    # Build a parser configured with the provided variables.  This
    # parser encapsulates the evaluation logic via parse actions.
    parser = _build_expression_parser(variables)
    try:
        result = parser.parseString(expression, parseAll=True)[0]
    except Exception as exc:
        # Re‑raise with additional context for easier debugging.
        raise ValueError(
            f"Failed to evaluate expression '{expression}': {exc}"
        ) from exc
    # Coerce floats that are mathematically integers back to int for
    # convenience.  Many arithmetic operations produce floats via the
    # numeric grammar; this step normalises results such as 25.0 to 25.
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result
