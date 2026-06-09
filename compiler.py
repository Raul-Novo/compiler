from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal


DEFAULT_LIBPATHS: tuple[str, ...] = ()
MSVC_LIBRARIES: tuple[str, ...] = (
    "legacy_stdio_definitions.lib",
    "ucrt.lib",
    "vcruntime.lib",
    "msvcrt.lib",
)

ExpressionType = Literal["int", "string"]


class CompilerError(Exception):
    def __init__(
        self,
        message: str,
        *,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def __str__(self) -> str:
        if self.line is None:
            return f"[ERROR] {self.message}"
        if self.column is None:
            return f"[ERROR] Line {self.line}: {self.message}"
        return f"[ERROR] Line {self.line}, column {self.column}: {self.message}"


class TokenKind(Enum):
    IDENTIFIER = "identifier"
    INTEGER = "integer"
    STRING = "string"
    KEYWORD = "keyword"
    OPERATOR = "operator"
    NEWLINE = "newline"
    EOF = "eof"


@dataclass(frozen=True, slots=True)
class Token:
    kind: TokenKind
    value: str
    line: int
    column: int


KEYWORDS: frozenset[str] = frozenset(
    {
        "let",
        "if",
        "else",
        "end",
        "while",
        "print",
        "input",
        "exit",
    }
)


class Lexer:
    def __init__(self, source: str) -> None:
        self.source = source
        self.index = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []

    def lex(self) -> list[Token]:
        while not self.is_at_end():
            char = self.current_char()

            if char in " \t\f\v":
                self.advance()
                continue

            if char == "#":
                self.skip_comment()
                continue

            if char == "\n" or char == "\r":
                self.lex_newline()
                continue

            if char == '"':
                self.tokens.append(self.lex_string())
                continue

            if char.isdigit():
                self.tokens.append(self.lex_integer())
                continue

            if char.isalpha() or char == "_":
                self.tokens.append(self.lex_identifier_or_keyword())
                continue

            self.tokens.append(self.lex_operator())

        self.tokens.append(Token(TokenKind.EOF, "", self.line, self.column))
        return self.tokens

    def is_at_end(self) -> bool:
        return self.index >= len(self.source)

    def current_char(self) -> str:
        if self.is_at_end():
            return "\0"
        return self.source[self.index]

    def peek_char(self, offset: int = 1) -> str:
        position = self.index + offset
        if position >= len(self.source):
            return "\0"
        return self.source[position]

    def advance(self) -> str:
        char = self.current_char()
        self.index += 1
        self.column += 1
        return char

    def lex_newline(self) -> None:
        start_line = self.line
        start_column = self.column

        if self.current_char() == "\r" and self.peek_char() == "\n":
            self.index += 2
        else:
            self.index += 1

        self.tokens.append(Token(TokenKind.NEWLINE, "\n", start_line, start_column))
        self.line += 1
        self.column = 1

    def skip_comment(self) -> None:
        while not self.is_at_end() and self.current_char() not in "\r\n":
            self.advance()

    def lex_string(self) -> Token:
        start_line = self.line
        start_column = self.column
        self.advance()
        chars: list[str] = []

        while not self.is_at_end():
            char = self.current_char()

            if char == '"':
                self.advance()
                return Token(TokenKind.STRING, "".join(chars), start_line, start_column)

            if char in "\r\n":
                raise CompilerError(
                    "unterminated string literal",
                    line=start_line,
                    column=start_column,
                )

            if char == "\\":
                chars.append(self.lex_escape_sequence(start_line, start_column))
                continue

            chars.append(char)
            self.advance()

        raise CompilerError(
            "unterminated string literal",
            line=start_line,
            column=start_column,
        )

    def lex_escape_sequence(self, start_line: int, start_column: int) -> str:
        self.advance()
        char = self.current_char()
        escapes: dict[str, str] = {
            "n": "\n",
            "r": "\r",
            "t": "\t",
            '"': '"',
            "\\": "\\",
            "0": "\0",
        }

        if char not in escapes:
            raise CompilerError(
                f"unknown escape sequence '\\{char}'",
                line=start_line,
                column=start_column,
            )

        self.advance()
        return escapes[char]

    def lex_integer(self) -> Token:
        start = self.index
        start_line = self.line
        start_column = self.column

        while self.current_char().isdigit():
            self.advance()

        return Token(
            TokenKind.INTEGER,
            self.source[start:self.index],
            start_line,
            start_column,
        )

    def lex_identifier_or_keyword(self) -> Token:
        start = self.index
        start_line = self.line
        start_column = self.column

        while self.current_char().isalnum() or self.current_char() == "_":
            self.advance()

        value = self.source[start:self.index]
        kind = TokenKind.KEYWORD if value in KEYWORDS else TokenKind.IDENTIFIER
        return Token(kind, value, start_line, start_column)

    def lex_operator(self) -> Token:
        start_line = self.line
        start_column = self.column
        char = self.current_char()
        two_chars = char + self.peek_char()

        if two_chars in {"==", "!=", "<=", ">="}:
            self.advance()
            self.advance()
            return Token(TokenKind.OPERATOR, two_chars, start_line, start_column)

        if char in "+-*/%=<>()":
            self.advance()
            return Token(TokenKind.OPERATOR, char, start_line, start_column)

        raise CompilerError(
            f"unexpected character '{char}'",
            line=start_line,
            column=start_column,
        )


@dataclass(frozen=True, slots=True)
class Program:
    statements: list[Statement]


@dataclass(frozen=True, slots=True)
class Statement:
    line: int


@dataclass(frozen=True, slots=True)
class LetStatement(Statement):
    name: str
    initializer: Expression


@dataclass(frozen=True, slots=True)
class AssignStatement(Statement):
    name: str
    value: Expression


@dataclass(frozen=True, slots=True)
class PrintStatement(Statement):
    value: Expression


@dataclass(frozen=True, slots=True)
class IfStatement(Statement):
    condition: Expression
    then_body: list[Statement]
    else_body: list[Statement] | None


@dataclass(frozen=True, slots=True)
class WhileStatement(Statement):
    condition: Expression
    body: list[Statement]


@dataclass(frozen=True, slots=True)
class ExitStatement(Statement):
    code: Expression | None


@dataclass(frozen=True, slots=True)
class Expression:
    line: int


@dataclass(frozen=True, slots=True)
class IntegerLiteral(Expression):
    value: int


@dataclass(frozen=True, slots=True)
class StringLiteral(Expression):
    value: str


@dataclass(frozen=True, slots=True)
class VariableExpression(Expression):
    name: str


@dataclass(frozen=True, slots=True)
class InputExpression(Expression):
    pass


@dataclass(frozen=True, slots=True)
class UnaryExpression(Expression):
    operator: str
    operand: Expression


@dataclass(frozen=True, slots=True)
class BinaryExpression(Expression):
    left: Expression
    operator: str
    right: Expression


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.index = 0

    def parse(self) -> Program:
        statements = self.parse_block(stop_keywords=frozenset())
        self.skip_newlines()

        if not self.check(TokenKind.EOF):
            token = self.current()
            raise CompilerError(
                f"unexpected token {self.describe_token(token)}",
                line=token.line,
                column=token.column,
            )

        return Program(statements)

    def parse_block(self, *, stop_keywords: frozenset[str]) -> list[Statement]:
        statements: list[Statement] = []

        while True:
            self.skip_newlines()

            if self.check(TokenKind.EOF):
                break

            token = self.current()
            if token.kind == TokenKind.KEYWORD and token.value in stop_keywords:
                break

            statements.append(self.parse_statement())

        return statements

    def parse_statement(self) -> Statement:
        token = self.current()

        if token.kind == TokenKind.KEYWORD:
            if token.value == "let":
                return self.parse_let_statement()
            if token.value == "print":
                return self.parse_print_statement()
            if token.value == "if":
                return self.parse_if_statement()
            if token.value == "while":
                return self.parse_while_statement()
            if token.value == "exit":
                return self.parse_exit_statement()
            if token.value == "else":
                raise CompilerError(
                    "else without matching if",
                    line=token.line,
                    column=token.column,
                )
            if token.value == "end":
                raise CompilerError(
                    "end without open block",
                    line=token.line,
                    column=token.column,
                )

        if token.kind == TokenKind.IDENTIFIER:
            return self.parse_assign_statement()

        raise CompilerError(
            f"expected statement, found {self.describe_token(token)}",
            line=token.line,
            column=token.column,
        )

    def parse_let_statement(self) -> LetStatement:
        start = self.consume_keyword("let")
        name = self.expect_identifier("expected variable name after 'let'")
        self.expect_operator("=", "expected '=' after variable name")

        if self.is_expression_end():
            raise CompilerError("expected expression after '='", line=self.current().line)

        initializer = self.parse_expression()
        self.require_statement_end()
        return LetStatement(start.line, name.value, initializer)

    def parse_assign_statement(self) -> AssignStatement:
        name = self.expect_identifier("expected variable name")
        self.expect_operator("=", "expected '=' after variable name")

        if self.is_expression_end():
            raise CompilerError("expected expression after '='", line=self.current().line)

        value = self.parse_expression()
        self.require_statement_end()
        return AssignStatement(name.line, name.value, value)

    def parse_print_statement(self) -> PrintStatement:
        start = self.consume_keyword("print")

        if self.is_expression_end():
            raise CompilerError(
                "expected expression after 'print'",
                line=start.line,
                column=start.column,
            )

        value = self.parse_expression()
        self.require_statement_end()
        return PrintStatement(start.line, value)

    def parse_if_statement(self) -> IfStatement:
        start = self.consume_keyword("if")

        if self.is_expression_end():
            raise CompilerError(
                "expected condition after 'if'",
                line=start.line,
                column=start.column,
            )

        condition = self.parse_expression()
        self.require_statement_end()
        then_body = self.parse_block(stop_keywords=frozenset({"else", "end"}))
        else_body: list[Statement] | None = None

        if self.match_keyword("else") is not None:
            self.require_statement_end()
            else_body = self.parse_block(stop_keywords=frozenset({"end"}))

        if self.match_keyword("end") is None:
            current = self.current()
            raise CompilerError(
                f"missing 'end' for if started on line {start.line}",
                line=current.line,
                column=current.column,
            )

        self.require_statement_end()
        return IfStatement(start.line, condition, then_body, else_body)

    def parse_while_statement(self) -> WhileStatement:
        start = self.consume_keyword("while")

        if self.is_expression_end():
            raise CompilerError(
                "expected condition after 'while'",
                line=start.line,
                column=start.column,
            )

        condition = self.parse_expression()
        self.require_statement_end()
        body = self.parse_block(stop_keywords=frozenset({"end"}))

        if self.match_keyword("end") is None:
            current = self.current()
            raise CompilerError(
                f"missing 'end' for while started on line {start.line}",
                line=current.line,
                column=current.column,
            )

        self.require_statement_end()
        return WhileStatement(start.line, condition, body)

    def parse_exit_statement(self) -> ExitStatement:
        start = self.consume_keyword("exit")
        code: Expression | None = None

        if not self.is_expression_end():
            code = self.parse_expression()

        self.require_statement_end()
        return ExitStatement(start.line, code)

    def parse_expression(self) -> Expression:
        return self.parse_equality()

    def parse_equality(self) -> Expression:
        expression = self.parse_comparison()

        while (operator := self.match_operator("==", "!=")) is not None:
            right = self.parse_comparison()
            expression = BinaryExpression(operator.line, expression, operator.value, right)

        return expression

    def parse_comparison(self) -> Expression:
        expression = self.parse_term()

        while (operator := self.match_operator("<", "<=", ">", ">=")) is not None:
            right = self.parse_term()
            expression = BinaryExpression(operator.line, expression, operator.value, right)

        return expression

    def parse_term(self) -> Expression:
        expression = self.parse_factor()

        while (operator := self.match_operator("+", "-")) is not None:
            right = self.parse_factor()
            expression = BinaryExpression(operator.line, expression, operator.value, right)

        return expression

    def parse_factor(self) -> Expression:
        expression = self.parse_unary()

        while (operator := self.match_operator("*", "/", "%")) is not None:
            right = self.parse_unary()
            expression = BinaryExpression(operator.line, expression, operator.value, right)

        return expression

    def parse_unary(self) -> Expression:
        operator = self.match_operator("+", "-")
        if operator is None:
            return self.parse_primary()

        operand = self.parse_unary()
        return UnaryExpression(operator.line, operator.value, operand)

    def parse_primary(self) -> Expression:
        token = self.current()

        if token.kind == TokenKind.INTEGER:
            self.advance()
            return IntegerLiteral(token.line, int(token.value))

        if token.kind == TokenKind.STRING:
            self.advance()
            return StringLiteral(token.line, token.value)

        if token.kind == TokenKind.IDENTIFIER:
            self.advance()
            return VariableExpression(token.line, token.value)

        if token.kind == TokenKind.KEYWORD and token.value == "input":
            self.advance()
            self.expect_operator("(", "expected '(' after 'input'")
            self.expect_operator(")", "expected ')' after 'input('")
            return InputExpression(token.line)

        if self.match_operator("(") is not None:
            expression = self.parse_expression()
            self.expect_operator(")", "expected ')' after expression")
            return expression

        if token.kind in {TokenKind.NEWLINE, TokenKind.EOF}:
            raise CompilerError("expected expression", line=token.line)

        raise CompilerError(
            f"expected expression, found {self.describe_token(token)}",
            line=token.line,
            column=token.column,
        )

    def current(self) -> Token:
        return self.tokens[self.index]

    def advance(self) -> Token:
        token = self.current()
        if token.kind != TokenKind.EOF:
            self.index += 1
        return token

    def check(self, kind: TokenKind, value: str | None = None) -> bool:
        token = self.current()
        if token.kind != kind:
            return False
        if value is not None and token.value != value:
            return False
        return True

    def match_keyword(self, value: str) -> Token | None:
        if self.check(TokenKind.KEYWORD, value):
            return self.advance()
        return None

    def consume_keyword(self, value: str) -> Token:
        token = self.match_keyword(value)
        if token is None:
            current = self.current()
            raise CompilerError(
                f"expected keyword '{value}'",
                line=current.line,
                column=current.column,
            )
        return token

    def match_operator(self, *values: str) -> Token | None:
        token = self.current()
        if token.kind == TokenKind.OPERATOR and token.value in values:
            return self.advance()
        return None

    def expect_operator(self, value: str, message: str) -> Token:
        token = self.match_operator(value)
        if token is None:
            current = self.current()
            raise CompilerError(message, line=current.line, column=current.column)
        return token

    def expect_identifier(self, message: str) -> Token:
        if self.check(TokenKind.IDENTIFIER):
            return self.advance()
        token = self.current()
        raise CompilerError(message, line=token.line, column=token.column)

    def skip_newlines(self) -> None:
        while self.check(TokenKind.NEWLINE):
            self.advance()

    def require_statement_end(self) -> None:
        if self.check(TokenKind.NEWLINE):
            self.skip_newlines()
            return

        if self.check(TokenKind.EOF):
            return

        token = self.current()
        raise CompilerError(
            f"unexpected token {self.describe_token(token)} at end of statement",
            line=token.line,
            column=token.column,
        )

    def is_expression_end(self) -> bool:
        return self.check(TokenKind.NEWLINE) or self.check(TokenKind.EOF)

    @staticmethod
    def describe_token(token: Token) -> str:
        if token.kind == TokenKind.EOF:
            return "end of file"
        if token.kind == TokenKind.NEWLINE:
            return "end of line"
        return repr(token.value)


@dataclass(frozen=True, slots=True)
class SemanticResult:
    variable_names: list[str]


class SemanticAnalyzer:
    def __init__(self) -> None:
        self.all_declared_names: set[str] = set()
        self.variable_names: list[str] = []

    def analyze(self, program: Program) -> SemanticResult:
        visible_names: set[str] = set()
        self.check_block(program.statements, visible_names)
        return SemanticResult(self.variable_names)

    def check_block(self, statements: list[Statement], visible_names: set[str]) -> None:
        for statement in statements:
            self.check_statement(statement, visible_names)

    def check_statement(self, statement: Statement, visible_names: set[str]) -> None:
        if isinstance(statement, LetStatement):
            initializer_type = self.check_expression(statement.initializer, visible_names)
            self.require_int(initializer_type, "variable initializer", statement.line)

            if statement.name in self.all_declared_names:
                raise CompilerError(
                    f"variable '{statement.name}' is already declared",
                    line=statement.line,
                )

            self.all_declared_names.add(statement.name)
            self.variable_names.append(statement.name)
            visible_names.add(statement.name)
            return

        if isinstance(statement, AssignStatement):
            if statement.name not in visible_names:
                raise CompilerError(
                    f"assignment to undeclared variable '{statement.name}'",
                    line=statement.line,
                )

            value_type = self.check_expression(statement.value, visible_names)
            self.require_int(value_type, "assignment value", statement.line)
            return

        if isinstance(statement, PrintStatement):
            self.check_expression(statement.value, visible_names)
            return

        if isinstance(statement, IfStatement):
            condition_type = self.check_expression(statement.condition, visible_names)
            self.require_int(condition_type, "if condition", statement.line)
            self.check_block(statement.then_body, set(visible_names))

            if statement.else_body is not None:
                self.check_block(statement.else_body, set(visible_names))

            return

        if isinstance(statement, WhileStatement):
            condition_type = self.check_expression(statement.condition, visible_names)
            self.require_int(condition_type, "while condition", statement.line)
            self.check_block(statement.body, set(visible_names))
            return

        if isinstance(statement, ExitStatement):
            if statement.code is not None:
                code_type = self.check_expression(statement.code, visible_names)
                self.require_int(code_type, "exit code", statement.line)
            return

        raise AssertionError(f"Unhandled statement type: {type(statement).__name__}")

    def check_expression(
        self,
        expression: Expression,
        visible_names: set[str],
    ) -> ExpressionType:
        if isinstance(expression, IntegerLiteral):
            self.check_i64_range(expression.value, expression.line)
            return "int"

        if isinstance(expression, StringLiteral):
            return "string"

        if isinstance(expression, VariableExpression):
            if expression.name not in visible_names:
                raise CompilerError(
                    f"use of undeclared variable '{expression.name}'",
                    line=expression.line,
                )
            return "int"

        if isinstance(expression, InputExpression):
            return "int"

        if isinstance(expression, UnaryExpression):
            operand_type = self.check_expression(expression.operand, visible_names)
            self.require_int(operand_type, f"unary '{expression.operator}'", expression.line)
            return "int"

        if isinstance(expression, BinaryExpression):
            left_type = self.check_expression(expression.left, visible_names)
            right_type = self.check_expression(expression.right, visible_names)
            self.require_int(left_type, f"operator '{expression.operator}'", expression.line)
            self.require_int(right_type, f"operator '{expression.operator}'", expression.line)

            literal_right = self.literal_integer_value(expression.right)
            if expression.operator in {"/", "%"} and literal_right == 0:
                raise CompilerError(
                    "division by zero literal",
                    line=expression.line,
                )

            return "int"

        raise AssertionError(f"Unhandled expression type: {type(expression).__name__}")

    @staticmethod
    def require_int(actual: ExpressionType, context: str, line: int) -> None:
        if actual != "int":
            raise CompilerError(f"{context} must be an integer", line=line)

    @staticmethod
    def literal_integer_value(expression: Expression) -> int | None:
        if isinstance(expression, IntegerLiteral):
            return expression.value

        if isinstance(expression, UnaryExpression):
            value = SemanticAnalyzer.literal_integer_value(expression.operand)
            if value is None:
                return None
            if expression.operator == "-":
                return -value
            if expression.operator == "+":
                return value

        return None

    @staticmethod
    def check_i64_range(value: int, line: int) -> None:
        if value < -(2**63) or value > 2**63 - 1:
            raise CompilerError("integer literal is outside signed 64-bit range", line=line)


class CodeGenerator:
    def __init__(self, variable_names: list[str]) -> None:
        self.variable_names = variable_names
        self.code_lines: list[str] = []
        self.string_literals: list[tuple[str, str]] = []
        self.label_counter = 0
        self.temp_counter = 0

    def generate(self, program: Program) -> str:
        self.emit_main(program)
        return "\n".join(self.build_assembly_lines()) + "\n"

    def emit_main(self, program: Program) -> None:
        self.code_lines.append("main:")
        self.emit("push rbp")
        self.emit("mov rbp, rsp")
        self.emit("sub rsp, 32")
        self.emit_comment("program body")

        for statement in program.statements:
            self.emit_statement(statement)

        self.emit_comment("default exit code")
        self.emit("xor eax, eax")
        self.code_lines.append("main_return:")
        self.emit("add rsp, 32")
        self.emit("pop rbp")
        self.emit("ret")
        self.code_lines.append("")
        self.emit_runtime_helpers()

    def emit_statement(self, statement: Statement) -> None:
        if isinstance(statement, LetStatement):
            self.emit_comment(f"let {statement.name}")
            self.emit_expression(statement.initializer)
            self.emit(f"mov qword [rel {self.variable_label(statement.name)}], rax")
            return

        if isinstance(statement, AssignStatement):
            self.emit_comment(f"assign {statement.name}")
            self.emit_expression(statement.value)
            self.emit(f"mov qword [rel {self.variable_label(statement.name)}], rax")
            return

        if isinstance(statement, PrintStatement):
            self.emit_print_statement(statement)
            return

        if isinstance(statement, IfStatement):
            self.emit_if_statement(statement)
            return

        if isinstance(statement, WhileStatement):
            self.emit_while_statement(statement)
            return

        if isinstance(statement, ExitStatement):
            self.emit_exit_statement(statement)
            return

        raise AssertionError(f"Unhandled statement type: {type(statement).__name__}")

    def emit_print_statement(self, statement: PrintStatement) -> None:
        if isinstance(statement.value, StringLiteral):
            label = self.add_string_literal(statement.value.value)
            self.emit_comment("print string")
            self.emit(f"lea rcx, [rel {label}]")
            self.emit("call print_string")
            return

        self.emit_comment("print integer")
        self.emit_expression(statement.value)
        self.emit("mov rcx, rax")
        self.emit("call print_int")

    def emit_if_statement(self, statement: IfStatement) -> None:
        else_label = self.new_label("if_else")
        end_label = self.new_label("if_end")

        self.emit_comment("if condition")
        self.emit_expression(statement.condition)
        self.emit("cmp rax, 0")

        if statement.else_body is None:
            self.emit(f"je {end_label}")
            self.emit_comment("if body")
            for child in statement.then_body:
                self.emit_statement(child)
            self.code_lines.append(f"{end_label}:")
            return

        self.emit(f"je {else_label}")
        self.emit_comment("if body")
        for child in statement.then_body:
            self.emit_statement(child)
        self.emit(f"jmp {end_label}")
        self.code_lines.append(f"{else_label}:")
        self.emit_comment("else body")
        for child in statement.else_body:
            self.emit_statement(child)
        self.code_lines.append(f"{end_label}:")

    def emit_while_statement(self, statement: WhileStatement) -> None:
        start_label = self.new_label("while_start")
        end_label = self.new_label("while_end")

        self.code_lines.append(f"{start_label}:")
        self.emit_comment("while condition")
        self.emit_expression(statement.condition)
        self.emit("cmp rax, 0")
        self.emit(f"je {end_label}")
        self.emit_comment("while body")
        for child in statement.body:
            self.emit_statement(child)
        self.emit(f"jmp {start_label}")
        self.code_lines.append(f"{end_label}:")

    def emit_exit_statement(self, statement: ExitStatement) -> None:
        self.emit_comment("exit")
        if statement.code is None:
            self.emit("xor eax, eax")
        else:
            self.emit_expression(statement.code)
        self.emit("jmp main_return")

    def emit_expression(self, expression: Expression) -> None:
        if isinstance(expression, IntegerLiteral):
            self.emit(f"mov rax, {expression.value}")
            return

        if isinstance(expression, VariableExpression):
            self.emit(f"mov rax, qword [rel {self.variable_label(expression.name)}]")
            return

        if isinstance(expression, InputExpression):
            self.emit("call read_int")
            return

        if isinstance(expression, UnaryExpression):
            self.emit_expression(expression.operand)
            if expression.operator == "-":
                self.emit("neg rax")
            return

        if isinstance(expression, BinaryExpression):
            self.emit_binary_expression(expression)
            return

        if isinstance(expression, StringLiteral):
            raise AssertionError("String literals are only emitted by print statements")

        raise AssertionError(f"Unhandled expression type: {type(expression).__name__}")

    def emit_binary_expression(self, expression: BinaryExpression) -> None:
        temp = self.new_temp()
        self.emit_expression(expression.left)
        self.emit(f"mov qword [rel {temp}], rax")
        self.emit_expression(expression.right)
        self.emit("mov rcx, rax")
        self.emit(f"mov rax, qword [rel {temp}]")

        if expression.operator == "+":
            self.emit("add rax, rcx")
            return

        if expression.operator == "-":
            self.emit("sub rax, rcx")
            return

        if expression.operator == "*":
            self.emit("imul rax, rcx")
            return

        if expression.operator == "/":
            self.emit("cqo")
            self.emit("idiv rcx")
            return

        if expression.operator == "%":
            self.emit("cqo")
            self.emit("idiv rcx")
            self.emit("mov rax, rdx")
            return

        jump_condition = {
            "==": "sete",
            "!=": "setne",
            "<": "setl",
            "<=": "setle",
            ">": "setg",
            ">=": "setge",
        }.get(expression.operator)

        if jump_condition is None:
            raise AssertionError(f"Unhandled binary operator: {expression.operator}")

        self.emit("cmp rax, rcx")
        self.emit(f"{jump_condition} al")
        self.emit("movzx rax, al")

    def emit_runtime_helpers(self) -> None:
        self.code_lines.append("; ================================================================")
        self.code_lines.append("; Runtime Helpers")
        self.code_lines.append("; ================================================================")

        self.code_lines.append("print_int:")
        self.emit("push rbp")
        self.emit("mov rbp, rsp")
        self.emit("sub rsp, 32")
        self.emit("mov rdx, rcx")
        self.emit("lea rcx, [rel fmt_int]")
        self.emit("call printf")
        self.emit("add rsp, 32")
        self.emit("pop rbp")
        self.emit("ret")
        self.code_lines.append("")

        self.code_lines.append("print_string:")
        self.emit("push rbp")
        self.emit("mov rbp, rsp")
        self.emit("sub rsp, 32")
        self.emit("mov rdx, rcx")
        self.emit("lea rcx, [rel fmt_string]")
        self.emit("call printf")
        self.emit("add rsp, 32")
        self.emit("pop rbp")
        self.emit("ret")
        self.code_lines.append("")

        self.code_lines.append("read_int:")
        self.emit("push rbp")
        self.emit("mov rbp, rsp")
        self.emit("sub rsp, 32")
        self.emit("lea rdx, [rel input_buffer]")
        self.emit("lea rcx, [rel fmt_scan_int]")
        self.emit("call scanf")
        self.emit("mov rax, qword [rel input_buffer]")
        self.emit("add rsp, 32")
        self.emit("pop rbp")
        self.emit("ret")

    def build_assembly_lines(self) -> list[str]:
        lines: list[str] = []
        lines.extend(
            [
                "; ================================================================",
                "; Header",
                "; ================================================================",
                "bits 64",
                "default rel",
                "",
                "; ================================================================",
                "; Constants",
                "; ================================================================",
                "section .data",
                'fmt_int: db "%lld", 10, 0',
                'fmt_string: db "%s", 10, 0',
                'fmt_scan_int: db "%lld", 0',
            ]
        )

        for label, value in self.string_literals:
            lines.append(f"{label}: db {self.format_db_bytes(value)}")

        lines.extend(
            [
                "",
                "; ================================================================",
                "; Variables",
                "; ================================================================",
                "section .bss",
                "input_buffer: resq 1",
            ]
        )

        for name in self.variable_names:
            lines.append(f"{self.variable_label(name)}: resq 1")

        for index in range(self.temp_counter):
            lines.append(f"tmp_{index}: resq 1")

        lines.extend(
            [
                "",
                "; ================================================================",
                "; Code",
                "; ================================================================",
                "section .text",
                "global main",
                "extern printf",
                "extern scanf",
                "",
            ]
        )
        lines.extend(self.code_lines)
        return lines

    def add_string_literal(self, value: str) -> str:
        label = f"str_{len(self.string_literals)}"
        self.string_literals.append((label, value))
        return label

    def new_label(self, prefix: str) -> str:
        label = f"L_{prefix}_{self.label_counter}"
        self.label_counter += 1
        return label

    def new_temp(self) -> str:
        label = f"tmp_{self.temp_counter}"
        self.temp_counter += 1
        return label

    def emit(self, instruction: str) -> None:
        self.code_lines.append(f"  {instruction}")

    def emit_comment(self, comment: str) -> None:
        self.code_lines.append(f"  ; {comment}")

    @staticmethod
    def variable_label(name: str) -> str:
        return f"var_{name}"

    @staticmethod
    def format_db_bytes(value: str) -> str:
        encoded = value.encode("utf-8") + b"\0"
        return ", ".join(str(byte) for byte in encoded)


@dataclass(frozen=True, slots=True)
class BuildArtifacts:
    source_path: Path
    asm_path: Path
    obj_path: Path
    exe_path: Path


@dataclass(frozen=True, slots=True)
class BuildOptions:
    emit_asm_only: bool
    run_after_build: bool
    libpaths: tuple[str, ...]


def compile_file(source_path: Path, options: BuildOptions) -> BuildArtifacts:
    source_path = source_path.resolve()
    if not source_path.exists():
        raise CompilerError(f"source file not found: {source_path}")

    artifacts = BuildArtifacts(
        source_path=source_path,
        asm_path=source_path.with_suffix(".asm"),
        obj_path=source_path.with_suffix(".obj"),
        exe_path=source_path.with_suffix(".exe"),
    )

    print("[CMD] Parsing")
    source = source_path.read_text(encoding="utf-8-sig")
    tokens = Lexer(source).lex()
    program = Parser(tokens).parse()

    print("[CMD] Checking")
    semantic_result = SemanticAnalyzer().analyze(program)

    print("[CMD] Emitting assembly")
    asm_text = CodeGenerator(semantic_result.variable_names).generate(program)
    artifacts.asm_path.write_text(asm_text, encoding="utf-8", newline="\n")

    if options.emit_asm_only:
        print(f"[OK] Emitted {artifacts.asm_path.name}")
        return artifacts

    print("[CMD] Assembling")
    assemble(artifacts.asm_path, artifacts.obj_path)

    print("[CMD] Linking")
    link(artifacts.obj_path, artifacts.exe_path, options.libpaths)

    print(f"[OK] Built {artifacts.exe_path.name}")

    if options.run_after_build:
        print("[CMD] Running")
        subprocess.run([str(artifacts.exe_path)], check=True)

    return artifacts


def assemble(asm_path: Path, obj_path: Path) -> None:
    nasm = shutil.which("nasm")
    if nasm is None:
        raise CompilerError("NASM not found. Install NASM or add nasm.exe to PATH.")

    run_checked(
        [
            nasm,
            "-f",
            "win64",
            str(asm_path),
            "-o",
            str(obj_path),
        ]
    )


def link(obj_path: Path, exe_path: Path, libpaths: tuple[str, ...]) -> None:
    linker = shutil.which("link.exe")
    if linker is None:
        raise CompilerError('MSVC environment not found. Run "msvc" before compiling.')

    command = [
        linker,
        str(obj_path),
        f"/OUT:{exe_path}",
        "/SUBSYSTEM:CONSOLE",
    ]
    command.extend(f"/LIBPATH:{libpath}" for libpath in libpaths)
    command.extend(MSVC_LIBRARIES)

    try:
        run_checked(command)
    except CompilerError as error:
        raise CompilerError(
            f'{error.message}. Make sure MSVC is active: run "msvc" before compiling.'
        ) from error


def run_checked(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise CompilerError(f"command not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        printable_command = " ".join(command)
        raise CompilerError(
            f"command failed with exit code {error.returncode}: {printable_command}"
        ) from error


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="compiler.py",
        description="Compile a simple educational language to NASM x86-64 for Windows.",
        epilog='This compiler uses MSVC link.exe. Run "msvc" in this terminal before building.',
    )
    parser.add_argument("source", type=Path, help="Source file, for example program.rl")
    parser.add_argument(
        "--emit-asm",
        action="store_true",
        help="Emit only the .asm file and stop before NASM/link.exe.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Do not run NASM or link.exe. Equivalent to --emit-asm.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the generated .exe after a successful build.",
    )
    parser.add_argument(
        "--libpath",
        action="append",
        default=[],
        help="Additional MSVC /LIBPATH value. Can be passed more than once.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    emit_asm_only = bool(args.emit_asm or args.no_build)

    try:
        if args.run and emit_asm_only:
            raise CompilerError("--run cannot be combined with --emit-asm or --no-build")

        options = BuildOptions(
            emit_asm_only=emit_asm_only,
            run_after_build=bool(args.run),
            libpaths=tuple(DEFAULT_LIBPATHS) + tuple(args.libpath),
        )
        compile_file(args.source, options)
        return 0
    except CompilerError as error:
        print(error)
        return 1
    except subprocess.CalledProcessError as error:
        print(f"[ERROR] Command failed with exit code {error.returncode}")
        return error.returncode


if __name__ == "__main__":
    raise SystemExit(main())
