//! Recursive descent parser for SQL statements combined with a Pratt parser for expressions.
//!
//! This module converts spanned tokens into an unbound SQL AST.
//! Prefix and infix expressions use Pratt parsing precedence climbing, resolving
//! operator precedence and associativity without complex recursive-descent nesting.

use crate::error::{BasaltError, Result};
use crate::sql::ast::{
    Expr, Literal, OrderByExpr, SelectItem, SelectStatement, Statement, TableRef,
};
use crate::sql::span::{Span, Spanned};
use crate::sql::token::{Keyword, Token};
use crate::types::coercion::{BinaryOp, UnaryOp};
use crate::types::data_type::DataType;

/// Parser state containing tokens and offset cursor.
pub struct Parser {
    tokens: Vec<Spanned<Token>>,
    position: usize,
}

impl Parser {
    /// Creates a new Parser over the token stream.
    pub fn new(tokens: Vec<Spanned<Token>>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    /// Parses a single SQL statement from the token stream.
    /// Fails if extra tokens remain after a parsed statement (except semicolon).
    pub fn parse_statement(&mut self) -> Result<Statement> {
        let stmt = match self.peek() {
            Token::Keyword(Keyword::Select) => {
                let select = self.parse_select()?;
                Statement::Select(select)
            }
            other => {
                return Err(BasaltError::Syntax {
                    span: self.current_span(),
                    message: format!("expected SELECT statement, found {other}"),
                });
            }
        };

        if self.matches(&Token::Semicolon) {
            // consume optional trailing semicolon
        }

        if !self.matches(&Token::Eof) {
            return Err(BasaltError::Syntax {
                span: self.current_span(),
                message: format!("expected EOF, found {}", self.peek()),
            });
        }

        Ok(stmt)
    }

    fn peek(&self) -> &Token {
        self.tokens
            .get(self.position)
            .map(|t| &t.value)
            .unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> &Spanned<Token> {
        let prev = &self.tokens[self.position];
        if self.position < self.tokens.len() - 1 {
            self.position += 1;
        }
        prev
    }

    fn current_span(&self) -> Span {
        self.tokens
            .get(self.position)
            .map(|t| t.span)
            .unwrap_or_else(|| {
                let last_pos = self.tokens.last().map(|t| t.span.end).unwrap_or(0);
                Span::new(last_pos, last_pos)
            })
    }

    fn expect(&mut self, expected: Token) -> Result<()> {
        let token = self.advance();
        if token.value == expected {
            Ok(())
        } else {
            Err(BasaltError::Syntax {
                span: token.span,
                message: format!("expected {expected}, found {}", token.value),
            })
        }
    }

    fn expect_identifier(&mut self) -> Result<String> {
        let token = self.advance();
        match &token.value {
            Token::Identifier(s) => Ok(s.clone()),
            other => Err(BasaltError::Syntax {
                span: token.span,
                message: format!("expected identifier, found {other}"),
            }),
        }
    }

    fn matches(&mut self, token: &Token) -> bool {
        if self.peek() == token {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Parses a SQL SELECT statement and its structural query clauses.
    fn parse_select(&mut self) -> Result<SelectStatement> {
        self.expect(Token::Keyword(Keyword::Select))?;

        let projections = self.parse_projections()?;

        self.expect(Token::Keyword(Keyword::From))?;

        let from = self.parse_table_ref()?;

        let selection = if self.matches(&Token::Keyword(Keyword::Where)) {
            Some(self.parse_expr(0)?)
        } else {
            None
        };

        let order_by = if self.matches(&Token::Keyword(Keyword::OrderBy)) {
            self.parse_order_by()?
        } else {
            Vec::new()
        };

        let limit = if self.matches(&Token::Keyword(Keyword::Limit)) {
            let token = self.advance();
            match &token.value {
                Token::Number(s) => s.parse::<u64>().map(Some).map_err(|_| BasaltError::Syntax {
                    span: token.span,
                    message: format!("invalid limit number '{s}'"),
                }),
                other => Err(BasaltError::Syntax {
                    span: token.span,
                    message: format!("expected number for LIMIT, found {other}"),
                }),
            }?
        } else {
            None
        };

        Ok(SelectStatement {
            projections,
            from,
            selection,
            order_by,
            limit,
        })
    }

    /// Parses comma-separated projection clauses.
    fn parse_projections(&mut self) -> Result<Vec<SelectItem>> {
        let mut projections = Vec::new();
        loop {
            if self.peek() == &Token::Star {
                self.advance();
                projections.push(SelectItem::Wildcard);
            } else {
                let expr = self.parse_expr(0)?;
                let alias = if self.matches(&Token::Keyword(Keyword::As)) {
                    Some(self.expect_identifier()?)
                } else if let Token::Identifier(s) = self.peek() {
                    let id = s.clone();
                    self.advance();
                    Some(id)
                } else {
                    None
                };
                projections.push(SelectItem::Expr { expr, alias });
            }

            if !self.matches(&Token::Comma) {
                break;
            }
        }
        Ok(projections)
    }

    /// Parses FROM table clauses with optional aliases.
    fn parse_table_ref(&mut self) -> Result<TableRef> {
        let name = self.expect_identifier()?;
        let alias = if self.matches(&Token::Keyword(Keyword::As)) {
            Some(self.expect_identifier()?)
        } else if let Token::Identifier(s) = self.peek() {
            let id = s.clone();
            self.advance();
            Some(id)
        } else {
            None
        };
        Ok(TableRef { name, alias })
    }

    /// Parses sorting clauses with ASC/DESC options.
    fn parse_order_by(&mut self) -> Result<Vec<OrderByExpr>> {
        let mut order_by = Vec::new();
        loop {
            let expr = self.parse_expr(0)?;
            let mut asc = true;
            if self.matches(&Token::Keyword(Keyword::Asc)) {
                asc = true;
            } else if self.matches(&Token::Keyword(Keyword::Desc)) {
                asc = false;
            }
            order_by.push(OrderByExpr { expr, asc });

            if !self.matches(&Token::Comma) {
                break;
            }
        }
        Ok(order_by)
    }

    /// Pratt Precedence climbing expression parser core.
    /// Parses subexpressions until hitting an operator with binding power below `min_bp`.
    fn parse_expr(&mut self, min_bp: u8) -> Result<Expr> {
        let mut left = self.parse_prefix()?;

        loop {
            let next_token = self.peek();

            // Handle postfix operators (like IS NULL)
            if let Some(postfix_bp) = self.postfix_binding_power(next_token) {
                if postfix_bp < min_bp {
                    break;
                }
                let op = self.advance().value.clone();
                left = self.parse_postfix(left, &op)?;
                continue;
            }

            // Handle infix operators
            if let Some((left_bp, right_bp)) = Self::infix_binding_power(next_token) {
                if left_bp < min_bp {
                    break;
                }
                let op_spanned = self.advance();
                let op = match &op_spanned.value {
                    Token::Plus => BinaryOp::Add,
                    Token::Minus => BinaryOp::Sub,
                    Token::Star => BinaryOp::Mul,
                    Token::Slash => BinaryOp::Div,
                    Token::Percent => BinaryOp::Mod,
                    Token::Eq => BinaryOp::Eq,
                    Token::NotEq => BinaryOp::NotEq,
                    Token::Lt => BinaryOp::Lt,
                    Token::LtEq => BinaryOp::LtEq,
                    Token::Gt => BinaryOp::Gt,
                    Token::GtEq => BinaryOp::GtEq,
                    Token::Keyword(Keyword::And) => BinaryOp::And,
                    Token::Keyword(Keyword::Or) => BinaryOp::Or,
                    _ => {
                        return Err(BasaltError::Syntax {
                            span: op_spanned.span,
                            message: format!("unexpected binary operator {}", op_spanned.value),
                        });
                    }
                };
                let right = self.parse_expr(right_bp)?;
                left = Expr::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                };
                continue;
            }

            break;
        }

        Ok(left)
    }

    /// Parses prefix elements: literal constants, identifiers, unary operators,
    /// parenthesized groupings, and CAST subexpressions.
    fn parse_prefix(&mut self) -> Result<Expr> {
        let token = self.advance();
        match &token.value {
            Token::Identifier(s) => Ok(Expr::Identifier(s.clone())),
            Token::Number(s) => {
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    let val = s.parse::<f64>().map_err(|_| BasaltError::Syntax {
                        span: token.span,
                        message: format!("invalid float literal '{s}'"),
                    })?;
                    Ok(Expr::Literal(Literal::Float(val)))
                } else {
                    let val = s.parse::<i64>().map_err(|_| BasaltError::Syntax {
                        span: token.span,
                        message: format!("invalid integer literal '{s}'"),
                    })?;
                    Ok(Expr::Literal(Literal::Integer(val)))
                }
            }
            Token::String(s) => Ok(Expr::Literal(Literal::String(s.clone()))),
            Token::Keyword(Keyword::True) => Ok(Expr::Literal(Literal::Boolean(true))),
            Token::Keyword(Keyword::False) => Ok(Expr::Literal(Literal::Boolean(false))),
            Token::Keyword(Keyword::Null) => Ok(Expr::Literal(Literal::Null)),

            Token::Minus => {
                let expr = self.parse_expr(13)?;
                Ok(Expr::Unary {
                    op: UnaryOp::Neg,
                    expr: Box::new(expr),
                })
            }
            Token::Keyword(Keyword::Not) => {
                let expr = self.parse_expr(5)?;
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(expr),
                })
            }
            Token::Plus => {
                // Unary plus is a no-op: just evaluate inside
                self.parse_expr(13)
            }
            Token::LParen => {
                let expr = self.parse_expr(0)?;
                self.expect(Token::RParen)?;
                Ok(Expr::Nested(Box::new(expr)))
            }
            Token::Keyword(Keyword::Cast) => {
                self.expect(Token::LParen)?;
                let expr = self.parse_expr(0)?;
                self.expect(Token::Keyword(Keyword::As))?;
                let to = self.parse_data_type()?;
                self.expect(Token::RParen)?;
                Ok(Expr::Cast {
                    expr: Box::new(expr),
                    to,
                })
            }
            other => Err(BasaltError::Syntax {
                span: token.span,
                message: format!("expected expression, found {other}"),
            }),
        }
    }

    /// Parses postfix operator chains (like IS NULL and IS NOT NULL).
    fn parse_postfix(&mut self, left: Expr, op: &Token) -> Result<Expr> {
        match op {
            Token::Keyword(Keyword::Is) => {
                let negated = self.matches(&Token::Keyword(Keyword::Not));
                self.expect(Token::Keyword(Keyword::Null))?;
                Ok(Expr::IsNull {
                    expr: Box::new(left),
                    negated,
                })
            }
            other => Err(BasaltError::Syntax {
                span: self.current_span(),
                message: format!("unexpected postfix operator {other}"),
            }),
        }
    }

    fn postfix_binding_power(&self, token: &Token) -> Option<u8> {
        match token {
            Token::Keyword(Keyword::Is) => Some(6),
            _ => None,
        }
    }

    fn infix_binding_power(token: &Token) -> Option<(u8, u8)> {
        match token {
            Token::Keyword(Keyword::Or) => Some((1, 2)),
            Token::Keyword(Keyword::And) => Some((3, 4)),
            Token::Eq | Token::NotEq | Token::Lt | Token::LtEq | Token::Gt | Token::GtEq => {
                Some((7, 8))
            }
            Token::Plus | Token::Minus => Some((9, 10)),
            Token::Star | Token::Slash | Token::Percent => Some((11, 12)),
            _ => None,
        }
    }

    /// Parses concrete type names (Int64, Float64, Utf8, Boolean).
    fn parse_data_type(&mut self) -> Result<DataType> {
        let token = self.advance();
        match &token.value {
            Token::Identifier(s) => match s.to_ascii_lowercase().as_str() {
                "int64" | "int" => Ok(DataType::Int64),
                "float64" | "float" | "double" => Ok(DataType::Float64),
                "utf8" | "string" | "varchar" | "text" => Ok(DataType::Utf8),
                "boolean" | "bool" => Ok(DataType::Boolean),
                _ => Err(BasaltError::Syntax {
                    span: token.span,
                    message: format!("unknown data type '{s}'"),
                }),
            },
            Token::Keyword(kw) => match kw {
                Keyword::True | Keyword::False | Keyword::Null => Err(BasaltError::Syntax {
                    span: token.span,
                    message: format!("expected data type, found keyword {kw}"),
                }),
                _ => match kw.name().to_ascii_lowercase().as_str() {
                    "int64" => Ok(DataType::Int64),
                    "float64" => Ok(DataType::Float64),
                    "utf8" => Ok(DataType::Utf8),
                    "boolean" => Ok(DataType::Boolean),
                    other => Err(BasaltError::Syntax {
                        span: token.span,
                        message: format!("unknown data type keyword '{other}'"),
                    }),
                },
            },
            other => Err(BasaltError::Syntax {
                span: token.span,
                message: format!("expected data type name, found {other}"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::lexer::Lexer;

    fn parse_expr_str(input: &str) -> Expr {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        parser.parse_expr(0).unwrap()
    }

    #[test]
    fn test_parse_precedence_arithmetic() {
        let expr = parse_expr_str("a + b * c");
        assert_eq!(
            expr,
            Expr::Binary {
                left: Box::new(Expr::Identifier("a".to_string())),
                op: BinaryOp::Add,
                right: Box::new(Expr::Binary {
                    left: Box::new(Expr::Identifier("b".to_string())),
                    op: BinaryOp::Mul,
                    right: Box::new(Expr::Identifier("c".to_string())),
                })
            }
        );
    }

    #[test]
    fn test_parse_precedence_associativity() {
        let expr = parse_expr_str("a - b - c");
        assert_eq!(
            expr,
            Expr::Binary {
                left: Box::new(Expr::Binary {
                    left: Box::new(Expr::Identifier("a".to_string())),
                    op: BinaryOp::Sub,
                    right: Box::new(Expr::Identifier("b".to_string())),
                }),
                op: BinaryOp::Sub,
                right: Box::new(Expr::Identifier("c".to_string())),
            }
        );
    }

    #[test]
    fn test_parse_precedence_logical() {
        let expr = parse_expr_str("NOT a AND b");
        assert_eq!(
            expr,
            Expr::Binary {
                left: Box::new(Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(Expr::Identifier("a".to_string()))
                }),
                op: BinaryOp::And,
                right: Box::new(Expr::Identifier("b".to_string()))
            }
        );
    }

    #[test]
    fn test_parse_postfix_is_null() {
        let expr = parse_expr_str("a IS NULL");
        assert_eq!(
            expr,
            Expr::IsNull {
                expr: Box::new(Expr::Identifier("a".to_string())),
                negated: false,
            }
        );

        let expr = parse_expr_str("a IS NOT NULL");
        assert_eq!(
            expr,
            Expr::IsNull {
                expr: Box::new(Expr::Identifier("a".to_string())),
                negated: true,
            }
        );

        let expr = parse_expr_str("a = b IS NULL");
        assert_eq!(
            expr,
            Expr::IsNull {
                expr: Box::new(Expr::Binary {
                    left: Box::new(Expr::Identifier("a".to_string())),
                    op: BinaryOp::Eq,
                    right: Box::new(Expr::Identifier("b".to_string())),
                }),
                negated: false,
            }
        );

        let expr = parse_expr_str("a + b IS NULL");
        assert_eq!(
            expr,
            Expr::IsNull {
                expr: Box::new(Expr::Binary {
                    left: Box::new(Expr::Identifier("a".to_string())),
                    op: BinaryOp::Add,
                    right: Box::new(Expr::Identifier("b".to_string())),
                }),
                negated: false,
            }
        );

        let expr = parse_expr_str("-a IS NULL");
        assert_eq!(
            expr,
            Expr::IsNull {
                expr: Box::new(Expr::Unary {
                    op: UnaryOp::Neg,
                    expr: Box::new(Expr::Identifier("a".to_string())),
                }),
                negated: false,
            }
        );
    }

    #[test]
    fn test_parse_cast() {
        let expr = parse_expr_str("CAST(a AS Float64)");
        assert_eq!(
            expr,
            Expr::Cast {
                expr: Box::new(Expr::Identifier("a".to_string())),
                to: DataType::Float64,
            }
        );
    }

    #[test]
    fn test_parse_select_statement() {
        let mut lexer =
            Lexer::new("SELECT a, b AS x FROM tbl WHERE c = 10 ORDER BY d DESC LIMIT 5;");
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let stmt = parser.parse_statement().unwrap();
        assert_eq!(
            stmt,
            Statement::Select(SelectStatement {
                projections: vec![
                    SelectItem::Expr {
                        expr: Expr::Identifier("a".to_string()),
                        alias: None,
                    },
                    SelectItem::Expr {
                        expr: Expr::Identifier("b".to_string()),
                        alias: Some("x".to_string()),
                    }
                ],
                from: TableRef {
                    name: "tbl".to_string(),
                    alias: None,
                },
                selection: Some(Expr::Binary {
                    left: Box::new(Expr::Identifier("c".to_string())),
                    op: BinaryOp::Eq,
                    right: Box::new(Expr::Literal(Literal::Integer(10))),
                }),
                order_by: vec![OrderByExpr {
                    expr: Expr::Identifier("d".to_string()),
                    asc: false,
                }],
                limit: Some(5),
            })
        );
    }

    #[test]
    fn test_parser_syntax_errors() {
        let mut lexer = Lexer::new("SELECT a FROM tbl WHERE");
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        assert!(parser.parse_statement().is_err());

        let mut lexer = Lexer::new("SELECT a tbl");
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        assert!(parser.parse_statement().is_err());
    }

    #[test]
    fn test_parse_unmatched_paren_errors() {
        let mut lexer = Lexer::new("SELECT (a FROM tbl");
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        assert!(parser.parse_statement().is_err());
    }

    #[test]
    fn test_parse_trailing_comma_in_projections_errors() {
        let mut lexer = Lexer::new("SELECT a, FROM tbl");
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        assert!(parser.parse_statement().is_err());
    }

    #[test]
    fn test_parse_empty_input_errors() {
        let mut lexer = Lexer::new("");
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        assert!(parser.parse_statement().is_err());
    }

    #[test]
    fn test_parse_nested_parens_preserve_precedence() {
        let expr = parse_expr_str("(a + b) * c");
        assert_eq!(
            expr,
            Expr::Binary {
                left: Box::new(Expr::Nested(Box::new(Expr::Binary {
                    left: Box::new(Expr::Identifier("a".to_string())),
                    op: BinaryOp::Add,
                    right: Box::new(Expr::Identifier("b".to_string())),
                }))),
                op: BinaryOp::Mul,
                right: Box::new(Expr::Identifier("c".to_string())),
            }
        );
    }

    #[test]
    fn test_parse_unary_minus_binds_tighter_than_binary_plus() {
        let expr = parse_expr_str("-a + b");
        assert_eq!(
            expr,
            Expr::Binary {
                left: Box::new(Expr::Unary {
                    op: UnaryOp::Neg,
                    expr: Box::new(Expr::Identifier("a".to_string())),
                }),
                op: BinaryOp::Add,
                right: Box::new(Expr::Identifier("b".to_string())),
            }
        );
    }

    #[test]
    fn test_parse_string_literal_with_escaped_quote() {
        let expr = parse_expr_str("'it''s'");
        assert_eq!(expr, Expr::Literal(Literal::String("it's".to_string())));
    }
}
