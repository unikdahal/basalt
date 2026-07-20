//! Lexical analyzer (tokenizer) for SQL.
//!
//! Converts a source query string into a sequence of tokens (`Spanned<Token>`).
//! Uses a maximal-munch approach where applicable (e.g., consuming characters
//! as long as they form a valid identifier or number). It also handles special
//! case combinations like `ORDER BY` and character escaping in single-quoted strings.

use crate::error::{BasaltError, Result};
use crate::sql::token::{Keyword, Token};
use crate::sql::span::{Span, Spanned};

/// Lexer state tracking indices and slice references.
pub struct Lexer<'a> {
    input: &'a str,
    chars: Vec<(usize, char)>,
    position: usize,
}

impl<'a> Lexer<'a> {
    /// Creates a new SQL Lexer over the query text.
    pub fn new(input: &'a str) -> Self {
        let chars = input.char_indices().collect();
        Self { input, chars, position: 0 }
    }

    /// Scans the entire source string and produces a list of spanned tokens.
    /// Appends a final EOF token at the end of the stream.
    pub fn tokenize(&mut self) -> Result<Vec<Spanned<Token>>> {
        let mut tokens = Vec::new();

        while !self.is_eof() {
            self.skip_whitespace();
            if self.is_eof() {
                break;
            }

            let start_pos = self.current_pos();
            if let Some(token) = self.next_token()? {
                let end_pos = self.current_pos();
                tokens.push(Spanned::new(token, Span::new(start_pos, end_pos)));
            }
        }

        let eof_pos = self.input.len();
        tokens.push(Spanned::new(Token::Eof, Span::new(eof_pos, eof_pos)));
        Ok(tokens)
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.position).map(|&(_, c)| c)
    }

    fn peek_next(&self) -> Option<char> {
        self.chars.get(self.position + 1).map(|&(_, c)| c)
    }

    fn advance(&mut self) -> Option<char> {
        if self.position < self.chars.len() {
            let (_, c) = self.chars[self.position];
            self.position += 1;
            Some(c)
        } else {
            None
        }
    }

    fn is_eof(&self) -> bool {
        self.position >= self.chars.len()
    }

    fn current_pos(&self) -> usize {
        self.chars.get(self.position).map(|&(idx, _)| idx).unwrap_or(self.input.len())
    }

    /// Skips whitespace and single-line SQL comments starting with `--`.
    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else if c == '-' && self.peek_next() == Some('-') {
                // Skip the comment line
                self.advance();
                self.advance();
                while let Some(c_comment) = self.peek() {
                    self.advance();
                    if c_comment == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Dispatches and resolves the next token from the character stream.
    fn next_token(&mut self) -> Result<Option<Token>> {
        let c = match self.peek() {
            Some(x) => x,
            None => return Ok(None),
        };

        // Identifiers and Keywords
        if c.is_ascii_alphabetic() || c == '_' {
            return Ok(Some(self.scan_identifier_or_keyword()?));
        }

        // Numeric Literals
        if c.is_ascii_digit() {
            return Ok(Some(self.scan_number()));
        }

        // Single-Quoted String Literals
        if c == '\'' {
            return Ok(Some(self.scan_string()?));
        }

        // Operators & Separators matching (ordering maximal munch <=, >=, <> before single chars)
        self.advance();
        match c {
            '+' => Ok(Some(Token::Plus)),
            '-' => Ok(Some(Token::Minus)),
            '*' => Ok(Some(Token::Star)),
            '/' => Ok(Some(Token::Slash)),
            '%' => Ok(Some(Token::Percent)),
            ',' => Ok(Some(Token::Comma)),
            ';' => Ok(Some(Token::Semicolon)),
            '.' => Ok(Some(Token::Dot)),
            '(' => Ok(Some(Token::LParen)),
            ')' => Ok(Some(Token::RParen)),
            '=' => Ok(Some(Token::Eq)),
            '!' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Some(Token::NotEq))
                } else {
                    Err(BasaltError::Syntax {
                        span: Span::new(self.current_pos() - 1, self.current_pos()),
                        message: "unexpected character '!'".to_string(),
                    })
                }
            }
            '<' => {
                if self.peek() == Some('>') {
                    self.advance();
                    Ok(Some(Token::NotEq))
                } else if self.peek() == Some('=') {
                    self.advance();
                    Ok(Some(Token::LtEq))
                } else {
                    Ok(Some(Token::Lt))
                }
            }
            '>' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Some(Token::GtEq))
                } else {
                    Ok(Some(Token::Gt))
                }
            }
            other => Err(BasaltError::Syntax {
                span: Span::new(self.current_pos() - 1, self.current_pos()),
                message: format!("unexpected character '{}'", other),
            }),
        }
    }

    /// Scans an identifier or keyword using the maximal-munch principle.
    /// Consumes alphanumeric characters and underscores as long as possible.
    fn scan_identifier_or_keyword(&mut self) -> Result<Token> {
        let mut name = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' {
                name.push(c);
                self.advance();
            } else {
                break;
            }
        }

        // Special case: check if it is "ORDER" followed by "BY".
        // Combines them into a single Token::Keyword(Keyword::OrderBy)
        // to simplify parsing logic.
        if name.to_ascii_lowercase() == "order" {
            let saved_pos = self.position;
            self.skip_whitespace();
            let mut next_word = String::new();
            while let Some(c) = self.peek() {
                if c.is_ascii_alphanumeric() || c == '_' {
                    next_word.push(c);
                    self.advance();
                } else {
                    break;
                }
            }

            if next_word.to_ascii_lowercase() == "by" {
                return Ok(Token::Keyword(Keyword::OrderBy));
            } else {
                // Rollback position if it was just a column named 'order'
                self.position = saved_pos;
            }
        }

        if let Some(kw) = Keyword::from_str(&name) {
            Ok(Token::Keyword(kw))
        } else {
            Ok(Token::Identifier(name))
        }
    }

    /// Scans a numeric value as raw text, matching integers, floats,
    /// and exponential notations (e.g. 1.2e-4).
    fn scan_number(&mut self) -> Token {
        let mut value = String::new();

        // Scan integer digits
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                value.push(c);
                self.advance();
            } else {
                break;
            }
        }

        // Scan decimal fractional digits
        if self.peek() == Some('.') && self.peek_next().map_or(false, |c| c.is_ascii_digit()) {
            value.push(self.advance().unwrap()); // push '.'
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    value.push(c);
                    self.advance();
                } else {
                    break;
                }
            }
        }

        // Scan exponent suffix
        if let Some(c) = self.peek() {
            if c == 'e' || c == 'E' {
                value.push(self.advance().unwrap());
                if let Some(sign) = self.peek() {
                    if sign == '+' || sign == '-' {
                        value.push(self.advance().unwrap());
                    }
                }
                while let Some(exp_c) = self.peek() {
                    if exp_c.is_ascii_digit() {
                        value.push(exp_c);
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }

        Token::Number(value)
    }

    /// Scans a string literal enclosed in single quotes.
    /// Supports escaping single quotes via double-quotes inside SQL strings (`''` -> `'`).
    fn scan_string(&mut self) -> Result<Token> {
        let start = self.current_pos();
        self.advance(); // consume opening quote

        let mut value = String::new();
        loop {
            match self.peek() {
                Some('\'') => {
                    self.advance();
                    if self.peek() == Some('\'') {
                        // Escaped quote character
                        value.push('\'');
                        self.advance();
                    } else {
                        // Correct closing quote
                        break;
                    }
                }
                Some(c) => {
                    value.push(c);
                    self.advance();
                }
                None => {
                    return Err(BasaltError::Syntax {
                        span: Span::new(start, self.input.len()),
                        message: "unterminated string literal".to_string(),
                    });
                }
            }
        }

        Ok(Token::String(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lex_basic() {
        let mut lexer = Lexer::new("SELECT a, 42, 3.14 FROM tbl WHERE a <= 10;");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens.len(), 14);
        assert_eq!(tokens[0].value, Token::Keyword(Keyword::Select));
        assert_eq!(tokens[1].value, Token::Identifier("a".to_string()));
        assert_eq!(tokens[2].value, Token::Comma);
        assert_eq!(tokens[3].value, Token::Number("42".to_string()));
        assert_eq!(tokens[4].value, Token::Comma);
        assert_eq!(tokens[5].value, Token::Number("3.14".to_string()));
        assert_eq!(tokens[6].value, Token::Keyword(Keyword::From));
        assert_eq!(tokens[7].value, Token::Identifier("tbl".to_string()));
        assert_eq!(tokens[8].value, Token::Keyword(Keyword::Where));
        assert_eq!(tokens[9].value, Token::Identifier("a".to_string()));
        assert_eq!(tokens[10].value, Token::LtEq);
        assert_eq!(tokens[11].value, Token::Number("10".to_string()));
        assert_eq!(tokens[12].value, Token::Semicolon);
    }

    #[test]
    fn test_lex_order_by() {
        let mut lexer = Lexer::new("SELECT * FROM tbl ORDER BY col DESC");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens.len(), 8);
        assert_eq!(tokens[4].value, Token::Keyword(Keyword::OrderBy));
    }

    #[test]
    fn test_lex_comment_and_string() {
        let mut lexer = Lexer::new("SELECT 'hello ''world''' -- comment line\nFROM tbl");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].value, Token::Keyword(Keyword::Select));
        assert_eq!(tokens[1].value, Token::String("hello 'world'".to_string()));
        assert_eq!(tokens[2].value, Token::Keyword(Keyword::From));
    }

    #[test]
    fn test_lex_floats() {
        let mut lexer = Lexer::new("123.456 1e-5 2.5E+3");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].value, Token::Number("123.456".to_string()));
        assert_eq!(tokens[1].value, Token::Number("1e-5".to_string()));
        assert_eq!(tokens[2].value, Token::Number("2.5E+3".to_string()));
    }

    #[test]
    fn test_lex_unterminated_string() {
        let mut lexer = Lexer::new("SELECT 'hello");
        let err = lexer.tokenize().unwrap_err();
        match err {
            BasaltError::Syntax { message, .. } => {
                assert!(message.contains("unterminated string literal"));
            }
            _ => panic!("Expected syntax error for unterminated string"),
        }
    }

    #[test]
    fn test_lex_invalid_char() {
        let mut lexer = Lexer::new("SELECT @var");
        let err = lexer.tokenize().unwrap_err();
        match err {
            BasaltError::Syntax { message, .. } => {
                assert!(message.contains("unexpected character '@'"));
            }
            _ => panic!("Expected syntax error for invalid character"),
        }
    }
}
