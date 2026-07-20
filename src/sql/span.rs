//! Byte-offset source positions for tracking locations in SQL query strings.
//!
//! This module provides the `Span` and `Spanned<T>` types. Spans are used to map
//! AST nodes, expressions, and tokens back to their exact character offsets
//! in the user's input query. This is essential for rendering precise, helpful
//! syntax and semantic compiler errors.

use std::fmt;

/// A byte-offset range in the source SQL query string.
/// Represents a half-open interval `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Span {
    /// Starting byte index of this span.
    pub start: usize,
    /// Ending byte index of this span (exclusive).
    pub end: usize,
}

impl Span {
    /// Creates a new `Span` with the given start and end byte offsets.
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Creates an empty (zero-width) span at a specific position.
    /// This is useful for representing insertions or zero-width elements.
    pub const fn empty(pos: usize) -> Self {
        Self {
            start: pos,
            end: pos,
        }
    }

    /// Merges two spans to cover the range from the start of the earliest
    /// to the end of the latest. This is used when combining smaller AST nodes
    /// into larger ones (e.g. merging operands and operators).
    pub fn merge(self, other: Self) -> Self {
        Self {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// A value wrapper that attaches a `Span` for source location tracking.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Spanned<T> {
    /// The wrapped value.
    pub value: T,
    /// The source span covering the characters that produced this value.
    pub span: Span,
}

impl<T> Spanned<T> {
    /// Wraps a value with a span.
    pub const fn new(value: T, span: Span) -> Self {
        Self { value, span }
    }

    /// Maps the wrapped value to a new value of type `U` using the provided function,
    /// while keeping the original `Span` intact.
    pub fn map<U, F>(self, f: F) -> Spanned<U>
    where
        F: FnOnce(T) -> U,
    {
        Spanned {
            value: f(self.value),
            span: self.span,
        }
    }
}

impl<T> std::ops::Deref for Spanned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> std::ops::DerefMut for Spanned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_display() {
        let span = Span::new(10, 20);
        assert_eq!(span.to_string(), "10..20");
    }

    #[test]
    fn test_span_merge() {
        let s1 = Span::new(5, 10);
        let s2 = Span::new(12, 15);
        assert_eq!(s1.merge(s2), Span::new(5, 15));
        assert_eq!(s2.merge(s1), Span::new(5, 15));

        let s3 = Span::new(8, 11);
        assert_eq!(s1.merge(s3), Span::new(5, 11));
    }

    #[test]
    fn test_span_empty() {
        let empty_span = Span::empty(5);
        assert_eq!(empty_span.start, 5);
        assert_eq!(empty_span.end, 5);
        assert_eq!(empty_span.to_string(), "5..5");
    }

    #[test]
    fn test_spanned_deref() {
        let mut s = Spanned::new("hello".to_string(), Span::new(0, 5));
        assert_eq!(s.len(), 5);
        assert_eq!(*s, "hello");

        s.push_str(" world");
        assert_eq!(*s, "hello world");
    }

    #[test]
    fn test_spanned_map() {
        let s = Spanned::new(42, Span::new(0, 2));
        let mapped = s.map(|v| v.to_string());
        assert_eq!(mapped.value, "42");
        assert_eq!(mapped.span, Span::new(0, 2));
    }
}
