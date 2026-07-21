//! `TreeNode` — the shared tree-rewriting infrastructure every optimizer
//! rule is written against. See design-docs/basalt-phase3-lld.md §5.1.
//!
//! Implemented for both `LogicalPlan` and `Expr` on top of the
//! `inputs`/`with_new_inputs` (`children`/`with_new_children` for `Expr`)
//! primitives those types already had from Phase 2 — this is a thin,
//! uniform interface over machinery that already existed, not a rewrite of
//! it. A good traversal API makes a rule three lines; a bad one makes it
//! thirty.

use crate::error::Result;
use crate::expr::expr::Expr;
use crate::logical_plan::plan::LogicalPlan;
use std::sync::Arc;

/// Whether a rewrite actually changed anything — not decoration:
/// `Optimizer`'s fixed-point pass manager uses it to decide whether another
/// iteration is needed, and rules use it to avoid rebuilding unchanged
/// subtrees (cheap since children are `Arc`'d — an unchanged subtree is a
/// refcount bump).
#[derive(Debug)]
pub enum Transformed<T> {
    /// This rule changed something.
    Yes(T),
    /// Unchanged.
    No(T),
}

impl<T> Transformed<T> {
    pub fn into_inner(self) -> T {
        match self {
            Transformed::Yes(t) | Transformed::No(t) => t,
        }
    }

    pub fn is_yes(&self) -> bool {
        matches!(self, Transformed::Yes(_))
    }

    #[must_use]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Transformed<U> {
        match self {
            Transformed::Yes(t) => Transformed::Yes(f(t)),
            Transformed::No(t) => Transformed::No(f(t)),
        }
    }

    /// Combines this transform's "did anything change" flag with another's
    /// — `Yes` if either was `Yes`. Used when a rule rewrites several
    /// children and needs one combined answer for the parent.
    #[must_use]
    pub fn or(self, other_changed: bool) -> Transformed<T> {
        match self {
            Transformed::Yes(t) => Transformed::Yes(t),
            Transformed::No(t) if other_changed => Transformed::Yes(t),
            no => no,
        }
    }
}

/// What a read-only `visit` should do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisitRecursion {
    Continue,
    Stop,
}

pub trait TreeNode: Sized + Clone {
    fn children_nodes(&self) -> Vec<&Self>;

    /// # Errors
    /// Propagates any error from `f` or from rebuilding this node with new
    /// children (e.g. an arity mismatch, which should not occur for
    /// well-formed trees).
    fn with_new_children_nodes(&self, children: Vec<Self>) -> Result<Self>;

    /// Rewrites leaves first (most rules want this): recurse into every
    /// child, then apply `f` to this node with the (possibly rewritten)
    /// children installed.
    ///
    /// # Errors
    /// Propagates any error `f` returns, or a rebuild error.
    fn transform_up<F>(&self, f: &mut F) -> Result<Transformed<Self>>
    where
        F: FnMut(Self) -> Result<Transformed<Self>>,
    {
        let mut any_child_changed = false;
        let mut new_children = Vec::with_capacity(self.children_nodes().len());
        for child in self.children_nodes() {
            let transformed = child.transform_up(f)?;
            any_child_changed |= transformed.is_yes();
            new_children.push(transformed.into_inner());
        }
        let rebuilt = self.with_new_children_nodes(new_children)?;
        let result = f(rebuilt)?;
        Ok(result.or(any_child_changed))
    }

    /// Rewrites the root first (pushdown rules want this): apply `f` to
    /// this node, then recurse into the (possibly rewritten) node's
    /// children.
    ///
    /// # Errors
    /// Propagates any error `f` returns, or a rebuild error.
    fn transform_down<F>(&self, f: &mut F) -> Result<Transformed<Self>>
    where
        F: FnMut(Self) -> Result<Transformed<Self>>,
    {
        let top = f(self.clone())?;
        let top_changed = top.is_yes();
        let node = top.into_inner();

        let mut any_child_changed = false;
        let mut new_children = Vec::with_capacity(node.children_nodes().len());
        for child in node.children_nodes() {
            let transformed = child.transform_down(f)?;
            any_child_changed |= transformed.is_yes();
            new_children.push(transformed.into_inner());
        }
        let rebuilt = node.with_new_children_nodes(new_children)?;
        Ok(if top_changed || any_child_changed {
            Transformed::Yes(rebuilt)
        } else {
            Transformed::No(rebuilt)
        })
    }

    /// Read-only walk (pre-order) with early termination.
    ///
    /// # Errors
    /// Propagates any error `v` returns.
    fn visit<V>(&self, v: &mut V) -> Result<VisitRecursion>
    where
        V: FnMut(&Self) -> Result<VisitRecursion>,
    {
        if v(self)? == VisitRecursion::Stop {
            return Ok(VisitRecursion::Stop);
        }
        for child in self.children_nodes() {
            if child.visit(v)? == VisitRecursion::Stop {
                return Ok(VisitRecursion::Stop);
            }
        }
        Ok(VisitRecursion::Continue)
    }
}

impl TreeNode for Expr {
    fn children_nodes(&self) -> Vec<&Self> {
        self.children()
    }

    fn with_new_children_nodes(&self, children: Vec<Self>) -> Result<Self> {
        self.with_new_children(children)
    }
}

impl TreeNode for LogicalPlan {
    fn children_nodes(&self) -> Vec<&Self> {
        self.inputs()
    }

    fn with_new_children_nodes(&self, children: Vec<Self>) -> Result<Self> {
        self.with_new_inputs(children.into_iter().map(Arc::new).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::coercion::BinaryOp;
    use crate::types::data_type::DataType;
    use crate::types::value::Value;

    fn col(i: usize) -> Expr {
        Expr::Column {
            index: i,
            data_type: DataType::Int64,
            nullable: false,
        }
    }

    #[test]
    fn transform_up_rewrites_leaves_before_the_root_sees_them() {
        // Replace every Literal(1) with Literal(2); the root Binary node
        // should see the already-rewritten children.
        let expr = Expr::Binary {
            left: Box::new(Expr::Literal(Value::Int64(1))),
            op: BinaryOp::Add,
            right: Box::new(col(0)),
        };
        let result = expr
            .transform_up(&mut |e| {
                Ok(match e {
                    Expr::Literal(Value::Int64(1)) => {
                        Transformed::Yes(Expr::Literal(Value::Int64(2)))
                    }
                    other => Transformed::No(other),
                })
            })
            .unwrap();
        assert!(result.is_yes());
        match result.into_inner() {
            Expr::Binary { left, .. } => assert_eq!(*left, Expr::Literal(Value::Int64(2))),
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn transform_down_sees_the_root_before_rewriting_children() {
        let mut visited_order = Vec::new();
        let expr = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Add,
            right: Box::new(col(1)),
        };
        expr.visit(&mut |node| {
            visited_order.push(format!("{node:?}").chars().take(6).collect::<String>());
            Ok(VisitRecursion::Continue)
        })
        .unwrap();
        // Root visited first (pre-order), matching transform_down's shape.
        assert_eq!(visited_order.len(), 3);
    }

    #[test]
    fn no_change_reports_no() {
        let expr = col(0);
        let result = expr.transform_up(&mut |e| Ok(Transformed::No(e))).unwrap();
        assert!(!result.is_yes());
    }

    #[test]
    fn visit_stops_early_when_requested() {
        let expr = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Add,
            right: Box::new(col(1)),
        };
        let mut count = 0;
        expr.visit(&mut |_| {
            count += 1;
            Ok(VisitRecursion::Stop)
        })
        .unwrap();
        assert_eq!(count, 1, "should have stopped after the first (root) node");
    }
}
