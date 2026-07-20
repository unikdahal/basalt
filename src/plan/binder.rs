//! `Binder` — SQL AST to bound query planning.
//!
//! Transforms the unbound AST into a `BoundQuery`. Resolves named identifiers
//! to direct index positions in the schema, checks parameter types, performs
//! implicit cast promotions where needed, and derives the final output schema.

use crate::error::{BasaltError, Result};
use crate::sql::ast;
use crate::expr::expr::{Expr, UnaryOp};
use crate::types::data_type::DataType;
use crate::types::schema::{Field, Schema};
use crate::types::value::Value;

/// An aligned projection expression with its calculated output header name.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundProjection {
    pub expr: Expr,
    pub output_name: String,
}

/// A sort criteria on a bound expression.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundOrderBy {
    pub expr: Expr,
    pub asc: bool,
}

/// A fully bound query representation containing resolved projection, filter,
/// ordering, and limit specifications, along with the derived schema.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundQuery {
    /// Source table name.
    pub source: String,
    /// Projection items mapped to bound expressions.
    pub projections: Vec<BoundProjection>,
    /// Bound boolean filter expression (WHERE clause).
    pub filter: Option<Expr>,
    /// Sort expressions.
    pub order_by: Vec<BoundOrderBy>,
    /// Row limit value.
    pub limit: Option<usize>,
    /// Pre-calculated output schema of the query result.
    pub output_schema: Schema,
}

/// Binder mapping names to catalog objects.
pub struct Binder<'a> {
    schema: &'a Schema,
}

impl<'a> Binder<'a> {
    /// Creates a new Binder over the input schema.
    pub fn new(schema: &'a Schema) -> Self {
        Self { schema }
    }

    /// Entry point: binds statement syntax nodes into resolved query models.
    pub fn bind_statement(&self, stmt: &ast::Statement) -> Result<BoundQuery> {
        match stmt {
            ast::Statement::Select(select) => self.bind_select(select),
        }
    }

    fn bind_select(&self, select: &ast::SelectStatement) -> Result<BoundQuery> {
        let source = select.from.name.clone();

        // 1. Bind projections (handling Wildcard * expansion to all fields)
        let mut projections = Vec::new();
        for (i, item) in select.projections.iter().enumerate() {
            match item {
                ast::SelectItem::Wildcard => {
                    for (index, field) in self.schema.fields().iter().enumerate() {
                        projections.push(BoundProjection {
                            expr: Expr::Column {
                                index,
                                data_type: field.data_type,
                                nullable: field.nullable,
                            },
                            output_name: field.name.clone(),
                        });
                    }
                }
                ast::SelectItem::Expr { expr, alias } => {
                    let bound_expr = self.bind_expr(expr)?;
                    let output_name = if let Some(a) = alias {
                        a.clone()
                    } else if let ast::Expr::Identifier(name) = expr {
                        name.clone()
                    } else {
                        format!("expr_{i}")
                    };
                    projections.push(BoundProjection {
                        expr: bound_expr,
                        output_name,
                    });
                }
            }
        }

        // 2. Bind filter clause (verifying the WHERE expression resolves to Boolean)
        let filter = if let Some(ref selection) = select.selection {
            let bound_filter = self.bind_expr(selection)?;
            let t = bound_filter.data_type()?;
            if t != DataType::Boolean {
                return Err(BasaltError::Type {
                    message: format!("WHERE clause must evaluate to Boolean, found {t}"),
                });
            }
            Some(bound_filter)
        } else {
            None
        };

        // 3. Bind sort keys
        let mut order_by = Vec::new();
        for ob in &select.order_by {
            let bound_expr = self.bind_expr(&ob.expr)?;
            order_by.push(BoundOrderBy {
                expr: bound_expr,
                asc: ob.asc,
            });
        }

        // 4. Bind row limits
        let limit = select.limit.map(|l| l as usize);

        // 5. Build query result schema
        let mut fields = Vec::with_capacity(projections.len());
        for p in &projections {
            let dt = p.expr.data_type()?;
            let nullable = p.expr.nullable();
            fields.push(Field::new(p.output_name.clone(), dt, nullable));
        }
        let output_schema = Schema::new(fields)?;

        Ok(BoundQuery {
            source,
            projections,
            filter,
            order_by,
            limit,
            output_schema,
        })
    }

    /// Recursive binder translating AST expressions to type-checked bound nodes.
    fn bind_expr(&self, expr: &ast::Expr) -> Result<Expr> {
        match expr {
            ast::Expr::Identifier(name) => {
                // Name resolution mapping to indices:
                // We resolve the identifier against column index offsets in the schema.
                // Resolving this once at bind time prevents string matches at runtime.
                let index = self.schema.index_of(name).ok_or_else(|| {
                    BasaltError::UnknownColumn { name: name.clone() }
                })?;
                let field = self.schema.field(index).unwrap();
                Ok(Expr::Column {
                    index,
                    data_type: field.data_type,
                    nullable: field.nullable,
                })
            }
            ast::Expr::Literal(lit) => {
                let val = match lit {
                    ast::Literal::Integer(x) => Value::Int64(*x),
                    ast::Literal::Float(x) => Value::Float64(*x),
                    ast::Literal::String(s) => Value::Utf8(s.clone()),
                    ast::Literal::Boolean(b) => Value::Boolean(*b),
                    ast::Literal::Null => Value::Null,
                };
                Ok(Expr::Literal(val))
            }
            ast::Expr::Binary { left, op, right } => {
                let mut bound_left = self.bind_expr(left)?;
                let mut bound_right = self.bind_expr(right)?;

                let lhs_type = bound_left.data_type()?;
                let rhs_type = bound_right.data_type()?;

                // Coercion insertion: Evaluates coercion promotion rules.
                // Wrap left or right operands in explicit Expr::Cast nodes when needed.
                let plan = crate::types::coercion::coerce_binary(*op, lhs_type, rhs_type)?;

                if let Some(target) = plan.lhs_cast {
                    bound_left = Expr::Cast {
                        expr: Box::new(bound_left),
                        to: target,
                    };
                }
                if let Some(target) = plan.rhs_cast {
                    bound_right = Expr::Cast {
                        expr: Box::new(bound_right),
                        to: target,
                    };
                }

                Ok(Expr::Binary {
                    left: Box::new(bound_left),
                    op: *op,
                    right: Box::new(bound_right),
                })
            }
            ast::Expr::Unary { op, expr } => {
                let bound_expr = self.bind_expr(expr)?;
                let t = bound_expr.data_type()?;
                match op {
                    UnaryOp::Neg => {
                        if !t.is_numeric() {
                            return Err(BasaltError::Type {
                                message: format!("cannot negate non-numeric type {t}"),
                            });
                        }
                    }
                    UnaryOp::Not => {
                        if t != DataType::Boolean {
                            return Err(BasaltError::Type {
                                message: format!("cannot logically invert non-boolean type {t}"),
                            });
                        }
                    }
                }
                Ok(Expr::Unary {
                    op: *op,
                    expr: Box::new(bound_expr),
                })
            }
            ast::Expr::Cast { expr, to } => {
                let bound_expr = self.bind_expr(expr)?;
                Ok(Expr::Cast {
                    expr: Box::new(bound_expr),
                    to: *to,
                })
            }
            ast::Expr::IsNull { expr, negated } => {
                let bound_expr = self.bind_expr(expr)?;
                if *negated {
                    Ok(Expr::IsNotNull(Box::new(bound_expr)))
                } else {
                    Ok(Expr::IsNull(Box::new(bound_expr)))
                }
            }
            ast::Expr::Nested(inner) => self.bind_expr(inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::expr::BinaryOp;
    use crate::sql::lexer::Lexer;
    use crate::sql::parser::Parser;

    fn test_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("score", DataType::Float64, true),
        ]).unwrap()
    }

    fn bind_sql(sql: &str, schema: &Schema) -> BoundQuery {
        let mut lexer = Lexer::new(sql);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let stmt = parser.parse_statement().unwrap();
        let binder = Binder::new(schema);
        binder.bind_statement(&stmt).unwrap()
    }

    #[test]
    fn test_bind_wildcard() {
        let schema = test_schema();
        let q = bind_sql("SELECT * FROM tbl", &schema);
        assert_eq!(q.projections.len(), 3);
        assert_eq!(q.projections[0].output_name, "id");
        assert_eq!(q.projections[1].output_name, "name");
        assert_eq!(q.projections[2].output_name, "score");
        assert_eq!(q.output_schema, schema);
    }

    #[test]
    fn test_bind_coercion() {
        let schema = test_schema();
        let q = bind_sql("SELECT id FROM tbl WHERE score > 80", &schema);
        let filter = q.filter.unwrap();
        assert!(matches!(
            filter,
            Expr::Binary {
                op: BinaryOp::Gt,
                right: box_right,
                ..
            } if matches!(*box_right, Expr::Cast { to: DataType::Float64, .. })
        ));
    }

    #[test]
    fn test_bind_type_error() {
        let schema = test_schema();
        let mut lexer = Lexer::new("SELECT id FROM tbl WHERE name > 10");
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let stmt = parser.parse_statement().unwrap();
        let binder = Binder::new(&schema);
        assert!(binder.bind_statement(&stmt).is_err());
    }

    #[test]
    fn test_bind_unknown_column_error() {
        let schema = test_schema();
        let mut lexer = Lexer::new("SELECT fake_col FROM tbl");
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let stmt = parser.parse_statement().unwrap();
        let binder = Binder::new(&schema);
        
        let err = binder.bind_statement(&stmt).unwrap_err();
        assert!(matches!(err, BasaltError::UnknownColumn { .. }));
    }

    #[test]
    fn test_bind_duplicate_column_bindings() {
        let schema = test_schema();
        let q = bind_sql("SELECT id, id AS id_alias FROM tbl", &schema);
        assert_eq!(q.projections.len(), 2);
        assert_eq!(q.projections[0].output_name, "id");
        assert_eq!(q.projections[1].output_name, "id_alias");
    }
}
