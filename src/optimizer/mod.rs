//! The cost-based optimizer. See design-docs/basalt-phase3-lld.md.
//!
//! Sits above `statistics`, below nothing (`statistics -> optimizer ->
//! {logical_plan, physical_plan}`). Bottom-up dynamic programming for join
//! ordering plus a rule engine for everything else — what Postgres and
//! DataFusion do, and the right scope for this project (see §7.4: a full
//! Cascades/Volcano top-down transformational optimizer is explicitly out
//! of scope).

pub mod cardinality;
pub mod cost;
pub mod join_order;
// LLD names this file `optimizer/optimizer.rs` for the pass manager
// specifically (mirroring `buffer/buffer.rs` from Phase 2); clippy reads it
// as a name clash with the parent module, but it's intentional.
#[allow(clippy::module_inception)]
pub mod optimizer;
pub mod rule;
pub mod rules;
pub mod tree_node;

pub use optimizer::default_optimizer;
pub use rule::{ApplyOrder, NoStatistics, Optimizer, OptimizerContext, OptimizerRule};
pub use tree_node::{Transformed, TreeNode, VisitRecursion};

#[cfg(test)]
mod integration_tests {
    //! End-to-end: `FROM orders o, customers c WHERE o.customer_id = c.id
    //! AND c.region = 'west' AND o.amount > 100`, the classic implicit-join
    //! shape (`EliminateCrossJoin`'s whole reason for existing), run through
    //! the full default rule pipeline, then physically planned and
    //! executed, checking the *results* are correct — not just that some
    //! rule fired.
    //!
    //! **A real, documented interaction gap found while writing this
    //! test**: `EliminateCrossJoin` promotes the equi-conjunct into
    //! `Join.on` and moves the other two (single-side) conjuncts into
    //! `Join.filter` in the same pass, since it consumes the outer `Filter`
    //! node entirely. `PredicatePushdown` only fires on a `Filter` node
    //! sitting directly over a `Join` — but there isn't one left by the
    //! time `PredicatePushdown` runs, so those single-side conjuncts never
    //! get pushed further down into their own base relation's scan. The
    //! plan is still **correct** (they're evaluated as part of the join's
    //! residual filter instead of before it), just not as deeply pushed as
    //! it could be. A follow-up rule teaching `EliminateCrossJoin` (or a
    //! new rule) to split leftover single-side conjuncts into a per-side
    //! pre-join `Filter` instead of `Join.filter` would close this gap.

    use std::sync::Arc;

    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::string::StringBuilder;
    use crate::array::types::Int64Type;
    use crate::batch::ColumnarBatch;
    use crate::expr::expr::Expr;
    use crate::logical_plan::builder::LogicalPlanBuilder;
    use crate::logical_plan::plan::{JoinType, LogicalPlan};
    use crate::optimizer::default_optimizer;
    use crate::optimizer::rule::NoStatistics;
    use crate::physical_plan::planner::PhysicalPlanner;
    use crate::physical_plan::scan::MemoryTableSource;
    use crate::types::coercion::BinaryOp;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema, SchemaRef};
    use crate::types::value::Value;

    fn orders_schema() -> SchemaRef {
        Arc::new(
            Schema::new(vec![
                Field::new("customer_id", DataType::Int64, false),
                Field::new("amount", DataType::Int64, false),
            ])
            .unwrap(),
        )
    }

    fn customers_schema() -> SchemaRef {
        Arc::new(
            Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("region", DataType::Utf8, false),
            ])
            .unwrap(),
        )
    }

    fn orders_source() -> Arc<MemoryTableSource> {
        // (customer_id, amount): (1,50) (1,200) (2,150) (2,80)
        let mut customer_id = PrimitiveBuilder::<Int64Type>::with_capacity(4);
        let mut amount = PrimitiveBuilder::<Int64Type>::with_capacity(4);
        for &(c, a) in &[(1i64, 50i64), (1, 200), (2, 150), (2, 80)] {
            customer_id.append_value(c);
            amount.append_value(a);
        }
        let batch = ColumnarBatch::try_new(
            orders_schema(),
            vec![Arc::new(customer_id.finish()), Arc::new(amount.finish())],
        )
        .unwrap();
        Arc::new(MemoryTableSource::new(orders_schema(), vec![batch]))
    }

    fn customers_source() -> Arc<MemoryTableSource> {
        // (id, region): (1, "west") (2, "east")
        let mut id = PrimitiveBuilder::<Int64Type>::with_capacity(2);
        let mut region = StringBuilder::with_capacity(2, 8);
        id.append_value(1);
        region.append_value("west").unwrap();
        id.append_value(2);
        region.append_value("east").unwrap();
        let batch = ColumnarBatch::try_new(
            customers_schema(),
            vec![Arc::new(id.finish()), Arc::new(region.finish())],
        )
        .unwrap();
        Arc::new(MemoryTableSource::new(customers_schema(), vec![batch]))
    }

    fn col(i: usize, dt: DataType) -> Expr {
        Expr::Column {
            index: i,
            data_type: dt,
            nullable: false,
        }
    }

    #[test]
    fn implicit_cross_join_query_optimizes_and_executes_correctly() {
        let orders = LogicalPlanBuilder::scan("orders", orders_source());
        let customers = LogicalPlanBuilder::scan("customers", customers_source()).build();

        let combined_schema = Arc::new(Schema::new_allow_duplicate_names(
            orders_schema()
                .fields()
                .iter()
                .cloned()
                .chain(customers_schema().fields().iter().cloned())
                .collect(),
        ));
        let cross_join = LogicalPlan::Join {
            left: orders.build(),
            right: customers,
            on: vec![],
            filter: None,
            join_type: JoinType::Inner,
            schema: combined_schema.clone(),
        };

        // o.customer_id = c.id AND c.region = 'west' AND o.amount > 100
        let predicate = Expr::Binary {
            left: Box::new(Expr::Binary {
                left: Box::new(Expr::Binary {
                    left: Box::new(col(0, DataType::Int64)),
                    op: BinaryOp::Eq,
                    right: Box::new(col(2, DataType::Int64)),
                }),
                op: BinaryOp::And,
                right: Box::new(Expr::Binary {
                    left: Box::new(col(3, DataType::Utf8)),
                    op: BinaryOp::Eq,
                    right: Box::new(Expr::Literal(Value::Utf8("west".to_string()))),
                }),
            }),
            op: BinaryOp::And,
            right: Box::new(Expr::Binary {
                left: Box::new(col(1, DataType::Int64)),
                op: BinaryOp::Gt,
                right: Box::new(Expr::Literal(Value::Int64(100))),
            }),
        };
        let plan = LogicalPlan::Filter {
            input: Arc::new(cross_join),
            predicate,
        };

        let optimized = default_optimizer().optimize(plan, &NoStatistics).unwrap();

        // EliminateCrossJoin must have promoted the equi-conjunct into a
        // real join condition — the whole point of the rule.
        let join_on_len = find_join(&optimized).map(|j| match j {
            LogicalPlan::Join { on, .. } => on.len(),
            _ => unreachable!(),
        });
        assert_eq!(
            join_on_len,
            Some(1),
            "expected the equi-conjunct promoted into Join.on: {optimized}"
        );

        // Execute and check the actual result: only customer 1 is in the
        // west region, and only their $200 order exceeds $100 — customer
        // 1's $50 order and customer 2's orders must not appear.
        let physical = PhysicalPlanner.create_physical_plan(&optimized).unwrap();
        let batches: Vec<_> = physical
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            total_rows, 1,
            "expected exactly one matching row (customer 1's $200 order): {optimized}"
        );
    }

    fn find_join(plan: &LogicalPlan) -> Option<&LogicalPlan> {
        if matches!(plan, LogicalPlan::Join { .. }) {
            return Some(plan);
        }
        plan.inputs().into_iter().find_map(find_join)
    }
}
