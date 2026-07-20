//! CLI / REPL entry point for Basalt.
//!
//! Provides a console-based REPL loop allowing users to input SQL queries.
//! Tables referenced in the query's FROM clause are dynamically loaded
//! from matching CSV files in the working directory and formatted as ASCII tables.

use std::io::{self, Write};
use std::path::Path;
use basalt::io::csv::{CsvReader, CsvReadOptions};
use basalt::exec::dataframe::execute;

fn main() {
    println!("============================================================");
    println!("Basalt SQL Engine CLI / REPL (Phase 1)");
    println!("------------------------------------------------------------");
    println!("Type your SQL queries at the prompt.");
    println!("The engine automatically loads '<table_name>.csv' from the");
    println!("current directory matching the FROM clause.");
    println!("Type 'exit' or 'quit' to exit.");
    println!("============================================================");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("basalt> ");
        if stdout.flush().is_err() {
            break;
        }

        let mut query = String::new();
        match stdin.read_line(&mut query) {
            Ok(0) => break, // EOF (Ctrl+D)
            Ok(_) => {}
            Err(e) => {
                println!("Error reading input: {e}");
                continue;
            }
        }

        let query = query.trim();
        if query.is_empty() {
            continue;
        }

        if query.eq_ignore_ascii_case("exit") || query.eq_ignore_ascii_case("quit") {
            println!("Goodbye!");
            break;
        }

        // Run query execution
        if let Err(e) = handle_query(query) {
            println!("Error: {e}");
        }
        println!();
    }
}

fn handle_query(sql: &str) -> basalt::Result<()> {
    // 1. Tokenize and parse first to extract the table name from FROM clause.
    let mut lexer = basalt::sql::lexer::Lexer::new(sql);
    let tokens = lexer.tokenize()?;
    let mut parser = basalt::sql::parser::Parser::new(tokens);
    
    let stmt = parser.parse_statement()?;
    let table_name = match &stmt {
        basalt::sql::ast::Statement::Select(select) => &select.from.name,
    };

    // 2. Resolve CSV file path
    let csv_filename = format!("{table_name}.csv");
    let csv_path = Path::new(&csv_filename);
    if !csv_path.exists() {
        return Err(basalt::error::BasaltError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "Table file '{csv_filename}' not found. Please create it in the current directory."
            ),
        )));
    }

    // 3. Load CSV data into RecordBatch
    println!("Loading {csv_filename}...");
    let batch = CsvReader::read_file(csv_path, &CsvReadOptions::default())?;
    println!("Loaded {} rows.", batch.num_rows());

    // 4. Bind and execute the query
    println!("Executing query...");
    let start_time = std::time::Instant::now();
    let result = execute(sql, batch)?;
    let elapsed = start_time.elapsed();

    // 5. Display the result
    // REPL table output formatting: The RecordBatch implements Display, internally calculating
    // the max width for each column to neatly align the data in a tabular layout 
    // suitable for terminal consumption.
    println!("{result}");
    println!("(Returned {} rows in {:?})", result.num_rows(), elapsed);

    Ok(())
}
