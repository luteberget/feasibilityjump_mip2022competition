pub mod cli;

use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;

pub struct MPSInstance {
    pub name: String,
    pub variables: Vec<Variable>,
    pub constraints: Vec<Constraint>,
}

impl MPSInstance {
    pub fn objective(&self) -> Option<(Option<Number>, &Vec<Cell>)> {
        let objective_row = self
            .constraints
            .iter()
            .find(|c| matches!(c.rowtype, RowType::None));
        objective_row.map(|c| (c.rhs, &c.cells))
    }
}

impl MPSInstance {
    pub fn calculate_objective(&self, solution: &[f64]) -> f64 {
        match self.objective() {
            Some((constant, obj)) => {
                let constant_term = constant.map(|n| n.as_f64()).unwrap_or(0.);
                -constant_term
                    + obj
                        .iter()
                        .map(|c| c.coeff.as_f64() * solution[c.var])
                        .sum::<f64>()
            }
            None => 0.,
        }
    }

    pub fn save_solution(
        &self,
        output_file: &str,
        solution: &[f64],
        objective: f64,
    ) -> std::io::Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(output_file)?;

        // Write objective value
        writeln!(file, "# Objective value: {}", objective)?;
        writeln!(file, "# Feasibility Jump Solution")?;
        writeln!(file)?;

        // Write variable assignments
        for (var_idx, value) in solution.iter().enumerate() {
            if var_idx < self.variables.len() {
                writeln!(file, "{} {}", self.variables[var_idx].name, value)?;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RowType {
    None,
    Equal,
    Lte,
    Gte,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Number {
    Int(i64),
    Float(f64),
}

impl Number {
    pub fn as_f64(self) -> f64 {
        match self {
            Number::Int(x) => x as f64,
            Number::Float(x) => x,
        }
    }
}

#[derive(Debug)]
pub struct Cell {
    pub var: usize,
    pub coeff: Number,
}

#[derive(Debug)]
pub struct Variable {
    pub name: String,
    pub var_type: VarType,
    pub lb: Option<Number>,
    pub ub: Option<Number>,
}

#[derive(Copy, Clone, Debug)]
pub enum VarType {
    Integer,
    Continuous,
}

#[derive(Debug)]

pub struct Constraint {
    pub rowtype: RowType,
    pub cells: Vec<Cell>,
    pub rhs: Option<Number>,
}

pub fn print_stats(problem: &MPSInstance) {
    println!("instance {}", problem.name);
    println!("  rows {}", problem.constraints.len());
    let mut row_sizes: BTreeMap<usize, usize> = BTreeMap::new();
    for row in problem.constraints.iter() {
        *row_sizes.entry(row.cells.len()).or_default() += 1;
    }

    let mut ints = 0;
    let mut bins = 0;
    let mut cont = 0;
    for var in problem.variables.iter() {
        match var.var_type {
            VarType::Integer => {
                if var.lb == Some(Number::Int(0)) && var.ub == Some(Number::Int(1)) {
                    bins += 1;
                } else {
                    ints += 1;
                }
            }
            VarType::Continuous => {
                cont += 1;
            }
        }
    }

    // println!("rhs {}/{}", problem.rhs.iter().filter(|r| r.is_some()).count(), problem.rhs.len());

    println!("    cont {}  bin {}  int {}", cont, bins, ints);
    println!("    rowlens {:?}", row_sizes);
    // println!("  rhs {}", problem.rhs.len());
    println!("  vars {}", problem.variables.len());
}

#[derive(Debug)]
pub enum ParseError {
    UnexpectedEnd,
    Unexpected(String),
    InvalidRowType(String),
    InvalidMarker(String),
    InvalidBoundsType(String),
    DuplicateRowName(String),
    DuplicateVarName(String),
    UnexpectedField(String),
    UninitializedRow(String),
    UninitializedVar(String),
    ParseNumberFailed(String),
    ExpectedLine,
    ExpectedField,
    ExpectedKeyword(String),
    NonUniqueRhsName(String),
    NonUniqueBoundsSets,
}

pub fn parse(input_text: impl BufRead) -> Result<MPSInstance, ParseError> {
    let mut lines = input_text
        .lines()
        .map(Result::unwrap)
        .filter(|l| l.split_ascii_whitespace().next().is_some() && !l.starts_with('*'))
        .peekable();

    macro_rules! expect_line {
        () => {
            lines.next().ok_or(ParseError::ExpectedLine)?
        };
    }

    macro_rules! expect_field {
        ($a:expr) => {
            $a.next().ok_or(ParseError::ExpectedField)?
        };
    }

    macro_rules! expect_keyword {
        ($a:expr,$b:expr) => {
            if expect_field!($a) != $b {
                return Err(ParseError::ExpectedKeyword($b.to_string()));
            }
        };
    }

    let name_line_str = expect_line!();
    let mut name_line = name_line_str.split_ascii_whitespace();
    expect_keyword!(name_line, "NAME");
    let name = name_line.next().unwrap_or("Unnamed problem");

    //
    // ROWS (TYPES AND NAMES)
    //

    let rows_line = expect_line!();
    expect_keyword!(rows_line.split_ascii_whitespace(), "ROWS");

    let mut rows = Vec::new();
    let mut row_names = HashMap::new();

    while lines.peek().map(|n| n.starts_with(' ')).unwrap_or(false) {
        let row_str = expect_line!();
        let mut row = row_str.split_ascii_whitespace();
        let row_idx = rows.len();
        rows.push(Constraint {
            rhs: None,
            rowtype: (match expect_field!(row) {
                "N" => RowType::None,
                "E" => RowType::Equal,
                "G" => RowType::Gte,
                "L" => RowType::Lte,
                x => {
                    return Err(ParseError::InvalidRowType(x.to_string()));
                }
            }),
            cells: Vec::new(),
        });
        let row_name = expect_field!(row);
        if row_names.insert(row_name.to_string(), row_idx).is_some() {
            return Err(ParseError::DuplicateRowName(row_name.to_string()));
        }
    }

    //
    // COLUMNS
    //

    let columns_keyword_line = expect_line!();
    expect_keyword!(columns_keyword_line.split_ascii_whitespace(), "COLUMNS");
    let mut var_type = VarType::Continuous;

    let mut vars = Vec::new();
    let mut var_names = HashMap::new();

    while lines.peek().map(|n| n.starts_with(' ')).unwrap_or(false) {
        let col_line = expect_line!();
        let mut col = col_line.split_ascii_whitespace().peekable();
        let var_name = expect_field!(col);

        if col.peek() == Some(&"'MARKER'") {
            col.next();
            match expect_field!(col) {
                "'INTORG'" => var_type = VarType::Integer,
                "'INTEND'" => var_type = VarType::Continuous,
                x => {
                    return Err(ParseError::InvalidMarker(x.to_string()));
                }
            }
        } else {
            // Create variable
            let var_idx = *var_names.entry(var_name.to_string()).or_insert_with(|| {
                let idx = vars.len();
                vars.push(Variable {
                    name: var_name.to_string(),
                    var_type,
                    lb: Some(Number::Int(0)),
                    ub: None,
                });
                idx
            });

            // Create cells
            while col.peek().is_some() {
                let row = expect_field!(col);
                let coeff = expect_field!(col);

                let row_idx = *row_names
                    .get(row)
                    .ok_or_else(|| ParseError::UninitializedRow(row.to_string()))?;

                let num = parse_number(coeff)?;

                rows[row_idx].cells.push(Cell {
                    var: var_idx,
                    coeff: num,
                });
            }
        }
    }

    let rhs_keyword_line = expect_line!();
    expect_keyword!(rhs_keyword_line.split_ascii_whitespace(), "RHS");
    let mut rhs_name: Option<String> = None;

    while lines.peek().map(|n| n.starts_with(' ')).unwrap_or(false) {
        let rhs_line_str = expect_line!();
        let mut rhs_line = rhs_line_str.split_ascii_whitespace().peekable();
        let this_rhs_name = Some(expect_field!(rhs_line));

        if rhs_name.is_none() {
            rhs_name = this_rhs_name.map(str::to_string);
        } else if this_rhs_name != rhs_name.as_deref() {
            return Err(ParseError::NonUniqueRhsName(
                this_rhs_name.unwrap().to_string(),
            ));
        }

        while rhs_line.peek().is_some() {
            let row = expect_field!(rhs_line);
            let coeff = expect_field!(rhs_line);

            let row_idx = *row_names
                .get(row)
                .ok_or_else(|| ParseError::UninitializedRow(row.to_string()))?;

            let num = parse_number(coeff)?;

            rows[row_idx].rhs = Some(num);
        }
    }

    let mut bound_names = HashSet::new();

    if lines
        .peek()
        .map(|l| l.starts_with("BOUNDS"))
        .unwrap_or(false)
    {
        let bounds_keyword_line_str = expect_line!();
        expect_keyword!(bounds_keyword_line_str.split_ascii_whitespace(), "BOUNDS");
        while lines.peek().map(|n| n.starts_with(' ')).unwrap_or(false) {
            let bound_line_str = expect_line!();
            let mut bound = bound_line_str.split_ascii_whitespace().peekable();

            let bound_type_str = expect_field!(bound);
            let bound_name = expect_field!(bound);
            bound_names.insert(bound_name.to_string());

            let var = expect_field!(bound);
            let var_idx = *var_names
                .get(var)
                .ok_or_else(|| ParseError::UninitializedVar(var.to_string()))?;

            match bound_type_str {
                "FR" => {
                    vars[var_idx].lb = None;
                    vars[var_idx].ub = None;
                }
                "MI" => {}
                "PL" => {}
                "BV" => {
                    vars[var_idx].lb = Some(Number::Int(0));
                    vars[var_idx].ub = Some(Number::Int(1));
                    vars[var_idx].var_type = VarType::Integer;
                }
                "SC" => panic!("Unsupported: Semi-continuous"),
                "LO" | "UP" | "FX" | "LI" | "UI" => {
                    let coeff = expect_field!(bound);
                    let num = parse_number(coeff)?;

                    if bound_type_str == "LO" || bound_type_str == "FX" || bound_type_str == "LI" {
                        vars[var_idx].lb = Some(num);
                    }

                    if bound_type_str == "UP" || bound_type_str == "FX" || bound_type_str == "UI" {
                        vars[var_idx].ub = Some(num);
                    }

                    if bound_type_str == "LI" || bound_type_str == "UI" {
                        vars[var_idx].var_type = VarType::Integer;
                    }
                }
                bound => {
                    return Err(ParseError::InvalidBoundsType(bound.to_string()));
                }
            };

            if let Some(x) = bound.next() {
                return Err(ParseError::UnexpectedField(x.to_string()));
            }
        }
    }

    if bound_names.len() > 1 {
        return Err(ParseError::NonUniqueBoundsSets);
    }

    let endata_keyword_line = expect_line!();
    expect_keyword!(endata_keyword_line.split_ascii_whitespace(), "ENDATA");

    Ok(MPSInstance {
        name: name.to_string(),
        constraints: rows,
        variables: vars,
    })
}

fn parse_number(coeff: &str) -> Result<Number, ParseError> {
    if let Ok(x) = coeff.parse::<i64>() {
        return Ok(Number::Int(x));
    }

    let float = coeff
        .parse::<f64>()
        .map_err(|_| ParseError::ParseNumberFailed(coeff.to_string()))?;

    if (float.round() - float).abs() < 1e-8 {
        return Ok(Number::Int(float.round() as i64));
    }

    Ok(Number::Float(float))
}

pub const DEFAULT_EQ_TOLERANCE: f64 = 1e-9;
pub const DEFAULT_INT_TOLERANCE: f64 = 1e-6;

pub fn check_values(
    mps: &MPSInstance,
    var_values: &[f64],
    int_tolerance: f64,
    eq_tolerance: f64,
) -> Result<(), &'static str> {
    //println!("CHECKING {} {}", int_tolerance, eq_tolerance);
    for (var_idx, var) in mps.variables.iter().enumerate() {
        if let VarType::Integer = var.var_type {
            let value = var_values[var_idx];
            let is_int = (value.round() - value).abs() < int_tolerance;
            if !is_int {
                return Err("Integrality tolerance failed.");
            }
        }
    }
    for row in mps.constraints.iter() {
        let lhs = row
            .cells
            .iter()
            .map(|c| c.coeff.as_f64() * var_values[c.var])
            .sum::<f64>();

        let rhs = row.rhs.map(Number::as_f64).unwrap_or(0.);

        match row.rowtype {
            RowType::None => {}
            RowType::Equal => {
                if (lhs - rhs).abs() > eq_tolerance {
                    println!("Row failed {} {} {:?}", lhs, rhs, row);
                    println!("...");
                    return Err("Equality tolerance failed.");
                }
            }
            RowType::Lte => {
                if lhs > rhs {
                    return Err("Less-than constraint failed.");
                }
            }
            RowType::Gte => {
                if lhs < rhs {
                    return Err("Greater-than constraint failed.");
                }
            }
        }
        //println!(" row ok");
    }
    Ok(())
}

pub fn parse_solution(solution: &str) -> Result<Vec<(&str, f32)>, &'static str> {
    solution
        .lines()
        .map(|l| {
            let mut fields = l.split_ascii_whitespace();
            let var_name = fields.next().ok_or("Expected variable name")?;
            let value_str = fields.next().ok_or("Expected value")?;
            let value = value_str
                .parse::<f32>()
                .map_err(|_| "Could not parse number")?;
            Ok((var_name, value))
        })
        .collect()
}
