#![allow(unused)]

pub mod generate_svg;

use std::{
    fmt,
    ops::{Add, Div, Mul, Sub},
    slice::Iter,
};

#[derive(Clone)]
pub struct Value {
    pub current_evaluation: f64,
    pub gradient: Option<f64>,
    pub operation: ValueOperation,
}

#[derive(Clone)]
pub enum ValueOperation {
    Leaf,
    Addition(Box<Value>, Box<Value>),
    Subtraction(Box<Value>, Box<Value>),
    Multiplication(Box<Value>, Box<Value>),
    Division(Box<Value>, Box<Value>),
    Exponentiation { base: Box<Value>, exp: Box<Value> },
    Tanh(Box<Value>),
}

impl Value {
    pub fn leaf(current_evaluation: f64) -> Self {
        Value {
            current_evaluation,
            gradient: None,
            operation: ValueOperation::Leaf,
        }
    }

    pub fn children(&self) -> Vec<Value> {
        match &self.operation {
            ValueOperation::Leaf => vec![],
            ValueOperation::Addition(left, right)
            | ValueOperation::Subtraction(left, right)
            | ValueOperation::Multiplication(left, right)
            | ValueOperation::Division(left, right) => {
                vec![(**left).clone(), (**right).clone()]
            }
            ValueOperation::Exponentiation { base, exp } => {
                vec![(**base).clone(), (**exp).clone()]
            }
            ValueOperation::Tanh(value) => vec![(**value).clone()],
        }
    }

    pub fn evaluate_value(&self) -> f64 {
        self.current_evaluation
    }

    pub fn pow(&self, exp: Value) -> Value {
        Value {
            current_evaluation: self.evaluate_value().powf(exp.evaluate_value()),
            gradient: None,
            operation: ValueOperation::Exponentiation {
                base: Box::new(self.clone()),
                exp: Box::new(exp),
            },
        }
    }

    pub fn tanh(&self) -> Value {
        Value {
            current_evaluation: self.evaluate_value().tanh(),
            gradient: None,
            operation: ValueOperation::Tanh(Box::new(self.clone())),
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data=)")
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        Value {
            current_evaluation: self.current_evaluation + rhs.current_evaluation,
            gradient: None,
            operation: ValueOperation::Addition(Box::new(self), Box::new(rhs)),
        }
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        Value {
            current_evaluation: self.current_evaluation - rhs.current_evaluation,
            gradient: None,
            operation: ValueOperation::Subtraction(Box::new(self), Box::new(rhs)),
        }
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        Value {
            current_evaluation: self.current_evaluation * rhs.current_evaluation,
            gradient: None,
            operation: ValueOperation::Multiplication(Box::new(self), Box::new(rhs)),
        }
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        Value {
            current_evaluation: self.current_evaluation / rhs.current_evaluation,
            gradient: None,
            operation: ValueOperation::Division(Box::new(self), Box::new(rhs)),
        }
    }
}
