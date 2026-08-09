#![allow(unused)]

pub mod generate_svg;

use std::{
    fmt,
    ops::{Add, Div, Mul, Sub},
};

#[derive(Clone)]
pub struct Value {
    pub current_evaluation: f64,
    pub gradiant: Option<f64>,
    pub operation: ValueOperation,
}

#[derive(Clone)]
pub enum ValueOperation {
    Leaf,
    Addition(Box<Value>, Box<Value>),
    Subtraction(Box<Value>, Box<Value>),
    Multiplication(Box<Value>, Box<Value>),
    Division(Box<Value>, Box<Value>),
}

impl Value {
    pub fn leaf(current_evaluation: f64) -> Self {
        Value {
            current_evaluation,
            gradiant: None,
            operation: ValueOperation::Leaf,
        }
    }

    pub fn children(&self) -> Option<(Value, Value)> {
        match &self.operation {
            ValueOperation::Leaf => None,
            ValueOperation::Addition(left, right)
            | ValueOperation::Subtraction(left, right)
            | ValueOperation::Multiplication(left, right)
            | ValueOperation::Division(left, right) => Some((*left.clone(), *right.clone())),
        }
    }

    pub fn evaluate_value(&self) -> f64 {
        self.current_evaluation
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
            gradiant: None,
            operation: ValueOperation::Addition(Box::new(self), Box::new(rhs)),
        }
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        Value {
            current_evaluation: self.current_evaluation - rhs.current_evaluation,
            gradiant: None,
            operation: ValueOperation::Subtraction(Box::new(self), Box::new(rhs)),
        }
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        Value {
            current_evaluation: self.current_evaluation * rhs.current_evaluation,
            gradiant: None,
            operation: ValueOperation::Multiplication(Box::new(self), Box::new(rhs)),
        }
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        Value {
            current_evaluation: self.current_evaluation / rhs.current_evaluation,
            gradiant: None,
            operation: ValueOperation::Division(Box::new(self), Box::new(rhs)),
        }
    }
}
