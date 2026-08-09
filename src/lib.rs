#![allow(unused)]

pub mod generate_svg;

use std::{
    fmt,
    ops::{Add, Div, Mul, Sub},
};

#[derive(Clone)]
pub enum Value {
    Leaf(f64),
    Addition(Box<Value>, Box<Value>),
    Subtraction(Box<Value>, Box<Value>),
    Multiplication(Box<Value>, Box<Value>),
    Division(Box<Value>, Box<Value>),
}

impl Value {
    pub fn new(data: f64) -> Self {
        Value::Leaf(data)
    }

    pub fn children(&self) -> Option<(Value, Value)> {
        match self {
            Value::Leaf(_) => None,
            Value::Addition(left, right) => Some((*left.clone(), *right.clone())),
            Value::Subtraction(left, right) => Some((*left.clone(), *right.clone())),
            Value::Multiplication(left, right) => Some((*left.clone(), *right.clone())),
            Value::Division(left, right) => Some((*left.clone(), *right.clone())),
        }
    }

    // Evaluate from top down
    pub fn evaluate(&self) -> f64 {
        match self {
            Value::Leaf(val) => *val,
            Value::Addition(left, right) => left.evaluate() + right.evaluate(),
            Value::Subtraction(left, right) => left.evaluate() - right.evaluate(),
            Value::Multiplication(left, right) => left.evaluate() * right.evaluate(),
            Value::Division(left, right) => left.evaluate() / right.evaluate(),
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
        Value::Addition(Box::new(self), Box::new(rhs))
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        Value::Subtraction(Box::new(self), Box::new(rhs))
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        Value::Multiplication(Box::new(self), Box::new(rhs))
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        Value::Division(Box::new(self), Box::new(rhs))
    }
}
