#![allow(unused)]

use std::{
    fmt,
    ops::{Add, Div, Mul, Sub},
};

enum Value {
    Leaf(f64),
    Addition(Box<Value>, Box<Value>),
    Subtraction(Box<Value>, Box<Value>),
    Multiplication(Box<Value>, Box<Value>),
    Division(Box<Value>, Box<Value>),
}

enum Example {
    Small(u8),
    Big([u8; 20000]),
}

impl Value {
    fn new(data: f64) -> Self {
        Value::Leaf(data)
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
