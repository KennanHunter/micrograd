#![allow(unused)]

pub mod generate_svg;
pub mod neuron;

#[cfg(test)]
mod tests;

use std::{
    cell::{Ref, RefCell, RefMut},
    fmt,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
    slice::Iter,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Value(Rc<RefCell<ValueInner>>);

#[derive(Clone, Debug, PartialEq)]
pub struct ValueInner {
    pub current_evaluation: f64,
    pub gradient: Option<f64>,
    pub operation: ValueOperation,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ValueOperation {
    Leaf,
    Addition(Value, Value),
    Subtraction(Value, Value),
    Multiplication(Value, Value),
    Exponentiation { base: Value, exp: Value },
    Exp { exp: Value },
    Tanh(Value),
}

impl From<ValueInner> for Value {
    fn from(value: ValueInner) -> Self {
        Value(Rc::new(RefCell::new(value)))
    }
}

impl Value {
    fn inner(&self) -> Ref<'_, ValueInner> {
        self.0
            .try_borrow()
            .expect("Value was already mutably borrowed")
    }

    fn inner_mut(&mut self) -> RefMut<'_, ValueInner> {
        self.0
            .try_borrow_mut()
            .expect("Value was already mutably borrowed")
    }

    pub fn leaf(current_evaluation: f64) -> Self {
        (ValueInner {
            current_evaluation,
            gradient: None,
            operation: ValueOperation::Leaf,
        })
        .into()
    }

    pub fn children(&self) -> Vec<Value> {
        let inner = self.inner();

        match &inner.operation {
            ValueOperation::Leaf => vec![],
            ValueOperation::Addition(left, right)
            | ValueOperation::Subtraction(left, right)
            | ValueOperation::Multiplication(left, right) => vec![left.clone(), right.clone()],
            ValueOperation::Exponentiation { base, exp } => vec![base.clone(), exp.clone()],
            ValueOperation::Exp { exp: value } | ValueOperation::Tanh(value) => vec![value.clone()],
        }
    }

    pub fn evaluate_value(&self) -> f64 {
        self.inner().current_evaluation
    }

    pub fn pow(&self, exp: Value) -> Value {
        ValueInner {
            current_evaluation: self.evaluate_value().powf(exp.evaluate_value()),
            gradient: None,
            operation: ValueOperation::Exponentiation {
                base: self.clone(),
                exp,
            },
        }
        .into()
    }

    pub fn tanh(&self) -> Value {
        ValueInner {
            current_evaluation: self.evaluate_value().tanh(),
            gradient: None,
            operation: ValueOperation::Tanh(self.clone()),
        }
        .into()
    }

    /// self raised to the power of e
    pub fn exp(&self) -> Value {
        ValueInner {
            current_evaluation: self.evaluate_value().exp(),
            gradient: None,
            operation: ValueOperation::Exp { exp: self.clone() },
        }
        .into()
    }

    pub fn backprop(&mut self, gradient: f64) {
        let mut inner = self.inner_mut();

        match &mut inner.gradient {
            Some(existing) => inner.gradient = Some(gradient + *existing),
            None => inner.gradient = Some(gradient),
        }

        let self_value = inner.current_evaluation;

        match &mut inner.operation {
            ValueOperation::Leaf => {}
            ValueOperation::Addition(left, right) => {
                left.backprop(gradient);
                right.backprop(gradient);
            }
            ValueOperation::Subtraction(left, right) => {
                left.backprop(gradient);
                right.backprop(-gradient);
            }
            ValueOperation::Multiplication(left, right) => {
                left.backprop(right.evaluate_value() * gradient);
                right.backprop(left.evaluate_value() * gradient);
            }
            ValueOperation::Tanh(child) => {
                child.backprop((1.0 - self_value.powi(2)) * gradient);
            }
            ValueOperation::Exponentiation { base, exp } => {
                let grad = exp.evaluate_value()
                    * base.evaluate_value().powf(exp.evaluate_value() - 1.0)
                    * gradient;

                base.backprop(grad);
            }
            ValueOperation::Exp { exp } => {
                let grad = exp.evaluate_value().exp() * gradient;

                exp.backprop(grad);
            }
        }
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        let current_evaluation = self.inner().current_evaluation + rhs.inner().current_evaluation;

        ValueInner {
            current_evaluation,
            gradient: None,
            operation: ValueOperation::Addition(self, rhs.clone()),
        }
        .into()
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        let current_evaluation = self.inner().current_evaluation - rhs.inner().current_evaluation;

        ValueInner {
            current_evaluation,
            gradient: None,
            operation: ValueOperation::Subtraction(self, rhs),
        }
        .into()
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        let current_evaluation = self.inner().current_evaluation * rhs.inner().current_evaluation;

        ValueInner {
            current_evaluation,
            gradient: None,
            operation: ValueOperation::Multiplication(self, rhs),
        }
        .into()
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        let current_evaluation =
            self.inner().current_evaluation * rhs.inner().current_evaluation.powi(-1);
        ValueInner {
            current_evaluation,
            gradient: None,
            operation: ValueOperation::Multiplication(self, rhs.pow(Value::leaf(-1.0))),
        }
        .into()
    }
}
