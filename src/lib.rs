#![allow(unused)]

pub mod generate_svg;

use std::{
    fmt,
    ops::{Add, Div, Mul, Sub},
    slice::Iter,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    pub current_evaluation: f64,
    pub gradient: Option<f64>,
    pub operation: ValueOperation,
}

#[derive(Clone, Debug, PartialEq)]
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

    pub fn children(&self) -> Vec<&Value> {
        match &self.operation {
            ValueOperation::Leaf => vec![],
            ValueOperation::Addition(left, right)
            | ValueOperation::Subtraction(left, right)
            | ValueOperation::Multiplication(left, right)
            | ValueOperation::Division(left, right) => vec![left, right],
            ValueOperation::Exponentiation { base, exp } => vec![base, exp],
            ValueOperation::Tanh(value) => vec![value],
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

    pub fn backprop(&mut self, gradient: f64) {
        self.gradient = Some(gradient);

        match &mut self.operation {
            ValueOperation::Leaf => {}
            ValueOperation::Addition(left, right) => {
                left.backprop(gradient);
                right.backprop(gradient);
            }
            ValueOperation::Subtraction(left, right) => {
                left.backprop(gradient);
                // check if this is right
                right.backprop(-gradient);
            }

            ValueOperation::Multiplication(left, right) => {
                left.backprop(right.evaluate_value() * gradient);
                right.backprop(left.evaluate_value() * gradient);
            }

            ValueOperation::Division(top, bottom) => {
                // let bottom_gradiant = todo!();

                // let top_gradiant = (gradient * bottom.evaluate_value().powi(2)
                //     + top.evaluate_value() * bottom_gradiant)
                //     / bottom.evaluate_value();

                // top.backprop(top_gradiant);
                //
                todo!("Implement division")
            }

            ValueOperation::Tanh(child) => {
                child.backprop((1.0 - child.evaluate_value().powi(2)) * gradient);
            }

            _ => todo!(""),
        }
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

#[cfg(test)]
mod tests {
    use std::fs::create_dir_all;

    use super::*;
    use crate::generate_svg::build_graph;

    #[test]
    fn simple_example() {
        let a = Value::leaf(1.0);
        let b = Value::leaf(2.0);
        let c = Value::leaf(3.0);

        let mut expr = (a + b) * c;

        let _ = create_dir_all("./renders/example");

        svg::save("./renders/example/values.svg", &build_graph(&expr, true)).unwrap();

        expr.backprop(1.0);

        svg::save(
            "./renders/example/values-with-backprop.svg",
            &build_graph(&expr, true),
        )
        .unwrap();

        println!("{:#?}", expr);

        assert_eq!(
            expr,
            Value {
                current_evaluation: 9.0,
                gradient: Some(1.0,),
                operation: ValueOperation::Multiplication(
                    Box::new(Value {
                        current_evaluation: 3.0,
                        gradient: Some(3.0,),
                        operation: ValueOperation::Addition(
                            Box::new(Value {
                                current_evaluation: 1.0,
                                gradient: Some(3.0,),
                                operation: ValueOperation::Leaf,
                            }),
                            Box::new(Value {
                                current_evaluation: 2.0,
                                gradient: Some(3.0,),
                                operation: ValueOperation::Leaf,
                            }),
                        ),
                    }),
                    Box::new(Value {
                        current_evaluation: 3.0,
                        gradient: Some(3.0,),
                        operation: ValueOperation::Leaf,
                    }),
                ),
            }
        );
    }

    #[test]
    fn subtraction_example() {
        let a = Value::leaf(5.0);
        let b = Value::leaf(2.0);
        let c = Value::leaf(3.0);

        let mut expr = (a - b) * c;

        expr.backprop(1.0);

        assert_eq!(
            expr,
            Value {
                current_evaluation: 9.0,
                gradient: Some(1.0),
                operation: ValueOperation::Multiplication(
                    Box::new(Value {
                        current_evaluation: 3.0,
                        gradient: Some(3.0),
                        operation: ValueOperation::Subtraction(
                            Box::new(Value {
                                current_evaluation: 5.0,
                                gradient: Some(3.0),
                                operation: ValueOperation::Leaf,
                            }),
                            Box::new(Value {
                                current_evaluation: 2.0,
                                gradient: Some(-3.0),
                                operation: ValueOperation::Leaf,
                            }),
                        ),
                    }),
                    Box::new(Value {
                        current_evaluation: 3.0,
                        gradient: Some(3.0),
                        operation: ValueOperation::Leaf,
                    }),
                ),
            }
        );
    }

    #[test]
    fn addition_self() {
        let a = Value::leaf(1.0);

        let mut expr = a.clone() + a;

        expr.backprop(1.0);

        assert_eq!(
            expr,
            Value {
                current_evaluation: 2.0,
                gradient: Some(1.0),
                operation: ValueOperation::Addition(
                    Box::new(Value {
                        current_evaluation: 1.0,
                        gradient: Some(2.0),
                        operation: ValueOperation::Leaf,
                    }),
                    Box::new(Value {
                        current_evaluation: 1.0,
                        gradient: Some(2.0),
                        operation: ValueOperation::Leaf,
                    }),
                ),
            }
        );
    }
}
