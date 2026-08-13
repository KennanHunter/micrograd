#![allow(unused)]

pub mod generate_svg;

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
    Division(Value, Value),
    Exponentiation { base: Value, exp: Value },
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
            | ValueOperation::Multiplication(left, right)
            | ValueOperation::Division(left, right) => vec![left.clone(), right.clone()],
            ValueOperation::Exponentiation { base, exp } => vec![base.clone(), exp.clone()],
            ValueOperation::Tanh(value) => vec![value.clone()],
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
                exp: exp,
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

    pub fn backprop(&mut self, gradient: f64) {
        let mut inner = self.inner_mut();

        match &mut inner.gradient {
            Some(existing) => inner.gradient = Some(gradient + *existing),
            None => inner.gradient = Some(gradient),
        }

        match &mut inner.operation {
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
        let current_evaluation = self.inner().current_evaluation / rhs.inner().current_evaluation;
        ValueInner {
            current_evaluation,
            gradient: None,
            operation: ValueOperation::Division(self, rhs),
        }
        .into()
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::create_dir_all, time::Duration};

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
            ValueInner {
                current_evaluation: 9.0,
                gradient: Some(1.0,),
                operation: ValueOperation::Multiplication(
                    (ValueInner {
                        current_evaluation: 3.0,
                        gradient: Some(3.0,),
                        operation: ValueOperation::Addition(
                            (ValueInner {
                                current_evaluation: 1.0,
                                gradient: Some(3.0,),
                                operation: ValueOperation::Leaf,
                            }
                            .into()),
                            (ValueInner {
                                current_evaluation: 2.0,
                                gradient: Some(3.0,),
                                operation: ValueOperation::Leaf,
                            }
                            .into()),
                        ),
                    }
                    .into()),
                    (ValueInner {
                        current_evaluation: 3.0,
                        gradient: Some(3.0,),
                        operation: ValueOperation::Leaf,
                    }
                    .into()),
                ),
            }
            .into()
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
            ValueInner {
                current_evaluation: 9.0,
                gradient: Some(1.0),
                operation: ValueOperation::Multiplication(
                    (ValueInner {
                        current_evaluation: 3.0,
                        gradient: Some(3.0),
                        operation: ValueOperation::Subtraction(
                            (ValueInner {
                                current_evaluation: 5.0,
                                gradient: Some(3.0),
                                operation: ValueOperation::Leaf,
                            }
                            .into()),
                            (ValueInner {
                                current_evaluation: 2.0,
                                gradient: Some(-3.0),
                                operation: ValueOperation::Leaf,
                            }
                            .into()),
                        ),
                    }
                    .into()),
                    (ValueInner {
                        current_evaluation: 3.0,
                        gradient: Some(3.0),
                        operation: ValueOperation::Leaf,
                    }
                    .into()),
                ),
            }
            .into()
        );
    }

    #[test]
    fn addition_self() {
        let a = Value::leaf(1.0);

        let mut expr = a.clone() + a;

        expr.backprop(1.0);

        assert_eq!(
            expr,
            ValueInner {
                current_evaluation: 2.0,
                gradient: Some(1.0),
                operation: ValueOperation::Addition(
                    (ValueInner {
                        current_evaluation: 1.0,
                        gradient: Some(2.0),
                        operation: ValueOperation::Leaf,
                    }
                    .into()),
                    (ValueInner {
                        current_evaluation: 1.0,
                        gradient: Some(2.0),
                        operation: ValueOperation::Leaf,
                    }
                    .into()),
                ),
            }
            .into()
        );
    }

    #[test]
    fn rc_paragraph() {
        struct Ex {}
        impl Drop for Ex {
            fn drop(&mut self) {
                eprintln!("Dropped")
            }
        }

        let a = Rc::new(Ex {});
        let b = a.clone();
        let c = a.clone();

        println!("Our Rc is referenced {} times", Rc::strong_count(&a));

        println!("Dropping a");
        drop(a);

        println!("Dropping b");
        drop(b);

        println!("Dropping c");
        drop(c);
    }

    #[test]
    #[should_panic]
    fn refcell_paragraph() {
        {
            let mut a = 0;
            let valid_mutable_reference = &mut a;

            *valid_mutable_reference = *valid_mutable_reference + 1;

            // let valid_immutable_reference = &a;
            //                              ^ cannot borrow a as immutable

            *valid_mutable_reference = *valid_mutable_reference + 1;
        }

        let mut a = RefCell::new(0);

        let mut valid_mutable_reference: RefMut<'_, i32> = a.borrow_mut();

        println!("Took valid mutable reference");

        let invalid_immutable_reference: Ref<'_, i32> = a.borrow();

        println!("Took a second mutable reference");

        *valid_mutable_reference += 1;

        drop(valid_mutable_reference);
    }
}
