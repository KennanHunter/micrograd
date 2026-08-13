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
            gradient: Some(1.0),
            label: None,
            operation: ValueOperation::Multiplication(
                (ValueInner {
                    current_evaluation: 3.0,
                    gradient: Some(3.0),
                    label: None,
                    operation: ValueOperation::Addition(
                        (ValueInner {
                            current_evaluation: 1.0,
                            gradient: Some(3.0),
                            operation: ValueOperation::Leaf,
                            label: None
                        }
                        .into()),
                        (ValueInner {
                            current_evaluation: 2.0,
                            gradient: Some(3.0),
                            operation: ValueOperation::Leaf,
                            label: None
                        }
                        .into()),
                    ),
                }
                .into()),
                (ValueInner {
                    current_evaluation: 3.0,
                    gradient: Some(3.0,),
                    operation: ValueOperation::Leaf,
                    label: None
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
            label: None,
            operation: ValueOperation::Multiplication(
                (ValueInner {
                    current_evaluation: 3.0,
                    gradient: Some(3.0),
                    label: None,
                    operation: ValueOperation::Subtraction(
                        (ValueInner {
                            current_evaluation: 5.0,
                            gradient: Some(3.0),
                            label: None,
                            operation: ValueOperation::Leaf,
                        }
                        .into()),
                        (ValueInner {
                            current_evaluation: 2.0,
                            gradient: Some(-3.0),
                            label: None,
                            operation: ValueOperation::Leaf,
                        }
                        .into()),
                    ),
                }
                .into()),
                (ValueInner {
                    current_evaluation: 3.0,
                    gradient: Some(3.0),
                    label: None,
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
            label: None,
            operation: ValueOperation::Addition(
                (ValueInner {
                    current_evaluation: 1.0,
                    gradient: Some(2.0),
                    operation: ValueOperation::Leaf,
                    label: None,
                }
                .into()),
                (ValueInner {
                    current_evaluation: 1.0,
                    gradient: Some(2.0),
                    operation: ValueOperation::Leaf,
                    label: None,
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

fn grad(v: &Value) -> f64 {
    v.inner().gradient.expect("no gradient set")
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "{label}: expected {expected}, got {actual}"
    );
}

/// Emulates a single neuron `tanh(w1*x1 + w2*x2 + b)`.
/// Expected values generated from karpathy's micrograd via `python/verify_neuron.py`.
#[test]
fn neuron_tanh() {
    let x1 = Value::leaf(2.0);
    let x2 = Value::leaf(0.0);
    let w1 = Value::leaf(-3.0);
    let w2 = Value::leaf(1.0);
    let b = Value::leaf(6.881373587019543);

    let n = x1.clone() * w1.clone() + x2.clone() * w2.clone() + b.clone();
    let mut o = n.tanh();

    o.backprop(1.0);

    assert_close(o.evaluate_value(), 0.7071067811865476, "o.data");
    assert_close(grad(&x1), -1.4999999999999996, "x1.grad");
    assert_close(grad(&x2), 0.4999999999999999, "x2.grad");
    assert_close(grad(&w1), 0.9999999999999998, "w1.grad");
    assert_close(grad(&w2), 0.0, "w2.grad");
    assert_close(grad(&b), 0.4999999999999999, "b.grad");
}

/// Same neuron but tanh is expanded through `exp`:
/// `tanh(n) = (e^(2n) - 1) / (e^(2n) + 1)`.
/// Exercises Exp, Sub, and Div (implemented via pow(-1)).
#[test]
fn neuron_tanh_via_exp() {
    let x1 = Value::leaf(2.0);
    let x2 = Value::leaf(0.0);
    let w1 = Value::leaf(-3.0);
    let w2 = Value::leaf(1.0);
    let b = Value::leaf(6.881373587019543);

    let n = x1.clone() * w1.clone() + x2.clone() * w2.clone() + b.clone();
    let e = (Value::leaf(2.0) * n.clone()).exp();
    let mut o = (e.clone() - Value::leaf(1.0)) / (e + Value::leaf(1.0));

    o.backprop(1.0);

    assert_close(o.evaluate_value(), 0.7071067811865477, "o.data");
    assert_close(grad(&x1), -1.5, "x1.grad");
    assert_close(grad(&w1), 1.0, "w1.grad");
    assert_close(grad(&w2), 0.0, "w2.grad");
    assert_close(grad(&b), 0.5, "b.grad");
}
