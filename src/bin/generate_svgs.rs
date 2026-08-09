use micrograd::{Value, generate_svg::build_graph};
use std::fs;

fn main() {
    let _ = fs::create_dir("./renders");

    svg::save(
        "./renders/image.svg",
        &build_graph(
            &Value::Addition(Box::new(Value::Leaf(2.0)), Box::new(Value::Leaf(4.0))),
            true,
        ),
    )
    .unwrap();

    let mut big = Value::Leaf(1.0);
    for i in 0..6 {
        let f = i as f64;
        let branch = (Value::Leaf(f + 1.0) + Value::Leaf(f + 2.0))
            * (Value::Leaf(f + 3.0) - Value::Leaf(f + 4.0));
        big = big + branch;
    }

    svg::save("./renders/image-deep.svg", &build_graph(&big, true)).unwrap();
}
