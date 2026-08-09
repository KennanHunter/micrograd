use micrograd::{Value, generate_svg::build_graph};
use std::fs;

fn main() {
    let _ = fs::create_dir("./renders");

    svg::save(
        "./renders/image.svg",
        &build_graph(
            &(Value::leaf(2.0) + Value::leaf(4.0) * (Value::leaf(4.0) - Value::leaf(1.0))),
            true,
        ),
    )
    .unwrap();

    let mut big = Value::leaf(1.0);
    for i in 0..6 {
        let f = i as f64;
        let branch = (Value::leaf(f + 1.0) + Value::leaf(f + 2.0))
            * (Value::leaf(f + 3.0) - Value::leaf(f + 4.0));
        big = big + branch;
    }

    svg::save("./renders/image-deep.svg", &build_graph(&big, true)).unwrap();
}
