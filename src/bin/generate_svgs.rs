use micrograd::{Value, generate_svg::build_graph};

fn main() {
    svg::save(
        "image.svg",
        &build_graph(&Value::Addition(
            Box::new(Value::Leaf(2.0)),
            Box::new(Value::Leaf(4.0)),
        )),
    )
    .unwrap();
}
