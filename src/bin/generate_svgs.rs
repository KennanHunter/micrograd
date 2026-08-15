use micrograd::{generate_svg::build_graph, l};
use std::fs;

fn main() {
    let _ = fs::create_dir("./renders");

    svg::save(
        "./renders/image.svg",
        &build_graph(&(l!(2.0) + l!(4.0) * (l!(4.0) - l!(1.0))), true),
    )
    .unwrap();

    let mut big = l!(1.0);
    for i in 0..6 {
        let f = i as f64;
        let branch = (l!(f + 1.0) + l!(f + 2.0)) * (l!(f + 3.0) - l!(f + 4.0));
        big = big + branch;
    }

    svg::save("./renders/image-deep.svg", &build_graph(&big, true)).unwrap();

    let pow_expr = l!(3.0).pow(l!(2.0)) + l!(1.0);
    svg::save("./renders/image-pow.svg", &build_graph(&pow_expr, true)).unwrap();
}
