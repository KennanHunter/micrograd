use micrograd::{Value, generate_svg, neuron::Neuron};

fn main() {
    let single_neuron = Neuron::new(5);

    let svg = generate_svg::build_graph(
        &single_neuron.infer(&[
            Value::leaf(1.0),
            Value::leaf(0.0),
            Value::leaf(0.5),
            Value::leaf(-0.2),
            Value::leaf(-0.6),
        ]),
        true,
    );

    svg::save("./renders/neuron_single.svg", &svg).unwrap();
}
