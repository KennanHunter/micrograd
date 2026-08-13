use micrograd::{Value, generate_svg, neuron::Neuron};

fn main() {
    let single_neuron = Neuron::new(5);

    let svg = generate_svg::build_graph(
        &single_neuron.infer(
            &([1.0, 0.0, 0.5, -0.2, -0.6]
                .into_iter()
                .map(|val| Value::leaf(val).with_label("Input"))
                .collect::<Vec<Value>>()),
        ),
        true,
    );

    svg::save("./renders/neuron_single.svg", &svg).unwrap();
}
