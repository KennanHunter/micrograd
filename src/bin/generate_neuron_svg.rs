use micrograd::{
    Value, generate_svg, l,
    neuron::Neuron,
    random::{SeededDistribution, UniformDistribution},
};

fn main() {
    let mut randomness = UniformDistribution::new("test-seed");

    let single_neuron = Neuron::new(5, &mut randomness);

    let svg = generate_svg::build_graph(
        &single_neuron.infer(
            &([1.0, 0.0, 0.5, -0.2, -0.6]
                .into_iter()
                .map(|val| l!(val).with_label("Input"))
                .collect::<Vec<Value>>()),
        ),
        true,
    );

    svg::save("./renders/neuron_single.svg", &svg).unwrap();
}
