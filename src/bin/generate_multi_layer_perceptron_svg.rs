use micrograd::{
    generate_svg, l,
    multi_layer_perceptron::MultiLayerPerceptron,
    random::{SeededDistribution, UniformDistribution},
    training::{calculate_loss_from_outputs_and_targets, format_values},
};

fn main() {
    let inputs = vec![
        vec![l!(2.0), l!(3.0), l!(-1.0)],
        vec![l!(3.0), l!(-1.0), l!(0.5)],
        vec![l!(0.5), l!(1.0), l!(1.0)],
        vec![l!(1.0), l!(1.0), l!(-1.0)],
    ];

    let targets = [l!(1.0), l!(-1.0), l!(-1.0), l!(1.0)];

    let mut randomness = UniformDistribution::new("test-seed");

    let perceptron = MultiLayerPerceptron::new(&[3, 1], &mut randomness);

    let mut total_loss = l!(0.0);

    for (input, target) in inputs.into_iter().zip(targets) {
        let output = perceptron.infer(&input);

        let loss = calculate_loss_from_outputs_and_targets(&output, std::slice::from_ref(&target));

        total_loss = (total_loss + loss.clone()).with_label("Example Loss Accumulation");

        println!(
            "Given inputs {} with target {} and an output of {}, we got a loss of {}",
            format_values(&input),
            target,
            format_values(&output),
            loss
        );
    }

    println!("Our total loss between all four examples is {}", total_loss);

    total_loss.backprop(1.0);

    let graph = generate_svg::build_graph(&total_loss.with_label("Total Loss"), true);

    svg::save("./renders/mlp_big_example.svg", &graph).unwrap();
}
