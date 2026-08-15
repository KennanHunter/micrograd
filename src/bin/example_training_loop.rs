use micrograd::{l, multi_layer_perceptron::MultiLayerPerceptron};

fn main() {
    let inputs = vec![
        vec![l!(2.0), l!(3.0), l!(-1.0)],
        vec![l!(3.0), l!(-1.0), l!(0.5)],
        vec![l!(0.5), l!(1.0), l!(1.0)],
        vec![l!(1.0), l!(1.0), l!(-1.0)],
    ];

    let targets = vec![vec![l!(1.0)], vec![l!(-1.0)], vec![l!(-1.0)], vec![l!(1.0)]];

    let mut perceptron = MultiLayerPerceptron::new(&[3, 4, 4, 1]);

    let input_refs: Vec<&[_]> = inputs.iter().map(Vec::as_slice).collect();
    let target_refs: Vec<&[_]> = targets.iter().map(Vec::as_slice).collect();

    perceptron.train(10, 0.01, &input_refs, &target_refs);
}
