use crate::{Value, l, random::SeededDistribution};
use std::iter;

#[derive(Clone)]
pub struct Neuron {
    bias: Value,
    weights: Vec<Value>,
}

impl Neuron {
    pub fn new(size: usize, randomness: &mut impl SeededDistribution) -> Self {
        let weights = (0..size)
            .map(|_| l!(randomness.next()).with_label("Neuron Weight"))
            .collect();

        Neuron {
            bias: l!(randomness.next()).with_label("Neuron Bias"),
            weights,
        }
    }

    pub fn infer(&self, x: &[Value]) -> Value {
        let mut activation = self.bias.clone();

        assert_eq!(
            x.len(),
            self.weights.len(),
            "Weights and input for neuron did not match. Neuron weight size: {}, input: {:#?}",
            self.weights.len(),
            x
        );

        for (input, weight) in x.iter().zip(self.weights.iter()) {
            let weighted_input = (input.clone() * weight.clone()).with_label("Input Weighting");

            activation = (activation + weighted_input).with_label("Input Accumulation");
        }

        activation.tanh().with_label("Neuron Activation")
    }

    pub fn parameters(&self) -> impl Iterator<Item = Value> + '_ {
        self.weights
            .iter()
            .cloned()
            .chain(iter::once(self.bias.clone()))
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        randomness: &mut impl SeededDistribution,
    ) -> Layer {
        Layer {
            neurons: (0..output_size)
                .map(|_| Neuron::new(input_size, randomness))
                .collect(),
        }
    }

    pub fn infer(&self, input: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.infer(input)).collect()
    }

    pub fn parameters(&self) -> impl Iterator<Item = Value> + '_ {
        self.neurons.iter().flat_map(|neuron| neuron.parameters())
    }
}
