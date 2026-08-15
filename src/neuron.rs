use crate::{Value, l};
use std::iter;

#[derive(Clone)]
pub struct Neuron {
    bias: Value,
    weights: Vec<Value>,
}

impl Neuron {
    pub fn new(size: usize) -> Self {
        // TODO: use random sampling?
        let weights = (0..size)
            .map(|_| l!(0.0).with_label("Neuron Weight"))
            .collect();

        Neuron {
            bias: l!(0.0).with_label("Neuron Bias"),
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
    fn new(input_size: usize, output_size: usize) -> Layer {
        Layer {
            neurons: (0..output_size).map(|_| Neuron::new(input_size)).collect(),
        }
    }

    fn infer(&self, input: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.infer(input)).collect()
    }

    fn parameters(&self) -> impl Iterator<Item = Value> + '_ {
        self.neurons.iter().flat_map(|neuron| neuron.parameters())
    }
}

pub struct MultiLayerPerceptron {
    layers: Vec<Layer>,
}

impl MultiLayerPerceptron {
    pub fn new(sizes: &[usize]) -> MultiLayerPerceptron {
        let layers = sizes.windows(2).map(|w| Layer::new(w[0], w[1])).collect();

        MultiLayerPerceptron { layers }
    }

    pub fn infer(&self, input: &[Value]) -> Vec<Value> {
        self.layers
            .iter()
            .fold(input.to_vec(), |acc, layer| layer.infer(&acc))
    }

    pub fn parameters(&self) -> impl Iterator<Item = Value> + '_ {
        self.layers.iter().flat_map(|layer| layer.parameters())
    }
}
