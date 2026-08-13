use std::iter::repeat;

use crate::Value;

pub struct Neuron {
    bias: Value,
    weights: Vec<Value>,
}

impl Neuron {
    pub fn new(size: usize) -> Self {
        // TODO: use random sampling?
        let weights =
            Vec::from_iter(repeat(Value::leaf(0.0).with_label("Neuron Weight")).take(size));

        Neuron {
            bias: Value::leaf(0.0).with_label("Neuron Bias"),
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
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn infer(&self, input: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.infer(input)).collect()
    }
}

pub struct MultiLayerPerceptron {
    layers: Vec<Layer>,
}

impl MultiLayerPerceptron {
    fn infer(&self, input: &[Value]) -> Vec<Value> {
        self.layers
            .iter()
            .fold(input.to_vec(), |acc, layer| layer.infer(&acc))
    }
}
