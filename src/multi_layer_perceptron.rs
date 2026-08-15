use crate::neuron::Layer;
use crate::training::calculate_loss_from_outputs_and_targets;
use crate::{Value, l};

pub struct MultiLayerPerceptron {
    pub(crate) layers: Vec<Layer>,
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

    /// Assumes gradient already calculated.
    pub fn step_all_parameters_by_learning_rate(&self, learning_rate: f64) {
        for mut param in self.parameters() {
            param.step_value(learning_rate)
        }
    }

    pub fn train(
        &mut self,
        iterations: usize,
        learning_rate: f64,
        inputs: &[&[Value]],
        targets: &[&[Value]],
    ) {
        for i in 0..iterations {
            // 1. Running a forward inference pass.
            let mut total_loss = l!(0.0).with_label("Total Loss");

            for (single_example_inputs, single_example_targets) in inputs.iter().zip(targets) {
                let output = self.infer(single_example_inputs);

                // 2. Calculating the loss
                let loss = calculate_loss_from_outputs_and_targets(&output, single_example_targets);

                total_loss = (total_loss + loss.clone()).with_label("Ex Loss Accumulation");
            }

            // 3. Zeroing the gradient from the last pass.
            total_loss.zero_gradient();

            // 4. Calculating the gradient given the loss.
            total_loss.backprop(1.0);

            // 5. Stepping every parameter by a small amount
            self.step_all_parameters_by_learning_rate(learning_rate);

            let (grad_min, grad_max, grad_sum) = self.parameters().fold(
                (f64::INFINITY, f64::NEG_INFINITY, 0.0),
                |(lo, hi, sum), p| {
                    let g = p.inner().gradient.unwrap_or(0.0);
                    (lo.min(g), hi.max(g), sum + g.abs())
                },
            );
            let (val_min, val_max) =
                self.parameters()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                        let v = p.inner().current_evaluation;
                        (lo.min(v), hi.max(v))
                    });

            println!(
                "Iter {i}: Loss = {:.4}  |  params ∈ [{:.4}, {:.4}]  grads ∈ [{:.4}, {:.4}]  Σ|grad| = {:.4}",
                total_loss.evaluate_value(),
                val_min,
                val_max,
                grad_min,
                grad_max,
                grad_sum,
            );
        }
    }
}
