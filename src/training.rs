use std::fmt::Write;

use crate::{Value, l, neuron::MultiLayerPerceptron};

pub fn format_values(values: &[Value]) -> String {
    let mut out = String::from("[\n");
    for v in values {
        writeln!(out, "  {},", v).unwrap();
    }
    out.push(']');
    out
}

pub fn calculate_loss_from_outputs_and_targets(output: &[Value], target: &[Value]) -> Value {
    assert_eq!(
        output.len(),
        target.len(),
        "loss calculation should use the same length of output and expected"
    );

    let total_loss = output
        .iter()
        .zip(target.iter())
        .map(|(out, expected)| (out.clone() - expected.clone()).with_label("Loss"))
        .fold(l!(0.0), |acc, node_loss| {
            (acc + node_loss)
                .with_label("Loss Accumulation")
                .pow(l!(2.0))
        });

    total_loss
}

#[cfg(test)]
mod tests {
    use super::*;
}
