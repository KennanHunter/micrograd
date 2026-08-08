use std::collections::VecDeque;

use svg::Document;
use svg::node::element::{Line, Rectangle, Text};
use svg::node::Text as TextNode;

use crate::Value;

struct NodeDescription {
    label: String,
    depth: usize,
    index: usize,
    parent: Option<usize>,
}

const NODE_WIDTH: f64 = 60.0;
const NODE_HEIGHT: f64 = 30.0;
const HORIZONTAL_SPACING: f64 = 20.0;
const VERTICAL_SPACING: f64 = 50.0;
const MARGIN: f64 = 20.0;

pub fn build_graph(expr: &Value) -> Document {
    let mut nodes: Vec<NodeDescription> = Vec::new();
    let mut queue: VecDeque<(Value, usize, Option<usize>)> = VecDeque::new();
    queue.push_back((expr.clone(), 0, None));

    while let Some((val, depth, parent)) = queue.pop_front() {
        let index = nodes.len();
        nodes.push(describe_node(&val, depth, index, parent));

        if let Some((left, right)) = val.children() {
            queue.push_back((left, depth + 1, Some(index)));
            queue.push_back((right, depth + 1, Some(index)));
        }
    }

    let mut rows: Vec<Vec<usize>> = Vec::new();
    for node in &nodes {
        if rows.len() <= node.depth {
            rows.resize_with(node.depth + 1, Vec::new);
        }
        rows[node.depth].push(node.index);
    }

    let max_row_len = rows.iter().map(|r| r.len()).max().unwrap_or(1);
    let width = MARGIN * 2.0
        + max_row_len as f64 * NODE_WIDTH
        + (max_row_len.saturating_sub(1)) as f64 * HORIZONTAL_SPACING;
    let height = MARGIN * 2.0
        + rows.len() as f64 * NODE_HEIGHT
        + rows.len().saturating_sub(1) as f64 * VERTICAL_SPACING;

    let mut positions: Vec<(f64, f64)> = vec![(0.0, 0.0); nodes.len()];
    for (depth, row) in rows.iter().enumerate() {
        let row_width = row.len() as f64 * NODE_WIDTH
            + (row.len().saturating_sub(1)) as f64 * HORIZONTAL_SPACING;
        let start_x = (width - row_width) / 2.0;
        let y = MARGIN + depth as f64 * (NODE_HEIGHT + VERTICAL_SPACING);
        for (i, &idx) in row.iter().enumerate() {
            let x = start_x + i as f64 * (NODE_WIDTH + HORIZONTAL_SPACING);
            positions[idx] = (x, y);
        }
    }

    let mut document = Document::new().set("viewBox", (0, 0, width, height));

    for node in &nodes {
        if let Some(parent) = node.parent {
            let (px, py) = positions[parent];
            let (cx, cy) = positions[node.index];
            let line = Line::new()
                .set("x1", px + NODE_WIDTH / 2.0)
                .set("y1", py + NODE_HEIGHT)
                .set("x2", cx + NODE_WIDTH / 2.0)
                .set("y2", cy)
                .set("stroke", "black")
                .set("stroke-width", 1);
            document = document.add(line);
        }
    }

    for node in &nodes {
        let (x, y) = positions[node.index];
        let rect = Rectangle::new()
            .set("x", x)
            .set("y", y)
            .set("width", NODE_WIDTH)
            .set("height", NODE_HEIGHT)
            .set("fill", "white")
            .set("stroke", "black")
            .set("stroke-width", 1);
        let text = Text::new("")
            .set("x", x + NODE_WIDTH / 2.0)
            .set("y", y + NODE_HEIGHT / 2.0)
            .set("text-anchor", "middle")
            .set("dominant-baseline", "middle")
            .set("font-family", "monospace")
            .set("font-size", 14)
            .add(TextNode::new(&node.label));
        document = document.add(rect).add(text);
    }

    document
}

fn describe_node(expr: &Value, depth: usize, index: usize, parent: Option<usize>) -> NodeDescription {
    let label = match expr {
        Value::Leaf(leaf) => format!("{:.3}", leaf),
        Value::Addition(_, _) => "+".to_owned(),
        Value::Subtraction(_, _) => "-".to_owned(),
        Value::Multiplication(_, _) => "*".to_owned(),
        Value::Division(_, _) => "/".to_owned(),
    };
    NodeDescription { label, depth, index, parent }
}
