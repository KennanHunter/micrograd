use crate::{Value, ValueOperation};
use std::collections::VecDeque;
use svg::Document;
use svg::node::Text as TextNode;
use svg::node::element::{Line, Rectangle, TSpan, Text};

struct NodeDescription {
    operation_label: String,
    value: Option<f64>,
    gradiant: Option<f64>,
    depth: usize,
    index: usize,
    parent: Option<usize>,
}

const NODE_WIDTH: f64 = 80.0;
const NODE_HEIGHT: f64 = 70.0;
const HORIZONTAL_SPACING: f64 = 20.0;
const VERTICAL_SPACING: f64 = 50.0;
const MARGIN: f64 = 20.0;
const LINE_HEIGHT: f64 = 16.0;
const MIN_WIDTH: f64 = 400.0;

pub fn build_graph(expr: &Value, with_values: bool) -> Document {
    // bfs the expression tree, tracking depth, parent index, and optionally value as we go
    let mut nodes: Vec<NodeDescription> = Vec::new();
    let mut queue: VecDeque<(Value, usize, Option<usize>, Option<f64>)> = VecDeque::new();
    queue.push_back((
        expr.clone(),
        0,
        None,
        if with_values {
            Some(expr.evaluate_value())
        } else {
            None
        },
    ));

    // Convert from the recursive Value to a simpler flat NodeDescription,
    // include depth so we can build the actual svg
    while let Some((val, depth, parent, value)) = queue.pop_front() {
        let index = nodes.len();
        nodes.push(describe_node(&val, depth, index, parent, value));

        if let Some((left, right)) = val.children() {
            let left_value = if with_values { Some(left.evaluate_value()) } else { None };
            let right_value = if with_values { Some(right.evaluate_value()) } else { None };
            queue.push_back((left, depth + 1, Some(index), left_value));
            queue.push_back((right, depth + 1, Some(index), right_value));
        }
    }

    // bucket node indices by depth so each row can be laid out independently
    let mut rows: Vec<Vec<usize>> = Vec::new();
    for node in &nodes {
        if rows.len() <= node.depth {
            // grow rows so rows[node.depth] is a valid slot
            rows.resize_with(node.depth + 1, Vec::new);
        }
        rows[node.depth].push(node.index);
    }

    // size the canvas off the widest row and the total depth
    let max_row_len = rows.iter().map(|r| r.len()).max().unwrap_or(1);
    let content_width = max_row_len as f64 * NODE_WIDTH
        + (max_row_len.saturating_sub(1)) as f64 * HORIZONTAL_SPACING;
    let width = (MARGIN * 2.0 + content_width).max(MIN_WIDTH);
    let height = MARGIN * 2.0
        + rows.len() as f64 * NODE_HEIGHT
        + rows.len().saturating_sub(1) as f64 * VERTICAL_SPACING;

    // center each row horizontally and stack them by depth
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
                .set("stroke", "white")
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
            .set("fill", "none")
            .set("stroke", "white")
            .set("stroke-width", 1);
        let cx = x + NODE_WIDTH / 2.0;
        let lines = [
            node.operation_label.clone(),
            node.value
                .map(|v| format!("v={:.3}", v))
                .unwrap_or_else(|| "v=?".to_owned()),
            node.gradiant
                .map(|g| format!("g={:.3}", g))
                .unwrap_or_else(|| "g=?".to_owned()),
        ];
        let total_height = LINE_HEIGHT * lines.len() as f64;
        let first_y = y + (NODE_HEIGHT - total_height) / 2.0 + LINE_HEIGHT * 0.75;

        let mut text = Text::new("")
            .set("text-anchor", "middle")
            .set("font-family", "monospace")
            .set("font-size", 12)
            .set("fill", "white");
        for (i, line) in lines.iter().enumerate() {
            let tspan = TSpan::new(line.as_str())
                .set("x", cx)
                .set("y", first_y + i as f64 * LINE_HEIGHT);
            text = text.add(tspan);
        }
        document = document.add(rect).add(text);
    }

    document
}

fn describe_node(
    expr: &Value,
    depth: usize,
    index: usize,
    parent: Option<usize>,
    value: Option<f64>,
) -> NodeDescription {
    let operation_label = match &expr.operation {
        ValueOperation::Leaf => "leaf".to_owned(),
        ValueOperation::Addition(_, _) => "+".to_owned(),
        ValueOperation::Subtraction(_, _) => "-".to_owned(),
        ValueOperation::Multiplication(_, _) => "*".to_owned(),
        ValueOperation::Division(_, _) => "/".to_owned(),
    };

    NodeDescription {
        operation_label,
        value,
        gradiant: expr.gradiant,
        depth,
        index,
        parent,
    }
}
