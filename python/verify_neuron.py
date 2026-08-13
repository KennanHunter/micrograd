import math
from micrograd.engine import Value

# Extend micrograd Value with tanh and exp (matches karpathy's makemore video).
def tanh(self):
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
        self.grad += (1 - t ** 2) * out.grad
    out._backward = _backward
    return out

def exp(self):
    x = self.data
    out = Value(math.exp(x), (self,), 'exp')

    def _backward():
        self.grad += out.data * out.grad
    out._backward = _backward
    return out

Value.tanh = tanh
Value.exp = exp

# --- Neuron: n = tanh(w1*x1 + w2*x2 + b) ---
x1 = Value(2.0)
x2 = Value(0.0)
w1 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.8813735870195432)

x1w1 = x1 * w1
x2w2 = x2 * w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b
o = n.tanh()

o.backward()

print("=== neuron: tanh(w1*x1 + w2*x2 + b) ===")
print(f"o.data  = {o.data!r}")
print(f"o.grad  = {o.grad!r}")
print(f"n.data  = {n.data!r}, n.grad  = {n.grad!r}")
print(f"x1.data = {x1.data!r}, x1.grad = {x1.grad!r}")
print(f"x2.data = {x2.data!r}, x2.grad = {x2.grad!r}")
print(f"w1.data = {w1.data!r}, w1.grad = {w1.grad!r}")
print(f"w2.data = {w2.data!r}, w2.grad = {w2.grad!r}")
print(f"b.data  = {b.data!r}, b.grad  = {b.grad!r}")

# --- Same neuron but tanh expanded through exp: tanh(x) = (e^(2x)-1)/(e^(2x)+1) ---
x1 = Value(2.0)
x2 = Value(0.0)
w1 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.8813735870195432)

n = x1 * w1 + x2 * w2 + b
e = (Value(2.0) * n).exp()
o = (e - Value(1.0)) / (e + Value(1.0))
o.backward()

print()
print("=== same neuron via exp: (e^(2n)-1)/(e^(2n)+1) ===")
print(f"o.data  = {o.data!r}")
print(f"n.data  = {n.data!r}, n.grad  = {n.grad!r}")
print(f"x1.grad = {x1.grad!r}")
print(f"w1.grad = {w1.grad!r}")
print(f"w2.grad = {w2.grad!r}")
print(f"b.grad  = {b.grad!r}")
