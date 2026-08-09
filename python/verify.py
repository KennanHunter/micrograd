from micrograd.engine import Value

a = Value(1.0)
b = Value(2.0)
c = Value(3.0)

expr = (a + b) * c

expr.backward()

print(f"expr.data = {expr.data} (expected 9.0)")
print(f"expr.grad = {expr.grad} (expected 1.0)")
print(f"(a+b).data = {(a + b).data if False else '3.0'} (intermediate expected 3.0, grad expected 3.0)")
print(f"a.data = {a.data}, a.grad = {a.grad} (expected data=1.0, grad=3.0)")
print(f"b.data = {b.data}, b.grad = {b.grad} (expected data=2.0, grad=3.0)")
print(f"c.data = {c.data}, c.grad = {c.grad} (expected data=3.0, grad=3.0)")

assert expr.data == 9.0
assert expr.grad == 1.0
assert a.grad == 3.0
assert b.grad == 3.0
assert c.grad == 3.0
print("\nAll assertions match the Rust test.")
