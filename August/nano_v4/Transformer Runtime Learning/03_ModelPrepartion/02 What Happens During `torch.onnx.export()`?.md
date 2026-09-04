# Lesson 2 — What Happens During `torch.onnx.export()`?

## 1. Why do we need ONNX export?

After adaptation we have:

```text
HF model
   ↓
Adaptation
   ↓
model_qc

model_qc is still a PyTorch nn.Module.

It is a Python program that can execute the model.

The next stage needs an explicit computation graph.

So conceptually:

model_qc
   +
concrete example inputs
   ↓
graph capture
   ↓
ONNX
2. What is the exporter doing?

The exporter takes the model and concrete inputs and captures a representable tensor computation.

Conceptually:

Python model
     +
example inputs
     ↓
model execution
     ↓
tensor operations
     ↓
captured graph
     ↓
ONNX

It does NOT simply copy the Python source code into ONNX.

3. Python Program vs Graph

Example Python:

def forward(self, x):
    x = self.linear(x)
    x = torch.relu(x)
    return x

Conceptual graph:

Input
  ↓
Linear
  ↓
ReLU
  ↓
Output

The graph represents the computation rather than the Python implementation.

4. Why are example inputs needed?

The computation can depend on the input.

Example:

if x.shape[1] == 128:
    x = self.layer1(x)
else:
    x = self.layer2(x)

If the example input has:

x.shape[1] = 128

the captured execution follows layer1.

Therefore:

Python program
      +
concrete input
      ↓
concrete execution
      ↓
captured computation

The graph represents the captured computation.

5. What happens to Python loops?

Suppose:

for layer in self.layers:
    x = layer(x)

If there are 3 layers, the execution is:

x
 ↓
layer0
 ↓
layer1
 ↓
layer2
 ↓
output

The graph can represent the concrete sequence of operations.

The Python loop itself does not need to remain in the graph.

This is conceptually called loop unrolling.

6. What happens to Python branches?

Example:

if use_sliding:
    ...
else:
    ...

If use_sliding is known during the captured execution, the resulting graph contains the operations corresponding to that execution path.

Therefore:

Python branch
      ↓
execution decision
      ↓
concrete graph path
7. Gemma4 Example

The adapted Gemma4 model conceptually contains:

for i, decoder_layer in enumerate(self.layers):

    if layer_type[i] == "sliding_attention":
        ...
    else:
        ...

    hidden_states = decoder_layer(...)

During graph capture:

35-layer loop
      ↓
35 concrete sequences of operations

The layer-type decisions are resolved for the concrete configuration.

8. Cache Representation

At the PyTorch level, cache behavior may involve abstractions such as:

past_key_values
DynamicCache

After adaptation, the desired cache interface is made explicit through tensors.

During export, the graph therefore contains explicit tensor inputs/outputs such as:

past_key_0_in
past_value_0_in
...
past_key_0_out
past_value_0_out
...

The Python cache abstraction is not carried into ONNX as an arbitrary Python object.

9. Mask Representation

Adaptation makes masks explicit model inputs:

attention_mask
swa_attention_mask

During export, these become graph inputs with concrete tensor structures/shapes based on the example inputs.

This is one reason adaptation must happen before preparation.

10. What Happens to Weights?

Model parameters such as:

Linear weight
Linear bias

are represented in the graph/model as constants or initializers/associated model data.

The graph therefore contains the information required to represent the computation.

11. What Does NOT Happen?

torch.onnx.export() does NOT mean:

PyTorch
   ↓
NPU executable

Instead:

PyTorch
   ↓
ONNX graph

ONNX is an intermediate representation.

Compilation comes later.

12. NanoV4 Preparation

Inside prepare_model() the first important transformation is conceptually:

model_qc
   +
example inputs
   ↓
torch.onnx.export()
   ↓
temporary Model.onnx

This ONNX is then consumed by the QAIRT conversion stage.

13. Why Adaptation Comes Before Export

Adaptation changes the model program so that the desired target-friendly computation can be captured.

Preparation/export then captures that computation.

Therefore:

Adaptation
=
change/rewrite the program

Preparation
=
capture and represent the computation
14. Five Questions
1. What representation are we in?

Before:

model_qc = PyTorch program

After:

Model.onnx = computation graph
2. What information does it contain?

model_qc contains Python model behavior, modules, weights, configuration and tensor operations.

Example inputs provide a concrete execution scenario.

3. What is still dynamic before capture?

Potentially:

loops
branches
Python object behavior
runtime decisions
4. Why does the next stage need ONNX?

QAIRT needs an explicit graph representation rather than arbitrary Python model code.

5. What changed?
Python model program
       +
concrete inputs
       ↓
captured tensor computation
       ↓
ONNX graph
15. Main Mental Model

Do not think:

torch.onnx.export()
=
convert Python source to ONNX

Think:

PyTorch model
      +
concrete example inputs
      ↓
capture tensor computation
      ↓
represent it as a graph
      ↓
ONNX

The graph represents the computation, not the original Python program.


---

## What we have now

We have established two very important boundaries:

```text
HF
 ↓
Adaptation
 ↓
model_qc

"Make the model program target-friendly."

Then:

model_qc
 ↓
ONNX

"Capture the concrete computation into an explicit graph."
