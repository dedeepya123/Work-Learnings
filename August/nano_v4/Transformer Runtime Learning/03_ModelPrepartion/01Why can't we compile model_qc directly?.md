# Lesson 1 — Why can't we compile model_qc directly?

We first need to understand one simple question:

## What exactly is model_qc?

Once that is clear, the need for Model Preparation becomes almost obvious.

1. Where are we?

We already learned the first transformation:
``` text
Hugging Face Gemma4
        ↓
    Adaptation
        ↓
     model_qc

We know adaptation changes the model so that it is more suitable for the target environment.
```
For example, in your Gemma4 adaptation we changed things such as:

- mask construction
- cache interface
- RoPE representation
- static input shapes
- target-friendly operations
- model inputs/outputs

But after all of that: model_qc is still: torch.nn.Module

That fact is the starting point of this lesson.

## 2. What does nn.Module actually mean?

Suppose we have a tiny model:
``` python
class MyModel(nn.Module):

    def __init__(self):
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

When we create:

model = MyModel()

what do we have?

We have an object.

That object contains:
 
MyModel object
│
├── linear1
├── linear2
├── weights
├── parameters
├── methods
├── forward()
└── Python behavior

So the model is not just a mathematical graph.

It is a Python object that knows how to execute the computation.
```
## 3. What happens when we call it?

When we do:
``` python
y = model(x)

roughly:

Python
  ↓
model.__call__()
  ↓
forward()
  ↓
linear1
  ↓
ReLU
  ↓
linear2
  ↓
y

The computation happens right now.

This is what we mean by eager execution.

The model says:

"Give me tensors, and I will execute the operations now."
``` 
## 4. This is different from having a graph
``` python
Consider:

x = self.linear1(x)
x = torch.relu(x)
x = self.linear2(x)

A human can immediately see:

x
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
y

But Python itself is executing instructions one after another.

There isn't necessarily a standalone object sitting there saying:

GRAPH:

Input
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
Output

That's the distinction we need.
```
## 5. Program vs computation graph
``` python
Think about a normal Python program:

a = 10

if a > 5:
    print("hello")
else:
    print("bye")

This is a program.

The program can make decisions.

Similarly:

def forward(self, x):

    if something:
        x = self.layer1(x)
    else:
        x = self.layer2(x)

    return x

is a program.

The program decides what to execute.

A computation graph is different.

It says:

Input
  ↓
Layer1
  ↓
Output

There is no ambiguity about which path exists.
```
So: **Python model = instructions for executing computation** and **Computation graph = explicit description of the computation**

## 6. Why does a compiler care?

Now imagine you are a compiler.

Someone gives you: model_qc

What would you like to know?

You want to know:

- What are the inputs?
- What operations happen?
- In what order?
- Which tensor feeds which operation?
- What are the tensor shapes?
- What are the dtypes?
- Which values are constants?
- What are the outputs?
- Which operations depend on which others?
Basically: "Show me the computation."

But model_qc gives you a Python program.
``` python
A Python program can potentially do almost anything.

For example:

def forward(self, x):

    if condition:
        ...
    
    for layer in layers:
        ...
    
    if another_condition:
        ...

The compiler doesn't want to execute arbitrary Python semantics on the NPU.

It wants a representation of the tensor computation.
```

## 7. A tiny example makes this very clear
``` python
Consider:

def forward(self, x):

    if x.shape[1] == 128:
        return self.layer1(x)
    else:
        return self.layer2(x)

Now suppose:

x.shape = [1, 128]

The actual execution is:

x
 ↓
layer1
 ↓
output

But if:

x.shape = [1, 256]

then:

x
 ↓
layer2
 ↓
output

The Python program describes multiple possible computations.

The particular execution chooses one.

A graph capture process can take a concrete execution and say:

"For this concrete case, this is the computation that happened."

Then we get:

x
 ↓
layer1
 ↓
output

as the captured graph.
```
## 8. This is why concrete example inputs matter

This is one of the most important ideas in Model Preparation.

Conceptually:

             model_qc
                +
        concrete example inputs
                │
                ▼
          execute the model
                │
                ▼
        observe/capture ops
                │
                ▼
       computation graph

The example inputs are not merely there because the API requires something.

They help define which concrete computation is being captured.

They also establish concrete things like:

tensor shapes
input/output structure
cache tensor shapes
mask shapes
execution path

This becomes particularly important for Gemma4.

9. Now look at your Gemma4 model

Your adapted model still conceptually has something like:

for i, decoder_layer in enumerate(self.layers[:num_layers_to_run]):

    if self.config.layer_types[i] == "sliding_attention":
        ...
    else:
        ...

    hidden_states = decoder_layer(...)

This is completely fine as a PyTorch program.

But let's look at it from the compiler's point of view.

It sees a program saying:

For each layer:
    inspect something
    choose a path
    execute operations

The graph we eventually want is more like:

Layer 0 operations
       ↓
Layer 1 operations
       ↓
Layer 2 operations
       ↓
...
Layer 34 operations

The loop doesn't need to exist anymore.

10. What happens to the 35-layer loop?

This is a beautiful example of the difference.

In model_qc

You have:

for i in range(35):
    ...

Conceptually one piece of Python code describes all 35 layers.

In the captured graph

The operations for the concrete execution are explicitly represented:

Layer 0:
  RMSNorm
  Q projection
  K projection
  V projection
  RoPE
  attention
  output projection
  residual
  ...

Layer 1:
  RMSNorm
  Q projection
  ...
  
Layer 2:
  ...
  
...

Layer 34:
  ...

So the graph contains the actual operation sequence.

The Python loop has effectively been unrolled during capture.

11. What happens to Gemma4's if?

Remember your attention types.

Conceptually:

if layer_type == "sliding_attention":
    use sliding mask
else:
    use full attention mask

Gemma4's configuration already tells us the layer type.

So during the concrete execution:

Layer 0 → sliding
Layer 1 → sliding
...

The graph doesn't need a runtime Python decision.

Instead, the corresponding operations for that path are captured.

So:

Python decision
      ↓
resolved during capture
      ↓
concrete graph

This is why adaptation and preparation work together.

12. This is where adaptation becomes much clearer

Before adaptation:

HF model

may contain behavior that isn't ideal for graph capture.

Adaptation says:

"Let's rewrite the model so that the computation we want can be represented properly."

Then we get:

model_qc

But we haven't captured it yet.

So:

Adaptation
=
change the program

Preparation
=
capture the computation represented by that program

That is the distinction I really want you to remember.

13. Static-graph-friendly does NOT mean static graph

This is an important terminology point.

When we say:

"model_qc is static-graph-friendly"

we do not mean:

"model_qc is already a static graph."

It means:

"This PyTorch program has been written in a way that makes it suitable for producing the desired static computation graph."

So:

model_qc
   ↓
static-graph-friendly program

Then:

prepare_model()
   ↓
actual graph capture/conversion
14. Why not simply execute Python on the NPU?

Because the NPU isn't a general Python execution environment.

You can't send:

for layer in self.layers:
    if something:
        ...

to an NPU and expect it to execute Python.

The NPU needs something much more concrete:

operation
operation
operation
operation
...

with tensor information and dependencies.

So there is a fundamental boundary:

Python world
──────────────
model_qc
Python
nn.Module
forward()
objects
control flow


        ↓


Graph world
──────────────
nodes
tensors
dependencies
shapes
dtypes
inputs
outputs

Model Preparation crosses this boundary.

15. What does the compiler gain from having a graph?

This is the really useful part.

Once the computation is explicit, the compiler can reason about the entire computation.

For example:

MatMul
  ↓
Add
  ↓
ReLU

It can potentially ask:

Can these operations be fused?

Or:

Tensor A
   ↓
Operation
   ↓
Tensor B

It can ask:

How should these tensors be laid out in memory?

Or:

Which implementation of this operation should I use?

Or:

Can I schedule these operations efficiently?

Or:

Can this operation execute on the NPU?

Those questions require an explicit representation of the computation.

16. This is the real reason for preparation

So when you see:

prepare_model(...)

don't think:

"This prepares the PyTorch model."

Think:

"This takes the target-friendly PyTorch program and turns its concrete computation into a representation that the graph/compiler toolchain can reason about."

That is a much better mental model.

17. Now let's place model_mpp

At the end of preparation, you get:

model_mpp

Don't worry yet about why it is PyTorch again. We'll dedicate a later lesson to that.

For now:

model_qc
   ↓
   preparation
   ↓
graph / IR
   ↓
generated prepared model
   ↓
model_mpp

The important thing is that model_mpp represents a frozen/prepared computation, rather than being another hand-written dynamic architecture.

18. What has changed?

Let's make the transformation very explicit.

Before preparation
model_qc

Python program
    │
    ├── modules
    ├── loops
    ├── branches
    ├── config
    ├── tensor operations
    ├── weights
    └── runtime behavior
After preparation

Conceptually:

prepared graph

Input
  ↓
Op
  ↓
Op
  ↓
Op
  ↓
...
  ↓
Output

The important change is:

The computation is now explicitly represented instead of being hidden inside arbitrary Python execution.

19. One subtle correction to our mental model

I don't want you to remember this too simplistically as:

PyTorch
 ↓
graph

because that's not quite precise.

The actual process is closer to:

PyTorch program
      +
concrete inputs
      ↓
execute/capture a particular computation
      ↓
graph representation

The execution matters.

The graph isn't created merely by looking at the Python source code.

The capture process observes/records the tensor computation that actually occurs.

This is why dynamic Python behavior that isn't executed for that example may not appear in the captured graph.

20. Now the entire Model Preparation motivation

We can finally see the reason for the stage:

HF architecture
      ↓
Adaptation
      ↓
model_qc

At this point:

"We have a target-friendly model program."

But the compiler needs:

"Tell me exactly what computation I am compiling."

Therefore:

model_qc
      +
concrete inputs
      ↓
Model Preparation
      ↓
explicit graph representation

And that graph can then be:

optimized
represented in QAIRT IR
emitted as model_mpp

We'll unpack those steps one by one.

21. The 5 questions for this lesson

Let's apply the framework we agreed to use.

1. What representation are we in?
model_qc
=
PyTorch nn.Module / Python program
2. What information does it contain?
weights
modules
forward logic
tensor operations
configuration
Python control flow
3. What is still dynamic?

Potentially:

loops
branches
runtime Python behavior
object-level abstractions
4. Why does the next stage need another representation?

Because the compiler needs an explicit computation graph rather than arbitrary Python behavior.

5. What changes?
Executable PyTorch program
            ↓
Concrete computation
            ↓
Explicit graph representation
22. The most important distinction from today's lesson

There are three things. Keep them separate:

1. MODEL PROGRAM

model_qc

"Here is code that knows how to execute the model."


2. GRAPH

ONNX / QAIRT IR

"Here is an explicit representation of the computation."


3. EXECUTABLE

Later stages

"Here is something the target hardware can actually execute."

We are currently learning how we move from #1 → #2.

We are not compiling to hardware yet.

23. Your mental model

If you remember only one picture from this lesson, remember this:

              model_qc
                  │
                  │
        "How the model is written"
                  │
                  ▼
        ┌───────────────────┐
        │ Concrete execution│
        │ with example input│
        └───────────────────┘
                  │
                  │ capture
                  ▼
        ┌───────────────────┐
        │ Computation Graph │
        │                   │
        │ ops               │
        │ tensors           │
        │ dependencies      │
        │ inputs / outputs  │
        └───────────────────┘

And the sentence I want you to be able to say naturally is:

model_qc is a target-friendly PyTorch program, not yet a static computation graph. Model Preparation captures its concrete tensor computation and converts it into explicit graph representations that the compiler/toolchain can reason about.

That's Lesson 1.

Next, we should go one level deeper into the first actual mechanism:

Lesson 2 — What exactly happens when torch.onnx.export(model_qc, example_inputs, ...) runs?
