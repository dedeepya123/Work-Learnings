# How I Want to Approach My Current Work

## The mindset

I don't need to impress anyone.

I don't need to prove that I already know Gemma.

I don't need to compare myself with the teammates who have been working on this code for much longer.

I am new to this codebase, and that's okay.

**My job is simple:**

> Come in → understand → ask why → implement → debug → learn → document → repeat.

My current task may look like "just API adaptation," but I can use it as my entry point into:

**Gemma → Transformer internals → QAIRT → runtime abstractions → on-device inference → performance.**

I will not underestimate the task.

---

# 1. Understand Before Changing

Whenever I encounter existing code, I will first ask:

* What is this doing?
* Why is it doing this?
* What is the input?
* What is the output?
* What are the tensor shapes?
* What happens to the data here?
* Is this standard Gemma/Transformer behavior?
* Is this a QAIRT/runtime-specific adaptation?
* What would happen if I removed it?

Don't immediately replace code just because I know the new API.

**Understand the original behavior first.**

---

# 2. Understand the New QAIRT API

For every API I have to use:

* What does this API represent?
* Why does it exist?
* What does it expect as input?
* What does it return?
* What happens internally?
* Is it merely an abstraction?
* Does it change execution semantics?
* Does it affect performance?
* Why is this API preferred over the original implementation?

The goal isn't:

> "I know how to call the API."

The goal is:

> **"I understand why this API is the correct representation of the original operation."**

---

# 3. Create an Operation Mapping

For every adaptation, maintain something like:

| Original implementation | QAIRT API | Purpose | Why this mapping? | Verified? |
| ----------------------- | --------- | ------- | ----------------- | --------- |
| Operation A             | API A     | ...     | ...               | ✅         |
| Operation B             | API B     | ...     | ...               | ❓         |
| Operation C             | API C     | ...     | ...               | ❓         |

Initially, it is completely fine to write:

**Why? → UNKNOWN**

That UNKNOWN becomes a question to investigate.

---

# 4. Build My Gemma Architecture Map

I don't need to learn all of Gemma upfront.

I will build my understanding while working.

```text
Gemma
 |
 +-- Embedding
 |
 +-- Transformer Block
       |
       +-- Normalization
       |
       +-- Attention
       |     |
       |     +-- Q
       |     +-- K
       |     +-- V
       |     +-- RoPE
       |     +-- KV Cache
       |
       +-- MLP
       |
       +-- Residual connections
```

Whenever I encounter something unfamiliar, I add it to the map.

The architecture should gradually become something I can explain myself.

---

# 5. Treat Every "Weird" Adaptation as a Question

If I see:

* Dynamic RoPE
* Scatter
* Gather
* Reshape
* Transpose
* Custom layers
* KV-cache handling
* Tensor layout changes
* Runtime-specific operations

I should NOT think:

> "This is some weird code. I'll just copy it."

Instead:

> **"Why does this exist?"**

For example:

### Dynamic RoPE

Ask:

* What does normal RoPE do?
* Why is this implementation dynamic?
* Dynamic with respect to what?
* Sequence length?
* Position?
* Tensor shape?
* Runtime constraint?
* Performance?
* Is this model behavior or runtime adaptation?

---

# 6. Maintain `questions.md`

Whenever I don't understand something:

```text
# Questions

## Gemma

- Why is DynamicRoPE used here?
- Why is this tensor reshaped here?
- What is the Q/K/V layout?

## QAIRT

- What abstraction does this API provide?
- Why does this operation need to go through QAIRT?
- What happens internally?

## Adaptation

- Is this only an API change?
- Are there semantic differences?
- Could this affect performance?
```

Before asking someone:

**First try to answer it myself.**

Then ask a focused question.

Not:

> "Can you explain this whole thing?"

Instead:

> "I understand what this operation does, but I'm not clear why this adaptation is required. Is it because of a QAIRT runtime constraint or because of the model implementation?"

That makes the conversation much more useful.

---

# 7. Use This Learning Loop

For every unfamiliar component:

**READ**

↓

**UNDERSTAND**

↓

**ASK WHY**

↓

**FORM A HYPOTHESIS**

↓

**IMPLEMENT**

↓

**RUN**

↓

**DEBUG**

↓

**VALIDATE**

↓

**DOCUMENT**

↓

**MOVE TO THE NEXT THING**

This is how I will turn a work task into real engineering knowledge.

---

# 8. Don't Try to Learn Everything Before Starting

I should NOT do:

> "First I need to completely learn Gemma."

Then:

> "Then Transformer architecture."

Then:

> "Then QAIRT."

Then:

> "Then runtime internals."

Then finally start the task.

Instead:

**Work → encounter concept → learn concept → apply it → continue.**

If I encounter RoPE:

→ Learn RoPE deeply.

If I encounter KV cache:

→ Learn KV cache deeply.

If I encounter a QAIRT graph/API concept:

→ Learn that deeply.

This is **just-in-time learning**.

---

# 9. Communicate Progress

I don't need to constantly talk to my manager.

But I should make my work visible.

When I have meaningful progress:

> "I mapped X and Y to the new QAIRT APIs and verified the behavior. I'm currently working through Z. I haven't hit a blocker yet."

If I'm blocked:

> "I'm blocked on X. I understand the original implementation, but I'm not sure whether the QAIRT API expects Y or Z. Could you point me in the right direction?"

This isn't about impressing anyone.

It's simply good engineering communication.

---

# 10. Don't Compare My Knowledge With My Teammates

They have history.

They were there when:

* Gemma was adapted.
* Problems were discovered.
* Optimizations were designed.
* Bugs were fixed.
* Decisions were made.

I wasn't there.

Therefore:

> **Their familiarity is not evidence that I am less capable.**

They have accumulated context.

I am now accumulating it.

My goal isn't to catch up overnight.

My goal is to understand one more thing every day.

---

# 11. Don't Call My Work "Just API Adaptation"

I should never underestimate the task.

The valuable version of this work is:

> Understand the original model implementation → understand the runtime abstraction → map the two correctly → preserve behavior → validate → understand performance implications.

That is real systems engineering.

The API call itself isn't the valuable part.

**Understanding the mapping is.**

---

# 12. Think About Future Resume Value

I will not exaggerate what I did.

I will not claim:

> "I designed Gemma optimization."

if I didn't.

Instead, I will build genuine experience that I can later describe truthfully:

> Adapted Gemma model components to a common Pythonic QAIRT runtime API for on-device inference, analyzing existing model operations and mapping them to runtime abstractions while validating behavioral correctness.

If I eventually work on:

* Performance
* Memory
* Quantization
* Runtime behavior
* Debugging
* Correctness
* NPU execution
* Latency
* Operator mapping

then those become additional concrete engineering achievements.

**The resume bullet should come from the work I actually understand and did.**

---

# 13. My Daily Rule

At the end of each workday, I should be able to answer:

### What did I understand today?

### What did I implement?

### What confused me?

### What question did I answer?

### What question remains?

### What will I investigate tomorrow?

Even if the answer is small, that's okay.

One day:

> "I understood this API."

Another day:

> "I understood why this reshape exists."

Another:

> "I understood the RoPE adaptation."

Another:

> "I figured out why the QAIRT API needs this representation."

Over months, these small pieces become **deep system knowledge.**

---

# 14. The Most Important Rule

I am not here to prove:

> "I already know everything."

I am here to become:

> **"Someone who can take an unfamiliar ML system, understand it from first principles, ask the right questions, modify it safely, debug it, and explain why it works."**

That is the engineer I am trying to become.

So when I enter work:

**Don't fear.**

**Don't overthink what people think about me.**

**Don't compare.**

**Don't try to impress.**

Instead:

> **Understand the code.**

> **Ask why.**

> **Do the work.**

> **Learn deeply.**

> **Document.**

> **Keep going.**

And then go home.

I don't have to become excellent in one week.

**I just have to keep becoming better.**
