# 01 - Waveforms: How Do Audio Models See Sound?

## Why Are We Learning This?

We already know that neural networks operate on tensors.

Whether we are building image models, language models, or audio models, the model never sees the real-world object directly. It only sees a numerical representation of that object.

For example, in an LLM:

```text
Text
 ↓
Tokens
 ↓
Embeddings
 ↓
Transformer
```

Although we think of the input as a sentence, the model never processes the sentence itself. It processes tensors representing that sentence.

The same idea applies to audio.

Before discussing spectrograms, codecs, diffusion models, or VoiceDM, we first need to answer a more fundamental question:

> How is sound represented inside a machine learning system?

Understanding this representation is important because the form of the input determines what information is available to the model and what patterns the model must learn.

---

## Sound in the Physical World

When a person speaks, sound travels through the air as a continuous physical wave.

Imagine saying:

```text
"Hello"
```

This creates vibrations in the air.

As those vibrations travel, the air pressure continuously changes over time.

Conceptually:

```text
Air Pressure

      /\        /\
     /  \      /  \
    /    \    /    \
___/      \__/      \____

                Time →
```

The sound exists as a continuous signal.

However, computers cannot directly store or process a continuous signal.

We must convert it into numbers.

---

## Sampling the Signal

To store sound digitally, we measure the signal at regular intervals.

This process is called **sampling**.

Instead of storing the entire continuous wave, we record its value repeatedly:

```text
Time 1 → 0.12
Time 2 → 0.15
Time 3 → 0.08
Time 4 → -0.04
Time 5 → -0.12
...
```

The result becomes a sequence of numbers:

```text
[0.12, 0.15, 0.08, -0.04, -0.12, ...]
```

Each number represents the strength of the signal at a specific instant in time.

This sequence is called a **waveform**.

---

## What Is a Waveform?

A waveform is the most basic digital representation of audio.

We can think of it as:

```text
Waveform = Audio Samples Over Time
```

For machine learning purposes:

```python
waveform = [sample_1, sample_2, sample_3, ...]
```

or

```python
waveform.shape = [num_samples]
```

For a batch:

```python
waveform.shape = [batch_size, num_samples]
```

The waveform contains all information needed to reconstruct the original sound.

Everything else in audio processing ultimately starts from the waveform.

---

## Sampling Rate

A waveform consists of samples.

The next question is:

> How many samples do we record every second?

This is determined by the **sampling rate**.

Examples:

```text
16 kHz = 16,000 samples per second
24 kHz = 24,000 samples per second
48 kHz = 48,000 samples per second
```

Suppose we record three seconds of speech at 16 kHz.

We obtain:

```text
3 × 16,000

=
48,000 samples
```

The resulting tensor might look like:

```python
waveform.shape = [48000]
```

For a batch of recordings:

```python
waveform.shape = [B, 48000]
```

Notice how quickly audio grows.

Even a short recording can contain tens of thousands of values.

---

## Connecting This to Deep Learning

Let's compare audio with text.

For text:

```text
"I love AI"
```

we may have:

```text
[101, 512, 278]
```

For audio:

```text
"Hello"
```

we may have:

```text
[0.12, 0.15, 0.08, -0.04, ...]
```

There is an important difference.

Text tokens already contain meaningful units:

```text
Words
Subwords
Characters
```

Waveforms do not.

A waveform contains only signal measurements.

The waveform does not explicitly tell us:

```text
What word was spoken
Who spoke it
What emotion was expressed
What language was used
```

The model must learn those higher-level patterns from the signal itself.

---

## Waveforms and Pixels

A useful analogy comes from computer vision.

Images are represented using pixels:

```python
image.shape = [H, W, C]
```

Pixels do not explicitly tell us:

```text
Face
Car
Tree
Dog
```

They only provide low-level information.

The neural network learns higher-level patterns.

Audio is similar.

```python
audio.shape = [Samples]
```

The waveform does not explicitly tell us:

```text
Phoneme
Word
Speaker
Emotion
```

Those patterns must be learned from the raw signal.

You can think of waveforms as the audio equivalent of pixels.

```text
Images
 ↓
Pixels
 ↓
Neural Network
```

```text
Audio
 ↓
Waveform Samples
 ↓
Neural Network
```

---

## Why Not Just Use Waveforms?

A waveform is a valid representation.

In fact, some modern architectures operate directly on raw audio.

However, a waveform mostly tells us:

```text
How the signal changes over time
```

Many acoustic patterns that humans naturally perceive are not immediately obvious from the waveform itself.

For example:

```text
Pitch
Tone
Frequency Structure
Speech Characteristics
```

all exist within the waveform, but they are not explicitly exposed.

As a result, researchers started looking for representations that make these patterns easier to observe.

This led to one of the most important ideas in audio processing:

```text
Waveform
 ↓
Spectrogram
```

A spectrogram reveals how frequencies change over time and often provides a more useful representation for speech-related tasks.

---

## Mental Model

Keep the following comparison in mind.

Language Models:

```text
Sentence
 ↓
Tokens
 ↓
Embeddings
 ↓
Transformer
```

Audio Models:

```text
Sound
 ↓
Waveform
 ↓
Audio Representation
 ↓
Neural Network
```

The representation is different, but the idea is exactly the same.

A model never receives the real-world object directly.

It receives a numerical representation of that object.

---

## Key Takeaways

- Neural networks operate on numerical representations rather than real-world objects.
- A waveform is the raw digital representation of audio.
- A waveform is a sequence of sampled signal values over time.
- The sampling rate determines how many samples are collected each second.
- Raw waveforms contain signal information but do not explicitly encode words, speakers, or emotions.
- Waveforms play a role similar to pixels in computer vision.
- Many audio systems transform waveforms into alternative representations before feeding them into a model.

---

## Interview Takeaway

A waveform is the most fundamental representation of audio in machine learning systems.

It consists of sampled signal amplitudes recorded over time. Similar to how images are represented by pixels, audio is represented by waveform samples. While a waveform contains all information needed to reconstruct the original sound, it does not explicitly expose higher-level speech patterns such as words, speakers, or emotions. Therefore, many audio models transform waveforms into more informative representations before processing them.

---

## Next Topic

# 02 - Frequency and Spectrograms: Why Transform a Waveform?

We know that a waveform contains all the information in a sound signal.

The next question is:

> If a waveform already contains everything, why do audio engineers transform it into a spectrogram?

To answer that, we first need an intuition for frequency and what it tells us about sound.
