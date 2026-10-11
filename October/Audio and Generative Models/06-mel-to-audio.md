# 04 - From Mel Spectrograms to Audio: What Is a Vocoder?

## Recap

In the previous lessons, we learned the progression:

```text
Sound
 ↓
Waveform
 ↓
Spectrogram
 ↓
Mel Spectrogram
```

We also learned that many historical speech systems did not generate audio directly.

Instead, they generated a mel spectrogram:

```text
Text
 ↓
Model
 ↓
Mel Spectrogram
```

At first glance, this seems like the problem is solved.

The model successfully converted text into a speech representation.

However, there is one major issue:

> A mel spectrogram is not sound.

You cannot play a mel spectrogram through a speaker.

Humans hear waveforms, not mel spectrograms.

So we need one more component.

---

## The Missing Piece

Suppose a TTS model predicts:

```text
Mel Spectrogram
```

Our output now looks something like:

```text
Frequency
 ^
 |
 |
 |
 +--------------> Time
```

This representation contains useful speech information, but it is still only a description of sound.

To actually hear speech, we need to construct a waveform:

```text
[0.12, 0.15, -0.08, ...]
```

This conversion step is the job of a **vocoder**.

---

## What Is a Vocoder?

At a high level:

```text
Vocoder

Input:
Mel Spectrogram

Output:
Waveform Audio
```

Conceptually:

```text
Mel Spectrogram
        ↓
     Vocoder
        ↓
    Waveform
```

You can think of the vocoder as a specialized decoder whose responsibility is:

> Turn an audio representation into actual sound.

---

## A Useful Analogy

Let's compare this to image generation.

Imagine a model produces:

```text
Image Latents
```

Those latents are not directly viewable.

We need a decoder.

```text
Latents
 ↓
Decoder
 ↓
Image
```

Similarly:

```text
Mel Spectrogram
 ↓
Vocoder
 ↓
Audio
```

The vocoder acts as the final conversion stage.

---

## Why Not Generate Waveforms Directly?

This was exactly the question researchers asked.

Why build:

```text
Text
 ↓
Mel Spectrogram
 ↓
Vocoder
 ↓
Waveform
```

instead of:

```text
Text
 ↓
Waveform
```

The answer is difficulty.

Generating:

```text
Hundreds of mel frames
```

is often easier than generating:

```text
Hundreds of thousands of audio samples
```

directly.

So engineers divide the problem into two stages:

### Stage 1

Learn:

```text
Text
 ↓
Mel Spectrogram
```

### Stage 2

Learn:

```text
Mel Spectrogram
 ↓
Waveform
```

Each stage becomes easier to solve.

---

## An Engineering Perspective

Think about a manufacturing pipeline.

Instead of building the final product in one giant step:

```text
Raw Material
 ↓
Finished Product
```

we split the process:

```text
Raw Material
 ↓
Intermediate Representation
 ↓
Finished Product
```

Deep learning systems often use the same idea.

The intermediate representation for many speech systems became:

```text
Mel Spectrogram
```

and the vocoder became the component responsible for the final conversion.

---

## Early Vocoders

Historically, vocoders relied heavily on signal processing techniques.

Engineers manually designed algorithms to reconstruct speech.

These systems worked, but quality was limited.

Researchers then started replacing handcrafted methods with neural networks.

---

## Neural Vocoders

A neural vocoder learns:

```text
Mel Spectrogram
        ↓
Waveform
```

using data.

Instead of manually specifying all reconstruction rules, we let the model learn them.

Examples include:

```text
WaveNet
WaveRNN
MelGAN
HiFi-GAN
BigVGAN
```

Although the architectures differ, their core purpose remains the same:

```text
Speech Representation
          ↓
       Vocoder
          ↓
      Waveform
```

---

## Why Were Neural Vocoders Important?

Once neural vocoders became good enough, speech quality improved dramatically.

Engineers could focus their main model on producing a good mel spectrogram.

The vocoder would handle reconstruction.

Conceptually:

```text
Text
 ↓
Acoustic Model
 ↓
Mel Spectrogram
 ↓
Neural Vocoder
 ↓
Speech
```

This design became dominant for many years.

---

## Connecting This to What We Already Know

Think about representations.

So far we have learned:

```text
Waveform
```

is one representation.

```text
Spectrogram
```

is another.

```text
Mel Spectrogram
```

is another.

A vocoder exists because different representations serve different purposes.

The model may prefer working with:

```text
Mel Spectrograms
```

while the user ultimately needs:

```text
Waveform Audio
```

The vocoder bridges that gap.

---

## The Limitation of Mel-Based Pipelines

Although this architecture worked extremely well, researchers encountered an important question:

> Are mel spectrograms the best intermediate representation?

A mel spectrogram is still hand-designed.

Humans decided:

```text
How frequencies are grouped
How information is compressed
How the representation is constructed
```

This inspired a new idea.

Instead of manually designing the representation:

```text
Can a neural network learn its own audio representation?
```

This question led to:

```text
Audio Autoencoders
Neural Codecs
SoundStream
EnCodec
Codec Tokens
Audio Latents
```

These representations are at the heart of many modern audio systems, including architectures related to VoiceDM.

---

## Mental Model

Keep this pipeline in mind:

```text
Text
 ↓
Speech Model
 ↓
Mel Spectrogram
 ↓
Vocoder
 ↓
Waveform
```

The speech model focuses on generating a useful representation.

The vocoder focuses on reconstructing audio.

Together they produce speech.

---

## Key Takeaways

- A mel spectrogram is not audible sound.
- A waveform is required for playback.
- A vocoder converts an intermediate audio representation into waveform audio.
- Many historical TTS systems generated mel spectrograms instead of generating waveforms directly.
- Neural vocoders significantly improved speech quality.
- The vocoder acts as a decoder between an audio representation and audible speech.
- Modern systems started exploring learned audio representations instead of relying solely on mel spectrograms.

---

## Interview Takeaway

A vocoder is a component that converts an intermediate speech representation, commonly a mel spectrogram, into a waveform. In traditional TTS pipelines, the speech model predicts the mel spectrogram while the vocoder reconstructs the final audio. This decomposition simplifies the generation problem and allows specialized models to focus on different stages of the speech synthesis process.

---

## Next Topic

# 05 - Neural Audio Codecs: Beyond Mel Spectrograms

We now understand the classical pipeline:

```text
Text
 ↓
Mel Spectrogram
 ↓
Vocoder
 ↓
Waveform
```

The next question is:

> What if we stop hand-designing audio representations altogether and allow a neural network to learn the representation itself?

This idea leads us to neural codecs such as SoundStream and EnCodec, which form the foundation of many modern speech and audio models.
