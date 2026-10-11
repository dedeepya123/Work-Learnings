# 05 - Neural Audio Codecs: Beyond Mel Spectrograms

## Recap

So far, we have built the following story:

```text
Sound
 ↓
Waveform
 ↓
Spectrogram
 ↓
Mel Spectrogram
```

and then:

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

This became the foundation of many successful TTS systems.

However, researchers eventually asked a fundamental question:

> Why are we manually designing the audio representation?

After all, in deep learning we often let the model learn the representation itself.

For example:

```text
Words
 ↓
Embeddings
```

were originally handcrafted features in older NLP systems.

Modern neural networks learn useful representations automatically.

Could we do the same for audio?

---

## The Core Problem

Consider a mel spectrogram.

It is useful.

It works well.

However, humans designed it.

We decided:

```text
How to convert audio into frequencies
How to compress frequencies
How many mel bins to use
How to organize information
```

The representation is fixed before training begins.

This raises an interesting question:

> What if a neural network could discover a better representation directly from the data?

Instead of manually engineering the representation, we allow the model to learn one.

This idea leads to neural audio codecs.

---

## First, What Is a Codec?

Most people have already used codecs without realizing it.

Examples:

```text
MP3
AAC
FLAC
```

A codec generally performs two operations:

```text
Compression
```

and

```text
Reconstruction
```

Conceptually:

```text
Audio
 ↓
Encoder
 ↓
Compressed Representation
 ↓
Decoder
 ↓
Audio
```

The goal is:

> Store less information while preserving as much quality as possible.

---

## Thinking Like Deep Learning Engineers

Imagine compression from a machine learning perspective.

Instead of designing compression rules ourselves:

```text
If X happens
Store Y
```

we let a neural network learn:

```text
Audio
 ↓
Encoder
 ↓
Latent Representation
 ↓
Decoder
 ↓
Audio
```

This should look familiar.

It is very similar to an autoencoder.

---

## A Reminder: Autoencoders

You may already know this architecture:

```text
Input
 ↓
Encoder
 ↓
Latent
 ↓
Decoder
 ↓
Reconstructed Input
```

The latent representation is smaller than the original input.

Because of that constraint, the model learns to keep only the most important information.

Neural audio codecs use the same intuition.

---

## The Big Idea

Instead of manually creating:

```text
Mel Spectrograms
```

we learn a representation directly.

```text
Audio
 ↓
Neural Encoder
 ↓
Learned Representation
 ↓
Neural Decoder
 ↓
Audio
```

The learned representation becomes:

```text
Compact
Expressive
Data Driven
```

This representation is often called:

```text
Audio Latents
```

or

```text
Codec Tokens
```

depending on the architecture.

---

## Why Is This Powerful?

Think about image generation.

Modern image systems rarely generate pixels directly.

Instead:

```text
Image
 ↓
Encoder
 ↓
Latents
 ↓
Diffusion Model
 ↓
Latents
 ↓
Decoder
 ↓
Image
```

The model works in a compressed space.

Generation becomes easier.

Audio systems discovered the same idea.

Rather than generating:

```text
Hundreds of thousands
of waveform samples
```

the model generates:

```text
A compact learned representation
```

which can later be decoded into audio.

---

## Enter Neural Audio Codecs

This idea led to architectures such as:

```text
SoundStream
```

and

```text
EnCodec
```

Conceptually:

```text
Audio
 ↓
Codec Encoder
 ↓
Learned Audio Representation
 ↓
Codec Decoder
 ↓
Audio
```

Notice the difference from mel spectrogram pipelines.

---

### Traditional Pipeline

```text
Text
 ↓
Mel Spectrogram
 ↓
Vocoder
 ↓
Waveform
```

---

### Codec-Based Pipeline

```text
Text
 ↓
Audio Latents
 ↓
Codec Decoder
 ↓
Waveform
```

The intermediate representation is now learned rather than handcrafted.

---

## Understanding SoundStream

For our future VoiceDM discussions, SoundStream is especially important.

At a high level:

```text
Audio
 ↓
SoundStream Encoder
 ↓
Audio Latents
 ↓
SoundStream Decoder
 ↓
Audio
```

The encoder learns:

```text
How to compress speech
```

The decoder learns:

```text
How to reconstruct speech
```

Together they create a learned audio representation.

---

## Why Modern Models Like This Approach

Imagine two choices.

### Option 1

Generate:

```text
240,000 waveform samples
```

for a ten-second recording.

---

### Option 2

Generate:

```text
A few hundred latent vectors
```

and let a decoder reconstruct the waveform.

Which sounds easier?

Almost always:

```text
Option 2
```

This is why learned representations became so attractive.

The generation problem becomes much smaller.

---

## Connecting This to VoiceDM

Now we can start recognizing some pieces of modern audio systems.

Instead of:

```text
Text
 ↓
Mel Spectrogram
 ↓
Vocoder
 ↓
Audio
```

we often see:

```text
Text
 ↓
Transformer / DiT
 ↓
Audio Features
 ↓
SoundStream Decoder
 ↓
Audio
```

The model is no longer generating a mel spectrogram.

It is generating a learned audio representation.

That representation is then decoded into speech.

---

## A Very Important Mental Shift

Historically, engineers thought:

```text
Speech
 ↓
Mel Spectrogram
 ↓
Audio
```

Modern systems increasingly think:

```text
Speech
 ↓
Learned Representation
 ↓
Audio
```

The representation is discovered during training rather than manually designed beforehand.

This is one of the biggest conceptual shifts in modern audio modeling.

---

## What Are Audio Latents?

The learned representation produced by the encoder is often called a latent.

Think of it as:

```text
Compressed Audio Knowledge
```

The latent is not directly audible.

Humans cannot listen to it.

However, it contains enough information for the decoder to reconstruct speech.

Conceptually:

```text
Audio
 ↓
Encoder
 ↓
Latents
 ↓
Decoder
 ↓
Audio
```

This idea should feel very similar to:

```text
Text Tokens
 ↓
Embeddings
 ↓
Hidden States
```

in language models.

---

## Mental Model

At this point our story becomes:

```text
Waveform
 ↓
Spectrogram
 ↓
Mel Spectrogram
```

and then:

```text
Handcrafted Representations
 ↓
Mel Spectrograms
```

eventually evolved into:

```text
Learned Representations
 ↓
Audio Latents
 ↓
Codec Models
```

Modern systems increasingly prefer learned representations because they allow the model to discover compact and useful ways of representing speech.

---

## Key Takeaways

- Mel spectrograms are handcrafted audio representations.
- Researchers began exploring learned audio representations.
- A codec compresses audio and later reconstructs it.
- Neural audio codecs use encoder-decoder architectures.
- SoundStream and EnCodec are examples of neural audio codecs.
- Neural codecs learn compact audio representations from data.
- Modern generative audio systems often generate audio latents instead of generating waveforms directly.
- Codec-based representations reduced the difficulty of audio generation.

---

## Interview Takeaway

A neural audio codec is an encoder-decoder system that learns a compact audio representation directly from data. Instead of relying on handcrafted representations such as mel spectrograms, the codec learns a latent representation that preserves important speech information while remaining easier to model. Architectures such as SoundStream and EnCodec are widely used because modern generative models can generate these learned representations and then use a codec decoder to reconstruct high-quality speech.

---

## Next Topic

# 06 - Audio Latents and Codec Tokens: What Exactly Is the Model Generating?

We now understand that modern systems often generate learned audio representations instead of mel spectrograms.

The next question is:

> What exactly are these representations?

Are they vectors?
Are they tokens?
Are they embeddings?
Are they compressed audio?

To answer that, we need to understand audio latents and codec tokens, the building blocks that connect neural codecs to modern audio generation systems such as VoiceDM.
