# Waveform vs Spectrogram: Representing Audio for a Model

When we speak, we produce a sound signal that changes over time. Different speech sounds have different acoustic characteristics, and these changes carry information about the speech.

To process audio digitally, we sample the sound signal at regular time intervals. Each sample records the signal's amplitude at a particular instant. The sequence of samples forms a **waveform**.

A waveform is therefore one way to represent audio digitally. It describes how the signal's amplitude changes over time. A neural network can process this representation directly.

However, the waveform contains raw signal measurements rather than an explicit description of the acoustic patterns present in speech. A model processing raw audio must learn useful patterns from those measurements.

We can also transform the waveform into a **spectrogram**. Instead of showing only the amplitude at each instant, a spectrogram shows how the signal's frequency content changes over time. This can make useful acoustic patterns easier to represent and learn.

The important distinction is that a spectrogram is not a different recording. It is another representation calculated from the same audio signal.

This is similar to a broader idea in deep learning: how we represent the input influences the patterns a model can learn from it. Different representations expose different properties of the same underlying data.

For audio models, the choice depends on the architecture and task. Some models process raw waveforms, while others use spectrograms, mel spectrograms or learned representations.

Understanding the input representation is important when implementing a model because it determines what the model receives, the shape of its input tensors, and the preprocessing required before inference.

The next step is to understand how a spectrogram is calculated. The Short-Time Fourier Transform (STFT) analyses short sections of the waveform to estimate how the frequency content changes over time.
