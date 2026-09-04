``` text
Adapted HF model
      │
      │ model_qc
      ▼
┌─────────────────────┐
│  MODEL PREPARATION  │
│                     │
│ PyTorch → ONNX      │
│ ONNX → QAIRT IR     │
│ IR → optimized IR   │
│ optimized IR → MPP  │
└──────────┬──────────┘
           ▼
       model_mpp
```

<img width="1161" height="456" alt="image" src="https://github.com/user-attachments/assets/2721600c-494e-4107-9f28-5d9826897096" />
