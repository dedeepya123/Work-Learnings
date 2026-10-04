Problem Statement:
  We have 2D RopE implementation where we are splitting the intermediate tenor into 4 splits and doing cos, sin some opertaions
  <img width="506" height="284" alt="image" src="https://github.com/user-attachments/assets/1cb69d37-ea13-4911-ac01-c90a2af93126" />
But in the memory layout crouton format that tensor 16 in last dim is not using it fully causing perf siiue so what mahematically can we change so crouton can be fully utilized?
