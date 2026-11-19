# Third-party code

The single-row Metal kernel uses the dot-product helpers from MTPLX 2.11.3 by Youssof Altoukhi. The helpers are in `src/weightpack/metal/q4_dot.h`. MTPLX uses the Apache License 2.0. Its copyright notice and license ship inside the wheel. [Upstream project](https://github.com/youssofal/MTPLX) · [License](src/weightpack/notices/MTPLX-LICENSE.txt) · [Notice](src/weightpack/notices/MTPLX-NOTICE.txt)

The 8- and 16-row kernels use the Steel matrix helpers installed with MLX 0.32.2. MLX is copyright Apple Inc. and uses the MIT License. [MLX source](https://github.com/ml-explore/mlx/tree/v0.32.2) · [License](src/weightpack/notices/MLX-LICENSE.txt)

Model weights and tokenizers come from their respective publishers. The benchmarks identify the exact checkpoints used.
