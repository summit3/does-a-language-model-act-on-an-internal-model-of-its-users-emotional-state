# Run M. Sampled subtraction on the 30 distressed val prompts (T=0.7, 5 samples each, 150 per fraction), rules v3

| metric | fraction 0 (unsteered) | fraction -0.04 along distressed_md |
|---|---|---|
| acknowledges | 0.267 [0.202, 0.343] (n=150) | 0.000 [0.000, 0.025] (n=150) |
| infers user state | 0.133 [0.088, 0.197] (n=150) | 0.000 [0.000, 0.025] (n=150) |
| abandons task | 0.053 [0.027, 0.102] (n=150) | 0.000 [0.000, 0.025] (n=150) |
| correct | 0.912 [0.849, 0.950] (n=125) | 0.912 [0.849, 0.950] (n=125) |

Answer: subtracting the direction removes acknowledgement of a distressed preamble (0.267 -> 0.000, non-overlapping CIs), with inference and abandonment also at 0 and correctness unchanged; 2/150 reply texts identical across conditions, incoherence 0.007 at -0.04.
