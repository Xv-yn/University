# ELISA

### Direct ELISA

```txt
       [Substrate]
       [Antibody]
       [Antigen]
-------------------------
    [Microplate Well]
```

1. **Coating**: Coat the microplate well with a sample (containing Antigen and others).
   - **Simple Terms**: Coat with the molecule that we want to detect.
2. **Blocking**: Add a non-reactive protein to prevent unspecific binding.
   - **Analogy**: There are many colors the antigen can bind to; blocking ensures that
     the antigen con only bind to the color blue.
3. Add the specified Antibody to bind with the Antigen.
4. Add a substrate to bind to the Antibody to produce a color change.
5. Measure color change via Spectrophotometer.

This allows us to measure antigen concentration in a sample.

### Indirect ELISA

```txt
       [Substrate]
       [Secondary Antibody]
       [Primary Antibody]
       [Antigen]
-------------------------
    [Microplate Well]
```

1. **Coating**
2. **Blocking**
3. Add the Primary Antibody from the sample (what is being measured)
4. Add the Secondary Antibody (allows the substrate to bind)
5. Add the substrate
6. Measure color change via Spectrophotometer.

This allows us to measure antibody concentration in a sample.

### Sandwich ELISA

```txt
       [Substrate]
       [Detection Antibody]
       [Antigen]
       [Specified Antibody]
-------------------------
    [Microplate Well]
```

1. **Coating** but with an Specified Antibody instead of an Antigen.
2. **Blocking**
3. Add the sample, allowing for antigens to bind with the Specified Antibody.
4. Add the detection antibody that binds to the antigen.
5. Add the substrate.
6. Measure color change via Spectrophotometer.

This allows us to measure antigen concentration in a sample.

## Reading Results (Westgard Rules)

```txt
  ^
  |
  |............................................. +3 SD
  |
  |
  |............................................. +2 SD
  |
  |
  |............................................. +1 SD
  |
  |
  |............................................. Mean
  |
  |
  |............................................. -1 SD
  |
  |
  |............................................. -2 SD
  |
  |
  |............................................. -3 SD
  |
  |
  +-------------------------------------------->
```

| Rule             | Notation                   | Description                                                                 | Indicates: random or systematic error                            |
| ---------------- | -------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **Warning rule** | 1 2s (read: one‐two‐sigma) | A single control result exceeds ±2 SD from the mean.                        | A warning—**not** automatic rejection; further review needed.    |
| **Reject rules** | 1 3s                       | A single result exceeds ±3 SD from the mean.                                | Likely large systematic error (bias) or very large random error. |
|                  | 2 2s                       | Two consecutive results exceed ±2 SD on the same side of the mean.          | Likely systematic error (shift) starting.                        |
|                  | 4 1s                       | Four consecutive results exceed ±1 SD on the same side of the mean.         | Systematic bias/trend becoming apparent.                         |
|                  | 10 x                       | Ten consecutive results fall on the same side of the mean (above or below). | Very strong indication of systematic shift.                      |
