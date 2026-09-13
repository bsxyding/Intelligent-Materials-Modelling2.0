# LLM Prompt Templates — LOOCV Folds 1–10

## Overview

This file contains the **exact prompt template** used for all four LLMs
(DeepSeek-V3, DeepSeek-R1, ChatGPT-4o, GPT-5) under the leave-one-out
cross-validation (LOOCV) protocol described in Section 2.2.2 of the manuscript.

**Experimental design:**
- 10 folds × 5 independent runs × 4 LLMs = **200 prompt submissions** total.
- Each fold holds out one sample (target) and provides the remaining nine
  as reference data.
- The placeholder `LLM_Name` in the CSV schema was replaced with the actual
  model identifier (e.g., `DeepSeek-V3`) at execution time.
- The placeholder `run=1` was replaced with the actual run number (1–5).
- All sessions were isolated (no conversational carryover between folds or runs).
- Decoding parameters were held at default; external tools, retrieval, and
  code execution were disabled.

**Notation:**
- Structural descriptors: pore diameter (PD, µm), contact angle (CA, °),
  thickness (T, mm), porosity (P, %).
- Mechanical targets: Young's modulus (E, N/mm²), tensile strength (TS, N/mm²),
  elongation at break (EL, %).

---

# Fold 1 — Target = S1

```text
You are provided with detailed experimental data on known polysulfone (PSF) membrane samples. Leveraging your comprehensive knowledge and intuition from scientific literature and domain expertise in polymer physics, predict the missing mechanical properties for the target sample listed below based solely on the provided reference data.

### Instructions
- Carefully analyze the provided reference dataset (nine samples).
- Predict the three target mechanical properties clearly and directly, without referencing external tools, programming, or model training.
- Base your judgment on polymer membrane physics and the structure–property trends observable in the reference data.

### Validation protocol
- Treat this as leave-one-out cross-validation: use ONLY the nine rows in the reference table to infer structure–property trends.
- Predict the three mechanical properties for the target sample.
- Do not use external tools, code execution, or web retrieval.
- Keep units exactly as specified.
- Output requirement: reply with CSV only inside a single code block — no prose, no extra text.
- Include the provided meta fields in every CSV row: model_name=LLM_Name, run=1–5.
- Return exactly three rows (one per property) with this header:
  model_name,run,sample,property,units,predicted
- Round all predicted values to two decimal places.

### Reference PSF membrane data (use these nine rows only)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) | Young's modulus (N/mm²) | Tensile strength (N/mm²) | Elongation at break (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|------------------------:|-------------------------:|------------------------:|
| S2    | 0.364              | 93.3               | 0.202          | 79.16        | 90.18                   | 3.97                     | 46.61                  |
| S3    | 0.569              | 78.9               | 0.175          | 78.87        | 152.50                  | 6.56                     | 49.13                  |
| S4    | 0.451              | 66.4               | 0.136          | 73.60        | 231.78                  | 9.61                     | 61.30                  |
| S5    | 0.408              | 66.6               | 0.154          | 73.49        | 182.59                  | 7.43                     | 60.69                  |
| S6    | 0.336              | 79.5               | 0.177          | 69.09        | 235.65                  | 9.65                     | 64.46                  |
| S7    | 0.403              | 81.3               | 0.240          | 71.18        | 176.51                  | 7.25                     | 69.04                  |
| S8    | 0.319              | 84.5               | 0.180          | 78.32        | 126.42                  | 5.22                     | 42.91                  |
| S9    | 0.842              | 90.0               | 0.224          | 78.86        | 107.67                  | 4.53                     | 51.25                  |
| S10   | 0.298              | 82.9               | 0.188          | 74.02        | 168.57                  | 7.13                     | 67.15                  |

### Target PSF membrane (predict its mechanical properties)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|
| S1    | 0.522              | 94.6               | 0.273          | 77.67        |

### Expected CSV output
```csv
model_name,run,sample,property,units,predicted
LLM_Name,1,S1,Young's modulus,N/mm^2,<value>
LLM_Name,1,S1,Tensile strength,N/mm^2,<value>
LLM_Name,1,S1,Elongation at break,%,<value>
```

**Important:** Respond with the CSV code block only. Do not use any programming language, code execution, or external tools. Use your domain knowledge and the reference data to make the prediction.
```

---

# Fold 2 — Target = S2

```text
You are provided with detailed experimental data on known polysulfone (PSF) membrane samples. Leveraging your comprehensive knowledge and intuition from scientific literature and domain expertise in polymer physics, predict the missing mechanical properties for the target sample listed below based solely on the provided reference data.

### Instructions
- Carefully analyze the provided reference dataset (nine samples).
- Predict the three target mechanical properties clearly and directly, without referencing external tools, programming, or model training.
- Base your judgment on polymer membrane physics and the structure–property trends observable in the reference data.

### Validation protocol
- Treat this as leave-one-out cross-validation: use ONLY the nine rows in the reference table to infer structure–property trends.
- Predict the three mechanical properties for the target sample.
- Do not use external tools, code execution, or web retrieval.
- Keep units exactly as specified.
- Output requirement: reply with CSV only inside a single code block — no prose, no extra text.
- Include the provided meta fields in every CSV row: model_name=LLM_Name, run=1–5.
- Return exactly three rows (one per property) with this header:
  model_name,run,sample,property,units,predicted
- Round all predicted values to two decimal places.

### Reference PSF membrane data (use these nine rows only)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) | Young's modulus (N/mm²) | Tensile strength (N/mm²) | Elongation at break (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|------------------------:|-------------------------:|------------------------:|
| S1    | 0.522              | 94.6               | 0.273          | 77.67        | 117.17                  | 4.82                     | 42.07                  |
| S3    | 0.569              | 78.9               | 0.175          | 78.87        | 152.50                  | 6.56                     | 49.13                  |
| S4    | 0.451              | 66.4               | 0.136          | 73.60        | 231.78                  | 9.61                     | 61.30                  |
| S5    | 0.408              | 66.6               | 0.154          | 73.49        | 182.59                  | 7.43                     | 60.69                  |
| S6    | 0.336              | 79.5               | 0.177          | 69.09        | 235.65                  | 9.65                     | 64.46                  |
| S7    | 0.403              | 81.3               | 0.240          | 71.18        | 176.51                  | 7.25                     | 69.04                  |
| S8    | 0.319              | 84.5               | 0.180          | 78.32        | 126.42                  | 5.22                     | 42.91                  |
| S9    | 0.842              | 90.0               | 0.224          | 78.86        | 107.67                  | 4.53                     | 51.25                  |
| S10   | 0.298              | 82.9               | 0.188          | 74.02        | 168.57                  | 7.13                     | 67.15                  |

### Target PSF membrane (predict its mechanical properties)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|
| S2    | 0.364              | 93.3               | 0.202          | 79.16        |

### Expected CSV output
```csv
model_name,run,sample,property,units,predicted
LLM_Name,1,S2,Young's modulus,N/mm^2,<value>
LLM_Name,1,S2,Tensile strength,N/mm^2,<value>
LLM_Name,1,S2,Elongation at break,%,<value>
```

**Important:** Respond with the CSV code block only. Do not use any programming language, code execution, or external tools. Use your domain knowledge and the reference data to make the prediction.
```

---

# Fold 3 — Target = S3

```text
You are provided with detailed experimental data on known polysulfone (PSF) membrane samples. Leveraging your comprehensive knowledge and intuition from scientific literature and domain expertise in polymer physics, predict the missing mechanical properties for the target sample listed below based solely on the provided reference data.

### Instructions
- Carefully analyze the provided reference dataset (nine samples).
- Predict the three target mechanical properties clearly and directly, without referencing external tools, programming, or model training.
- Base your judgment on polymer membrane physics and the structure–property trends observable in the reference data.

### Validation protocol
- Treat this as leave-one-out cross-validation: use ONLY the nine rows in the reference table to infer structure–property trends.
- Predict the three mechanical properties for the target sample.
- Do not use external tools, code execution, or web retrieval.
- Keep units exactly as specified.
- Output requirement: reply with CSV only inside a single code block — no prose, no extra text.
- Include the provided meta fields in every CSV row: model_name=LLM_Name, run=1–5.
- Return exactly three rows (one per property) with this header:
  model_name,run,sample,property,units,predicted
- Round all predicted values to two decimal places.

### Reference PSF membrane data (use these nine rows only)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) | Young's modulus (N/mm²) | Tensile strength (N/mm²) | Elongation at break (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|------------------------:|-------------------------:|------------------------:|
| S1    | 0.522              | 94.6               | 0.273          | 77.67        | 117.17                  | 4.82                     | 42.07                  |
| S2    | 0.364              | 93.3               | 0.202          | 79.16        | 90.18                   | 3.97                     | 46.61                  |
| S4    | 0.451              | 66.4               | 0.136          | 73.60        | 231.78                  | 9.61                     | 61.30                  |
| S5    | 0.408              | 66.6               | 0.154          | 73.49        | 182.59                  | 7.43                     | 60.69                  |
| S6    | 0.336              | 79.5               | 0.177          | 69.09        | 235.65                  | 9.65                     | 64.46                  |
| S7    | 0.403              | 81.3               | 0.240          | 71.18        | 176.51                  | 7.25                     | 69.04                  |
| S8    | 0.319              | 84.5               | 0.180          | 78.32        | 126.42                  | 5.22                     | 42.91                  |
| S9    | 0.842              | 90.0               | 0.224          | 78.86        | 107.67                  | 4.53                     | 51.25                  |
| S10   | 0.298              | 82.9               | 0.188          | 74.02        | 168.57                  | 7.13                     | 67.15                  |

### Target PSF membrane (predict its mechanical properties)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|
| S3    | 0.569              | 78.9               | 0.175          | 78.87        |

### Expected CSV output
```csv
model_name,run,sample,property,units,predicted
LLM_Name,1,S3,Young's modulus,N/mm^2,<value>
LLM_Name,1,S3,Tensile strength,N/mm^2,<value>
LLM_Name,1,S3,Elongation at break,%,<value>
```

**Important:** Respond with the CSV code block only. Do not use any programming language, code execution, or external tools. Use your domain knowledge and the reference data to make the prediction.
```

---

# Fold 4 — Target = S4

```text
You are provided with detailed experimental data on known polysulfone (PSF) membrane samples. Leveraging your comprehensive knowledge and intuition from scientific literature and domain expertise in polymer physics, predict the missing mechanical properties for the target sample listed below based solely on the provided reference data.

### Instructions
- Carefully analyze the provided reference dataset (nine samples).
- Predict the three target mechanical properties clearly and directly, without referencing external tools, programming, or model training.
- Base your judgment on polymer membrane physics and the structure–property trends observable in the reference data.

### Validation protocol
- Treat this as leave-one-out cross-validation: use ONLY the nine rows in the reference table to infer structure–property trends.
- Predict the three mechanical properties for the target sample.
- Do not use external tools, code execution, or web retrieval.
- Keep units exactly as specified.
- Output requirement: reply with CSV only inside a single code block — no prose, no extra text.
- Include the provided meta fields in every CSV row: model_name=LLM_Name, run=1–5.
- Return exactly three rows (one per property) with this header:
  model_name,run,sample,property,units,predicted
- Round all predicted values to two decimal places.

### Reference PSF membrane data (use these nine rows only)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) | Young's modulus (N/mm²) | Tensile strength (N/mm²) | Elongation at break (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|------------------------:|-------------------------:|------------------------:|
| S1    | 0.522              | 94.6               | 0.273          | 77.67        | 117.17                  | 4.82                     | 42.07                  |
| S2    | 0.364              | 93.3               | 0.202          | 79.16        | 90.18                   | 3.97                     | 46.61                  |
| S3    | 0.569              | 78.9               | 0.175          | 78.87        | 152.50                  | 6.56                     | 49.13                  |
| S5    | 0.408              | 66.6               | 0.154          | 73.49        | 182.59                  | 7.43                     | 60.69                  |
| S6    | 0.336              | 79.5               | 0.177          | 69.09        | 235.65                  | 9.65                     | 64.46                  |
| S7    | 0.403              | 81.3               | 0.240          | 71.18        | 176.51                  | 7.25                     | 69.04                  |
| S8    | 0.319              | 84.5               | 0.180          | 78.32        | 126.42                  | 5.22                     | 42.91                  |
| S9    | 0.842              | 90.0               | 0.224          | 78.86        | 107.67                  | 4.53                     | 51.25                  |
| S10   | 0.298              | 82.9               | 0.188          | 74.02        | 168.57                  | 7.13                     | 67.15                  |

### Target PSF membrane (predict its mechanical properties)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|
| S4    | 0.451              | 66.4               | 0.136          | 73.60        |

### Expected CSV output
```csv
model_name,run,sample,property,units,predicted
LLM_Name,1,S4,Young's modulus,N/mm^2,<value>
LLM_Name,1,S4,Tensile strength,N/mm^2,<value>
LLM_Name,1,S4,Elongation at break,%,<value>
```

**Important:** Respond with the CSV code block only. Do not use any programming language, code execution, or external tools. Use your domain knowledge and the reference data to make the prediction.
```

---

# Fold 5 — Target = S5

```text
You are provided with detailed experimental data on known polysulfone (PSF) membrane samples. Leveraging your comprehensive knowledge and intuition from scientific literature and domain expertise in polymer physics, predict the missing mechanical properties for the target sample listed below based solely on the provided reference data.

### Instructions
- Carefully analyze the provided reference dataset (nine samples).
- Predict the three target mechanical properties clearly and directly, without referencing external tools, programming, or model training.
- Base your judgment on polymer membrane physics and the structure–property trends observable in the reference data.

### Validation protocol
- Treat this as leave-one-out cross-validation: use ONLY the nine rows in the reference table to infer structure–property trends.
- Predict the three mechanical properties for the target sample.
- Do not use external tools, code execution, or web retrieval.
- Keep units exactly as specified.
- Output requirement: reply with CSV only inside a single code block — no prose, no extra text.
- Include the provided meta fields in every CSV row: model_name=LLM_Name, run=1–5.
- Return exactly three rows (one per property) with this header:
  model_name,run,sample,property,units,predicted
- Round all predicted values to two decimal places.

### Reference PSF membrane data (use these nine rows only)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) | Young's modulus (N/mm²) | Tensile strength (N/mm²) | Elongation at break (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|------------------------:|-------------------------:|------------------------:|
| S1    | 0.522              | 94.6               | 0.273          | 77.67        | 117.17                  | 4.82                     | 42.07                  |
| S2    | 0.364              | 93.3               | 0.202          | 79.16        | 90.18                   | 3.97                     | 46.61                  |
| S3    | 0.569              | 78.9               | 0.175          | 78.87        | 152.50                  | 6.56                     | 49.13                  |
| S4    | 0.451              | 66.4               | 0.136          | 73.60        | 231.78                  | 9.61                     | 61.30                  |
| S6    | 0.336              | 79.5               | 0.177          | 69.09        | 235.65                  | 9.65                     | 64.46                  |
| S7    | 0.403              | 81.3               | 0.240          | 71.18        | 176.51                  | 7.25                     | 69.04                  |
| S8    | 0.319              | 84.5               | 0.180          | 78.32        | 126.42                  | 5.22                     | 42.91                  |
| S9    | 0.842              | 90.0               | 0.224          | 78.86        | 107.67                  | 4.53                     | 51.25                  |
| S10   | 0.298              | 82.9               | 0.188          | 74.02        | 168.57                  | 7.13                     | 67.15                  |

### Target PSF membrane (predict its mechanical properties)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|
| S5    | 0.408              | 66.6               | 0.154          | 73.49        |

### Expected CSV output
```csv
model_name,run,sample,property,units,predicted
LLM_Name,1,S5,Young's modulus,N/mm^2,<value>
LLM_Name,1,S5,Tensile strength,N/mm^2,<value>
LLM_Name,1,S5,Elongation at break,%,<value>
```

**Important:** Respond with the CSV code block only. Do not use any programming language, code execution, or external tools. Use your domain knowledge and the reference data to make the prediction.
```

---

# Fold 6 — Target = S6

```text
You are provided with detailed experimental data on known polysulfone (PSF) membrane samples. Leveraging your comprehensive knowledge and intuition from scientific literature and domain expertise in polymer physics, predict the missing mechanical properties for the target sample listed below based solely on the provided reference data.

### Instructions
- Carefully analyze the provided reference dataset (nine samples).
- Predict the three target mechanical properties clearly and directly, without referencing external tools, programming, or model training.
- Base your judgment on polymer membrane physics and the structure–property trends observable in the reference data.

### Validation protocol
- Treat this as leave-one-out cross-validation: use ONLY the nine rows in the reference table to infer structure–property trends.
- Predict the three mechanical properties for the target sample.
- Do not use external tools, code execution, or web retrieval.
- Keep units exactly as specified.
- Output requirement: reply with CSV only inside a single code block — no prose, no extra text.
- Include the provided meta fields in every CSV row: model_name=LLM_Name, run=1–5.
- Return exactly three rows (one per property) with this header:
  model_name,run,sample,property,units,predicted
- Round all predicted values to two decimal places.

### Reference PSF membrane data (use these nine rows only)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) | Young's modulus (N/mm²) | Tensile strength (N/mm²) | Elongation at break (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|------------------------:|-------------------------:|------------------------:|
| S1    | 0.522              | 94.6               | 0.273          | 77.67        | 117.17                  | 4.82                     | 42.07                  |
| S2    | 0.364              | 93.3               | 0.202          | 79.16        | 90.18                   | 3.97                     | 46.61                  |
| S3    | 0.569              | 78.9               | 0.175          | 78.87        | 152.50                  | 6.56                     | 49.13                  |
| S4    | 0.451              | 66.4               | 0.136          | 73.60        | 231.78                  | 9.61                     | 61.30                  |
| S5    | 0.408              | 66.6               | 0.154          | 73.49        | 182.59                  | 7.43                     | 60.69                  |
| S7    | 0.403              | 81.3               | 0.240          | 71.18        | 176.51                  | 7.25                     | 69.04                  |
| S8    | 0.319              | 84.5               | 0.180          | 78.32        | 126.42                  | 5.22                     | 42.91                  |
| S9    | 0.842              | 90.0               | 0.224          | 78.86        | 107.67                  | 4.53                     | 51.25                  |
| S10   | 0.298              | 82.9               | 0.188          | 74.02        | 168.57                  | 7.13                     | 67.15                  |

### Target PSF membrane (predict its mechanical properties)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|
| S6    | 0.336              | 79.5               | 0.177          | 69.09        |

### Expected CSV output
```csv
model_name,run,sample,property,units,predicted
LLM_Name,1,S6,Young's modulus,N/mm^2,<value>
LLM_Name,1,S6,Tensile strength,N/mm^2,<value>
LLM_Name,1,S6,Elongation at break,%,<value>
```

**Important:** Respond with the CSV code block only. Do not use any programming language, code execution, or external tools. Use your domain knowledge and the reference data to make the prediction.
```

---

# Fold 7 — Target = S7

```text
You are provided with detailed experimental data on known polysulfone (PSF) membrane samples. Leveraging your comprehensive knowledge and intuition from scientific literature and domain expertise in polymer physics, predict the missing mechanical properties for the target sample listed below based solely on the provided reference data.

### Instructions
- Carefully analyze the provided reference dataset (nine samples).
- Predict the three target mechanical properties clearly and directly, without referencing external tools, programming, or model training.
- Base your judgment on polymer membrane physics and the structure–property trends observable in the reference data.

### Validation protocol
- Treat this as leave-one-out cross-validation: use ONLY the nine rows in the reference table to infer structure–property trends.
- Predict the three mechanical properties for the target sample.
- Do not use external tools, code execution, or web retrieval.
- Keep units exactly as specified.
- Output requirement: reply with CSV only inside a single code block — no prose, no extra text.
- Include the provided meta fields in every CSV row: model_name=LLM_Name, run=1–5.
- Return exactly three rows (one per property) with this header:
  model_name,run,sample,property,units,predicted
- Round all predicted values to two decimal places.

### Reference PSF membrane data (use these nine rows only)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) | Young's modulus (N/mm²) | Tensile strength (N/mm²) | Elongation at break (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|------------------------:|-------------------------:|------------------------:|
| S1    | 0.522              | 94.6               | 0.273          | 77.67        | 117.17                  | 4.82                     | 42.07                  |
| S2    | 0.364              | 93.3               | 0.202          | 79.16        | 90.18                   | 3.97                     | 46.61                  |
| S3    | 0.569              | 78.9               | 0.175          | 78.87        | 152.50                  | 6.56                     | 49.13                  |
| S4    | 0.451              | 66.4               | 0.136          | 73.60        | 231.78                  | 9.61                     | 61.30                  |
| S5    | 0.408              | 66.6               | 0.154          | 73.49        | 182.59                  | 7.43                     | 60.69                  |
| S6    | 0.336              | 79.5               | 0.177          | 69.09        | 235.65                  | 9.65                     | 64.46                  |
| S8    | 0.319              | 84.5               | 0.180          | 78.32        | 126.42                  | 5.22                     | 42.91                  |
| S9    | 0.842              | 90.0               | 0.224          | 78.86        | 107.67                  | 4.53                     | 51.25                  |
| S10   | 0.298              | 82.9               | 0.188          | 74.02        | 168.57                  | 7.13                     | 67.15                  |

### Target PSF membrane (predict its mechanical properties)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|
| S7    | 0.403              | 81.3               | 0.240          | 71.18        |

### Expected CSV output
```csv
model_name,run,sample,property,units,predicted
LLM_Name,1,S7,Young's modulus,N/mm^2,<value>
LLM_Name,1,S7,Tensile strength,N/mm^2,<value>
LLM_Name,1,S7,Elongation at break,%,<value>
```

**Important:** Respond with the CSV code block only. Do not use any programming language, code execution, or external tools. Use your domain knowledge and the reference data to make the prediction.
```

---

# Fold 8 — Target = S8

```text
You are provided with detailed experimental data on known polysulfone (PSF) membrane samples. Leveraging your comprehensive knowledge and intuition from scientific literature and domain expertise in polymer physics, predict the missing mechanical properties for the target sample listed below based solely on the provided reference data.

### Instructions
- Carefully analyze the provided reference dataset (nine samples).
- Predict the three target mechanical properties clearly and directly, without referencing external tools, programming, or model training.
- Base your judgment on polymer membrane physics and the structure–property trends observable in the reference data.

### Validation protocol
- Treat this as leave-one-out cross-validation: use ONLY the nine rows in the reference table to infer structure–property trends.
- Predict the three mechanical properties for the target sample.
- Do not use external tools, code execution, or web retrieval.
- Keep units exactly as specified.
- Output requirement: reply with CSV only inside a single code block — no prose, no extra text.
- Include the provided meta fields in every CSV row: model_name=LLM_Name, run=1–5.
- Return exactly three rows (one per property) with this header:
  model_name,run,sample,property,units,predicted
- Round all predicted values to two decimal places.

### Reference PSF membrane data (use these nine rows only)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) | Young's modulus (N/mm²) | Tensile strength (N/mm²) | Elongation at break (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|------------------------:|-------------------------:|------------------------:|
| S1    | 0.522              | 94.6               | 0.273          | 77.67        | 117.17                  | 4.82                     | 42.07                  |
| S2    | 0.364              | 93.3               | 0.202          | 79.16        | 90.18                   | 3.97                     | 46.61                  |
| S3    | 0.569              | 78.9               | 0.175          | 78.87        | 152.50                  | 6.56                     | 49.13                  |
| S4    | 0.451              | 66.4               | 0.136          | 73.60        | 231.78                  | 9.61                     | 61.30                  |
| S5    | 0.408              | 66.6               | 0.154          | 73.49        | 182.59                  | 7.43                     | 60.69                  |
| S6    | 0.336              | 79.5               | 0.177          | 69.09        | 235.65                  | 9.65                     | 64.46                  |
| S7    | 0.403              | 81.3               | 0.240          | 71.18        | 176.51                  | 7.25                     | 69.04                  |
| S9    | 0.842              | 90.0               | 0.224          | 78.86        | 107.67                  | 4.53                     | 51.25                  |
| S10   | 0.298              | 82.9               | 0.188          | 74.02        | 168.57                  | 7.13                     | 67.15                  |

### Target PSF membrane (predict its mechanical properties)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|
| S8    | 0.319              | 84.5               | 0.180          | 78.32        |

### Expected CSV output
```csv
model_name,run,sample,property,units,predicted
LLM_Name,1,S8,Young's modulus,N/mm^2,<value>
LLM_Name,1,S8,Tensile strength,N/mm^2,<value>
LLM_Name,1,S8,Elongation at break,%,<value>
```

**Important:** Respond with the CSV code block only. Do not use any programming language, code execution, or external tools. Use your domain knowledge and the reference data to make the prediction.
```

---

# Fold 9 — Target = S9

```text
You are provided with detailed experimental data on known polysulfone (PSF) membrane samples. Leveraging your comprehensive knowledge and intuition from scientific literature and domain expertise in polymer physics, predict the missing mechanical properties for the target sample listed below based solely on the provided reference data.

### Instructions
- Carefully analyze the provided reference dataset (nine samples).
- Predict the three target mechanical properties clearly and directly, without referencing external tools, programming, or model training.
- Base your judgment on polymer membrane physics and the structure–property trends observable in the reference data.

### Validation protocol
- Treat this as leave-one-out cross-validation: use ONLY the nine rows in the reference table to infer structure–property trends.
- Predict the three mechanical properties for the target sample.
- Do not use external tools, code execution, or web retrieval.
- Keep units exactly as specified.
- Output requirement: reply with CSV only inside a single code block — no prose, no extra text.
- Include the provided meta fields in every CSV row: model_name=LLM_Name, run=1–5.
- Return exactly three rows (one per property) with this header:
  model_name,run,sample,property,units,predicted
- Round all predicted values to two decimal places.

### Reference PSF membrane data (use these nine rows only)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) | Young's modulus (N/mm²) | Tensile strength (N/mm²) | Elongation at break (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|------------------------:|-------------------------:|------------------------:|
| S1    | 0.522              | 94.6               | 0.273          | 77.67        | 117.17                  | 4.82                     | 42.07                  |
| S2    | 0.364              | 93.3               | 0.202          | 79.16        | 90.18                   | 3.97                     | 46.61                  |
| S3    | 0.569              | 78.9               | 0.175          | 78.87        | 152.50                  | 6.56                     | 49.13                  |
| S4    | 0.451              | 66.4               | 0.136          | 73.60        | 231.78                  | 9.61                     | 61.30                  |
| S5    | 0.408              | 66.6               | 0.154          | 73.49        | 182.59                  | 7.43                     | 60.69                  |
| S6    | 0.336              | 79.5               | 0.177          | 69.09        | 235.65                  | 9.65                     | 64.46                  |
| S7    | 0.403              | 81.3               | 0.240          | 71.18        | 176.51                  | 7.25                     | 69.04                  |
| S8    | 0.319              | 84.5               | 0.180          | 78.32        | 126.42                  | 5.22                     | 42.91                  |
| S10   | 0.298              | 82.9               | 0.188          | 74.02        | 168.57                  | 7.13                     | 67.15                  |

### Target PSF membrane (predict its mechanical properties)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|
| S9    | 0.842              | 90.0               | 0.224          | 78.86        |

### Expected CSV output
```csv
model_name,run,sample,property,units,predicted
LLM_Name,1,S9,Young's modulus,N/mm^2,<value>
LLM_Name,1,S9,Tensile strength,N/mm^2,<value>
LLM_Name,1,S9,Elongation at break,%,<value>
```

**Important:** Respond with the CSV code block only. Do not use any programming language, code execution, or external tools. Use your domain knowledge and the reference data to make the prediction.
```

---

# Fold 10 — Target = S10

```text
You are provided with detailed experimental data on known polysulfone (PSF) membrane samples. Leveraging your comprehensive knowledge and intuition from scientific literature and domain expertise in polymer physics, predict the missing mechanical properties for the target sample listed below based solely on the provided reference data.

### Instructions
- Carefully analyze the provided reference dataset (nine samples).
- Predict the three target mechanical properties clearly and directly, without referencing external tools, programming, or model training.
- Base your judgment on polymer membrane physics and the structure–property trends observable in the reference data.

### Validation protocol
- Treat this as leave-one-out cross-validation: use ONLY the nine rows in the reference table to infer structure–property trends.
- Predict the three mechanical properties for the target sample.
- Do not use external tools, code execution, or web retrieval.
- Keep units exactly as specified.
- Output requirement: reply with CSV only inside a single code block — no prose, no extra text.
- Include the provided meta fields in every CSV row: model_name=LLM_Name, run=1–5.
- Return exactly three rows (one per property) with this header:
  model_name,run,sample,property,units,predicted
- Round all predicted values to two decimal places.

### Reference PSF membrane data (use these nine rows only)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) | Young's modulus (N/mm²) | Tensile strength (N/mm²) | Elongation at break (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|------------------------:|-------------------------:|------------------------:|
| S1    | 0.522              | 94.6               | 0.273          | 77.67        | 117.17                  | 4.82                     | 42.07                  |
| S2    | 0.364              | 93.3               | 0.202          | 79.16        | 90.18                   | 3.97                     | 46.61                  |
| S3    | 0.569              | 78.9               | 0.175          | 78.87        | 152.50                  | 6.56                     | 49.13                  |
| S4    | 0.451              | 66.4               | 0.136          | 73.60        | 231.78                  | 9.61                     | 61.30                  |
| S5    | 0.408              | 66.6               | 0.154          | 73.49        | 182.59                  | 7.43                     | 60.69                  |
| S6    | 0.336              | 79.5               | 0.177          | 69.09        | 235.65                  | 9.65                     | 64.46                  |
| S7    | 0.403              | 81.3               | 0.240          | 71.18        | 176.51                  | 7.25                     | 69.04                  |
| S8    | 0.319              | 84.5               | 0.180          | 78.32        | 126.42                  | 5.22                     | 42.91                  |
| S9    | 0.842              | 90.0               | 0.224          | 78.86        | 107.67                  | 4.53                     | 51.25                  |

### Target PSF membrane (predict its mechanical properties)

| Sample | Pore diameter (µm) | Contact angle (°) | Thickness (mm) | Porosity (%) |
|-------:|-------------------:|-------------------:|---------------:|-------------:|
| S10   | 0.298              | 82.9               | 0.188          | 74.02        |

### Expected CSV output
```csv
model_name,run,sample,property,units,predicted
LLM_Name,1,S10,Young's modulus,N/mm^2,<value>
LLM_Name,1,S10,Tensile strength,N/mm^2,<value>
LLM_Name,1,S10,Elongation at break,%,<value>
```

**Important:** Respond with the CSV code block only. Do not use any programming language, code execution, or external tools. Use your domain knowledge and the reference data to make the prediction.
```

---
