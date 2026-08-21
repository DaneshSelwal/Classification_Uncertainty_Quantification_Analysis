import re

with open("DETAILED_DOCUMENTATION.md", "r") as f:
    content = f.read()

# 1. Update intro paragraph
content = content.replace("seven state-of-the-art", "twelve state-of-the-art")

# 2. Add to TOC
new_toc = """11. [Method 7 — SCMCP (Spatial Multi-Head Conformal Prediction)](#11-method-7--scmcp-spatial-multi-head-conformal-prediction)
12. [Method 8 — Focal Loss & CB Focal Loss](#12-method-8--focal-loss--cb-focal-loss)
13. [Method 9 — EDL & EDL_v2 (Evidential Deep Learning)](#13-method-9--edl--edl_v2)
14. [Method 10 — CDL (Credal Deep Learning)](#14-method-10--cdl-credal-deep-learning)
15. [Method 11 — MambaHSI](#15-method-11--mambahsi)
16. [Method 12 — DOFA (Dynamic Wavelength Tokenization)](#16-method-12--dofa-dynamic-wavelength-tokenization)
17. [Comparative Summary of All Methods](#17-comparative-summary-of-all-methods)"""

content = re.sub(r'11\. \[Method 7.*?12\. \[Comparative Summary of All Methods\]\(#12-comparative-summary-of-all-methods\)', new_toc, content, flags=re.DOTALL)

# Adjust subsequent TOC numbers
content = content.replace("13. [Evaluation", "18. [Evaluation")
content = content.replace("14. [Master", "19. [Master")
content = content.replace("15. [Examples", "20. [Examples")
content = content.replace("16. [References", "21. [References")

# 3. Add Methods 8-12 sections
new_methods = """## 12. Method 8 — Focal Loss & CB Focal Loss

**Directory:** `Classification/trials/focal_loss/` & `Classification/trials/cb_focal_loss/`

Focuses on resolving class imbalance via standard Focal Loss and Class-Balanced Focal Loss. Enhances the predictive confidence for minority classes and adjusts the margin of prediction sets appropriately.

---

## 13. Method 9 — EDL & EDL_v2

**Directory:** `Classification/trials/edl/` & `Classification/trials/edl_v2/`

Implements Evidential Deep Learning based on Subjective Logic. Replaces softmax with a Dirichlet distribution output to quantify epistemic and aleatoric uncertainty without the need for sampling or ensembling.

---

## 14. Method 10 — CDL (Credal Deep Learning)

**Directory:** `Classification/trials/cdl/`

Extends evidential principles using imprecise probabilities and credal sets. Captures severe uncertainty and conflicting evidence by tracking the bounds of allowable probability distributions.

---

## 15. Method 11 — MambaHSI

**Directory:** `Classification/mambahsi/`

Adapts the state-of-the-art Mamba State-Space Model architecture for Hyperspectral Image classification. Offers linear scaling for long-range spatial-spectral sequences, solving the quadratic bottleneck of Vision Transformers.

---

## 16. Method 12 — DOFA (Dynamic Wavelength Tokenization)

**Directory:** `Classification/dofa/`

Dynamic Wavelength Tokenization framework incorporating DOFA Spectral and DOFA Hiera Fusion approaches. Designed to handle varying continuous spectral channels across multi-sensor remote sensing payloads, ensuring highly stable token generation.

---

## 17. Comparative Summary of All Methods"""

content = content.replace("## 12. Comparative Summary of All Methods", new_methods)

# Update structural mentions
content = content.replace("## 13. Evaluation Metrics", "## 18. Evaluation Metrics")
content = content.replace("## 14. Master Hyperparameter", "## 19. Master Hyperparameter")
content = content.replace("## 15. Examples Directory", "## 20. Examples Directory")
content = content.replace("## 16. References", "## 21. References")

with open("DETAILED_DOCUMENTATION.md", "w") as f:
    f.write(content)

