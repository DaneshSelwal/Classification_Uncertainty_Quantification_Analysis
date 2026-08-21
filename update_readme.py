import re

with open("README.md", "r") as f:
    content = f.read()

new_structure = """```text
.
├── Classification/    # 📂 Main Project Folder (Upload this to MyDrive/)
│   ├── baseline/      # 📂 Standard UQ (MC Dropout, Temp Scaling)
│   ├── credit/        # 📂 CREDIT: Calibration-aware training
│   ├── dapm/          # 📂 DAPM: Deep Adaptive Predictive Modeling
│   ├── data/          # 📊 Raw Datasets (data.csv, multispectral/)
│   ├── dofa/          # 📂 DOFA: Dynamic Wavelength Tokenization (New)
│   ├── ensemble/      # 📂 CreDE: Credal Deep Ensembles
│   ├── mambahsi/      # 📂 MambaHSI Architecture for classification
│   ├── multicp/       # 📂 MultiCP: Multi-head Conformal Prediction
│   ├── multicp_sacp/  # 📂 SCMCP: Spatial MultiCP (New)
│   ├── sacp/          # 📂 SACP: Self-Adaptive Conformal Prediction
│   └── trials/        # 🧪 Experimental Methods
│       ├── cb_focal_loss/
│       ├── cdl/
│       ├── conformal_reg/
│       ├── edl/
│       ├── edl_v2/
│       └── focal_loss/
├── examples/          # 📁 Example Dataset Suites (untouched)
└── README.md          # 📑 Documentation
```"""

# Replace the tree block
content = re.sub(r'```text\n\.\n├── Classification/.*?```', new_structure, content, flags=re.DOTALL)

# Add DOFA method description if not present
dofa_method = """### Method 12: DOFA
**Location**: `Classification/dofa`
Dynamic Wavelength Tokenization.
* **Architecture**: DOFA Spectral and DOFA Hiera Fusion for robust multispectral tokenization.
* **Focus**: Analyzing prediction stability across continuous spectral channels.

---"""

if "Method 12: DOFA" not in content:
    content = content.replace("## 🚀 Getting Started (Colab-First)", dofa_method + "\n\n## 🚀 Getting Started (Colab-First)")

# Fix the method paths to reflect the 'trials/' directory
content = content.replace("`Classification/focal_loss` & `Classification/cb_focal_loss`", "`Classification/trials/focal_loss` & `Classification/trials/cb_focal_loss`")
content = content.replace("`Classification/edl` & `Classification/edl_v2`", "`Classification/trials/edl` & `Classification/trials/edl_v2`")
content = content.replace("`Classification/cdl`", "`Classification/trials/cdl`")

with open("README.md", "w") as f:
    f.write(content)

