# PEA Investigation Toolkit

**Dramatically accelerate issue investigation and bug fixing** with systematic tools, templates, and knowledge-building workflows.

## 🚀 Quick Start

### Investigate an Issue in 5 Minutes

```bash
# 1. Create reproduction script (1 minute)
python tools/create_reproduction.py --issue 408 --pattern empty-periods --accession 0000320193-18-000070

# 2. Run quick analysis (2 minutes)
python tools/quick_investigate.py --issue 408 --pattern empty-periods --compare

# 3. Generate comprehensive report (2 minutes)
python tools/quick_investigate.py --issue 408 --pattern empty-periods --full-analysis
```