"""Model-free helpers shared by `src.modelling.steps` and `src.modelling.transformers`.

Nothing here imports a step or a transformer: CV splits and sample weights (`cv`), IC and
z-score metrics (`metrics`), ensemble / horizon blending (`ensemble`), feature encoding and
column resolution (`features`), the projected store loader (`panel`) and artifact paths /
metadata (`artifacts`).
"""
