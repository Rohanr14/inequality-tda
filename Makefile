# ---------------------------------------------------------------------------
# Targets
#   make env        – create local venv + pip install
#   make process    – fetch & pickle ACS percentiles  (data_loader.py)
#   make analyse    – run PH pipeline, write CSV      (ph_pipeline.py)
#   make plots      – generate PNGs                   (viz.py)
#   make regression – fixed‑effects OLS
#   make sens       – 101 vs 201 bin correlation
#   make dashboard  – launch Streamlit locally
#   make demo       – env → process → analyse → plots → dashboard
# ---------------------------------------------------------------------------

PY?= python
PIP_ACT = . .venv/bin/activate
RUN = PYTHONPATH=src $(PY)

env:
	$(PY) -m venv .venv && $(PIP_ACT) && pip install -r requirements.txt

process:
	$(PIP_ACT) && $(RUN) -m data_loader

analyse:
	$(PIP_ACT) && $(RUN) -m ph_pipeline

plots:
	$(PIP_ACT) && $(RUN) -m viz

dashboard:
	$(PIP_ACT) && streamlit run src/dashboard/app.py

regression:
	$(PIP_ACT) && $(RUN) -m analysis.fixed_effects

sens:
	$(PIP_ACT) && $(RUN) -m analysis.bin_sensitivity

mobility:
	$(PIP_ACT) && $(RUN) -m analysis.mobility_validation

tda:
	$(PIP_ACT) && $(RUN) -m analysis.genuine_tda

wasserstein:
	$(PIP_ACT) && $(RUN) -m analysis.wasserstein_analysis

mapper:
	$(PIP_ACT) && $(RUN) -m analysis.mapper_graph

temporal:
	$(PIP_ACT) && $(RUN) -m analysis.temporal_holdout

crossoutcome:
	$(PIP_ACT) && $(RUN) -m analysis.cross_outcome_validation

advanced: mobility tda wasserstein mapper temporal crossoutcome

demo: env process analyse plots dashboard