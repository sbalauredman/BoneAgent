# BoneAgent

BoneAgent coordinates literature evidence, a three-tier digital twin, synthesis protocol generation, retrospective clinical feedback, and Pareto evaluation for calcium phosphate bone biomaterial discovery. Six typed agents exchange provenance-carrying messages while a multi-fidelity funnel screens more than 10,000 candidates with a surrogate, evaluates 100 scaffold geometries with finite-element models, and sends 10 candidates to quantum-level validation. The clinical channel learns failure boundaries and assimilates delayed outcomes into local neighborhoods of the property model.

## Scope

The release covers the closed-loop method, numerical objectives, physical task generation, clinical evidence normalization, data splitting, property surrogate training, statistical evaluation, and campaign orchestration. The principal objective is

`CBMD = 0.3 mechanics + 0.3 biology + 0.2 degradation + 0.2 clinical`.

Mechanics averages normalized compressive strength over 2–180 MPa and elastic modulus over 0.1–20 GPa. Biology averages osteoblast viability and osteogenic potential. Degradation rewards agreement with the 12–24 week bone-ingrowth window. The clinical term averages harmonized 24-month implant survival and bone-ingrowth fraction.

## Installation

Python 3.10 and an NVIDIA CUDA 12.1 environment are expected.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The Conda environment includes the FEniCS 2019.1.0 dependency required for scaffold mechanics.

```bash
conda env create -f environment.yml
conda activate boneagent
python -m pip install -e .
```

The container uses CUDA 12.1.1 and cuDNN 8.

```bash
docker build -t boneagent:1.0 .
docker run --gpus all --rm -v "$PWD/data:/workspace/data" -v "$PWD/records:/workspace/records" boneagent:1.0 --config configuration/campaign.yaml
```

VASP 6.4.1 is licensed separately and is not distributed in the container. Set the cluster launcher to the local licensed executable before submitting Tier 1 jobs.

## Data

All canonical dataset pages are listed in `dataset_sources.txt`. Data files are not bundled. Authentication and API terms remain those of each provider.

| Dataset | Version or snapshot | Records used | Access | Role |
|---|---:|---:|---|---|
| Materials Project CaP | study snapshot | 2,547 | Materials Project API | training and primary evaluation |
| AFLOW Bioceramics | study snapshot | 1,823 | AFLOW API | cross-validation and out-of-distribution evaluation |
| OQMD CaP Subset | study snapshot | 3,847 | OQMD download | formation-energy prediction |
| NOMAD DFT Calculations | study snapshot | 3,214 | NOMAD API | finite-element calibration and training |
| Matbench Formation Energy | Matbench 1.0 | 132,752 | Matbench package | MACE pretraining |
| ClinicalTrials.gov Bone Grafts | study query snapshot | 347 trials | ClinicalTrials.gov API | clinical feedback |
| PubMed and PMC Outcomes | study query snapshot | 812 articles | NCBI APIs | clinical feedback and failure modes |

Materials Project access requires an API key and is subject to its terms. AFLOW data use the provider's open database terms. OQMD data are available for research under its stated terms. NOMAD metadata are CC BY 4.0 while individual uploads can carry record-specific licenses. Matbench is distributed under BSD-3-Clause. ClinicalTrials.gov records are public government records. PubMed citations are public bibliographic records; full text must be restricted to PMC articles whose individual license permits the intended use.

After obtaining the source exports, create a content manifest:

```bash
boneagent-prepare --config configuration/campaign.yaml --input data/raw/*.csv --output data/manifest.json
```

The manifest records the byte count and SHA-256 digest of every local input. Keep provider identifiers in each normalized row so provenance remains traceable.

The primary evaluation uses random 70/15/15 splits and a second elements-out split. Expected test counts are 382 Materials Project samples, 273 AFLOW samples, and 577 OQMD samples, totaling 1,232 candidates. No private or patient-level records are expected. Clinical ingestion accepts aggregated outcomes only.

## Surrogate training

Pretraining uses the full Matbench formation-energy corpus, three message-passing layers, 128 hidden channels, spherical harmonic order two, correlation order three, eight Bessel functions, and a 5 Å cutoff.

```bash
boneagent-train --config configuration/pretraining.yaml --data data/processed/matbench.csv --output records/pretraining.pt
```

The pretraining schedule uses Adam, a learning rate of `5e-4`, weight decay of `1e-5`, cosine annealing, 500 epochs, batch size 32, and gradient norm clipping at 10.

```bash
boneagent-train --config configuration/finetuning.yaml --data data/processed/calcium_phosphate.csv --output records/finetuning.pt
```

Fine-tuning uses Adam at `1e-4` for at most 200 epochs, with validation-MAE early stopping after 20 epochs without improvement. The five campaign seeds are 42, 1024, 2048, 3407, and 7.

## Physical tiers

Tier 3 combines an equivariant crystal network with a scaffold-property ensemble. It is intended to rank more than 10,000 candidates per cycle at roughly 10 ms per candidate on the target GPU.

Tier 2 generates finite-element parameters for tetrahedral scaffold meshes. Geometries at porosity up to 0.7 require at least 50,000 elements. Higher-porosity structures require at least 100,000 elements. The boundary protocol follows uniaxial compression at 1 mm/min with a fixed bottom surface.

Tier 1 generates VASP inputs using PBE, PAW potentials, a 520 eV cutoff, Γ-centered meshes with density at least 30 Å, electronic tolerance `1e-6` eV, force tolerance `0.01` eV/Å, 15 Å surface vacuum, and a 3.5 eV effective Hubbard correction for calcium 3d states.

## Closed-loop campaign

```bash
boneagent-campaign --config configuration/campaign.yaml
```

The campaign begins with 10,000 candidates. It retains 100 after surrogate scoring and 10 after scaffold mechanics. Clinical constraints are refreshed every five cycles when new outcomes are present. Termination occurs after the Pareto hypervolume changes by less than one percent for two consecutive cycles. Campaign outputs are written atomically to `records/campaign_summary.json`.

The main configuration represents the reported hardware target:

- one NVIDIA A100 with 80 GB HBM2e for ML training and screening;
- one coordinator with 128 AMD EPYC 7763 CPU cores and 512 GB RAM;
- four DFT nodes with 64 CPU cores per node;
- approximately 72 wall-clock hours for an 8–10 cycle campaign;
- approximately 33 GPU-hours and 1,024 CPU-hours per campaign.

GPU work is time-multiplexed between the crystal surrogate and scaffold ensemble. DFT dominates elapsed time at approximately 40 hours, while finite-element evaluation contributes approximately 8 hours.

## Evaluation

```bash
boneagent-evaluate --config configuration/campaign.yaml --target records/target.csv --prediction records/prediction.csv --output records/metrics.json
```

Evaluation reports mean absolute error, root mean squared error, coefficient of determination, Spearman correlation, and a 10,000-resample percentile interval for absolute error. Campaign comparisons use two-sided paired Wilcoxon signed-rank tests across five seeds with Bonferroni correction across twelve baselines. The corrected significance threshold is 0.05.

The primary aggregate target is CBMD `0.847 ± 0.023` over five runs with a reported 95% interval of `[0.819, 0.876]`. Digital-twin validation also reports formation-energy MAE, property R², osteoinductivity AUC, rank correlation against experimental ordering, throughput, and cycles to convergence.

## Experiment variants

Configuration files isolate the reported component removals:

- `ablation_without_clinical_feedback.yaml` removes the clinical objective and constraints;
- `ablation_single_fidelity.yaml` removes the surrogate-to-FEM-to-DFT funnel;
- `ablation_without_coordination.yaml` removes the shared message bus;
- `ablation_without_failure_constraints.yaml` retains outcomes without hard boundaries;
- `ablation_without_assimilation.yaml` retains static clinical information without Kalman updates;
- `ood_bioactive_glass.yaml` transfers without retraining to the SiO2-CaO-Na2O-P2O5 system;
- `ood_titanium.yaml` transfers without retraining to Ti-6Al-4V.

## Repository map

`code/boneagent/agents` contains typed roles and message transport. `code/boneagent/twin` contains crystal, scaffold, uncertainty, and data-assimilation models. `code/boneagent/physics` emits quantum and continuum task definitions. `code/boneagent/evidence` normalizes clinical literature and virtual-center estimates. `code/boneagent/science` contains acquisition, scoring, Pareto selection, and hypervolume calculations. `code/boneagent/engine` owns optimization and training state. `code/boneagent/analysis` provides reported metrics and paired statistics. `configuration` holds the principal schedule, component removals, and transfer settings.

## Privacy and data governance

Only aggregate, de-identified, publicly released records belong in the clinical table. Do not ingest patient names, dates of birth, addresses, medical-record identifiers, free-text case notes, or row-level participant exports. Provider access tokens must be supplied through the local environment and must never be written into configuration files, logs, manifests, or campaign artifacts.

The virtual-center model groups aggregate reports into ClinicalTrials.gov, PubMed Europe, PubMed Americas, and PubMed Asia-Pacific strata. These labels describe evidence-source strata and must not be interpreted as treating institutions or patient-level sites.
