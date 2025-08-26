# AI in Neuroimaging

## MRI’s role in Today’s Diagnostic Medical Imaging

- When to use
  - **Soft Tissue Evaluation** – MRI is ideal when detailed images of soft tissues are required (e.g.,
    brain, spinal cord, muscles, joints, ligaments, internal organs).
  - **Early Detection** – Used to detect very small or subtle pathological changes (tumors,
    demyelination in multiple sclerosis, early ischemia).
  - **Long-Term Monitoring** – Since there is no ionising radiation, MRI is safe for repeated scans,
    making it suitable for chronic disease follow-up.
  - **Specialized Cases** – Functional MRI (fMRI) for brain activity, MR spectroscopy (MRS) for chemical
    composition, MR angiography (MRA) for vessels, and Diffusion MRI for white matter tracts.

- Where to use:
  - Neurology – Brain imaging for tumors, stroke, multiple sclerosis, dementia, epilepsy.
  - Orthopedics & Sports Medicine – Muscles, ligaments, joints, cartilage tears, bone marrow
    pathology.
  - Spinal Imaging – Intervertebral discs, nerve compression, spinal cord lesions.
  - Cardiology – Cardiac MRI for structural and functional assessment.
  - Oncology – Tumor characterization, staging, and treatment monitoring.
  - Abdominal & Pelvic Imaging – Liver, kidneys, uterus, prostate, bowel, etc.

- Why to use:
  - Safe
    - Works by imaging water molecules (protons).
    - No ionising radiation (unlike CT/X-ray).
    - Safer for repeated use in children and long-term patients.
  - Detailed
    - High-resolution images of soft tissues.
    - Can reveal early or small pathological changes.
    - Provides structural and pathological detail (e.g., tumors, ischemic lesions).
  - Flexible
    - Multiple sequences and contrasts (T1, T2, FLAIR, DWI, SWI).
    - Imaging in any plane (axial, coronal, sagittal).
    - Specialized techniques (fMRI, MRA, dMRI, MTR, etc.) allow functional, vascular, and
      metabolic imaging.

- Limitations (Artefacts and Challenges)
  - Artefacts – Motion artefacts (from patient movement), susceptibility artefacts (from metal
    implants), partial volume effects.
  - Not Always Clear Enough – Sometimes lacks detail due to noise, patient movement, or technical
    limitations.
  - Longer Scan Time – Compared to CT, MRI takes more time and requires patient cooperation.
  - Contraindications – Not suitable for patients with pacemakers, ferromagnetic implants, or severe
    claustrophobia.

## What we can we see and what we can't see from clinical MRI

### What We Can See with Clinical MRI

- Structures
  - White matter, gray matter, CSF (segmentation possible).
  - Cortical surface and cortical thickness.
  - Subcortical structures (hippocampus, thalamus, basal ganglia).
  - Whole-brain tractography (white matter fiber bundles).

- Pathologies
  - Tumors, stroke, multiple sclerosis plaques, demyelination.
  - Atrophy (thinning of cortex, loss of volume).
  - Inflammation, edema.
  - Vascular abnormalities (via MRA, SWI).

- Quantitative Imaging
  - Cortical thickness mapping.
  - Cortical parcellation (dividing brain into functional/structural regions).
  - Diffusion metrics (DTI, dMRI for fiber tracking).

### What We Can’t See (Limitations of Clinical MRI)

- Microscopic Pathology
  - Individual axons, dendrites, synapses.
  - Cellular-level changes (only visible with histology or microscopy).
  - Molecular & Functional Changes
  - Neurotransmitters, metabolic processes (needs advanced MRI: MRS, PET for metabolism).
  - Very early micro-lesions below resolution threshold.

- Dynamic Real-Time Activity
  - Neuronal firing is too fast (milliseconds) — fMRI only gives indirect blood flow signals (seconds).
  - Artefacts & Clarity Issues
  - Motion artefacts (patient movement).
  - Susceptibility artefacts (metal implants).
  - Signal dropouts in some regions (e.g., near sinuses, ear canals).

### Conventional vs. Advanced MRI

- Conventional MRI:
  - T1, T2, FLAIR, SWI, DWI.
  - Excellent for structural imaging and obvious lesions.

- Advanced MRI:
  - fMRI → brain activity via blood flow.
  - MRS → brain chemistry.
  - dMRI/DTI → white matter fiber tracking.
  - Quantitative MRI → cortical thickness, atrophy mapping.
  - Black-blood imaging → vascular inflammation.

## The Role of AI in Diagnostic Imaging.

- What AI Can Do
  - Classification → “Which class is it?”
    - E.g., normal vs abnormal, tumor vs non-tumor.
    - Examples:
      - MRI sequence identification
      - CT-based hemorrhage case identification
  - Detection → “Where is it?”
    - Locates abnormalities with bounding boxes (e.g., hemorrhage, MS lesions).
    - Examples:
      - MRI-based MS lesion capture
      - MRI-based MS lesion activity
      - CT-based hemorrhage detection & triage
  - Segmentation → “What’s the boundary/volume?”
    - Precise outlines of tumors, brain regions, MS lesions, or hemorrhage.
    - Examples:
      - Brain extraction & normalization
      - Brain tumor segmentation
      - Brain atrophy estimation (longitudinal)
      - Cortical parcellation & volumetric analysis
  - Generation → “Make personalized information”
    - Automatic reports, disease progression models, treatment predictions.
    - Examples:
      - Automatic clinical reports
      - Automatic diagnosis suggestions
      - Disease progression prediction

- Advantages of AI
  - Speed → AI can process in seconds what may take humans 30–40 minutes.
  - Consistency → Reduces inter-observer variability between radiologists.
  - Sensitivity → Detects small or subtle lesions humans may miss.
  - Scalability → Handles large datasets, helps with triage in emergencies.
  - Knowledge Extraction → Learns from massive amounts of annotated cases.

- Limitations of AI
  - Bias in Training Data → If trained on limited demographics, results may not generalize.
  - False Positives/Negatives → Can over-call or miss findings without human oversight.
  - Black Box Issue → Hard to explain AI’s decision-making process.
  - Dependence on Data Quality → Garbage in → garbage out (artefacts, low-quality scans mislead AI).
  - Not a Replacement for Humans → AI supports, but radiologists must interpret in clinical context.

- Human vs. AI
  - AI is faster but not always smarter.
  - AI extracts patterns from data, while humans use clinical reasoning.
  - Best model = Human + AI collaboration (AI flags, radiologists confirm).
