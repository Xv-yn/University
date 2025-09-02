# Pandemics

- Pandemics have had catastrophic effects across history (e.g., Black Death, Spanish Flu, COVID-19).
- Death tolls from pandemics rival or surpass major wars.
- Comparison of communicable vs. non-communicable diseases (2021) shows that while chronic diseases
  dominate globally, infectious disease outbreaks still cause significant mortality.

## Outbreaks

- Outbreaks are a recurring theme in popular culture and real-world crises.
- Influenza is a key concern because of:
  - Viral recombination: Exchange of genetic material between influenza strains in humans, birds,
    pigs, etc.
  - Historical pandemics:
    - 1918 Spanish Flu (H1N1),
    - 1957 Asian Flu (H2N2),
    - 1968 Hong Kong Flu (H3N2),
    - 2009 Swine Flu (H1N1pdm09).
- Reassortment between avian and human influenza strains drives new pandemics.

## Advances in Diagnostics

- Progression: Serology → PCR → High-throughput sequencing (HTS).
- HIV testing as an example: different assays balance cost, speed, and resolution.
- Genomics (especially whole genome sequencing, WGS) now revolutionizes diagnostics:
  - Provides high-resolution data,
  - Detects new or unknown pathogens,
  - Helps track transmission.

## Genomic Epidemiology

- Uses sequencing data to:
  - Identify pathogen lineages,
  - Build transmission trees,
  - Map outbreaks in real-time.
- Example workflows: outbreak isolates vs. control isolates, phylogenetic reconstruction, and
  epidemiological inference.

## Influenza Recombination & Bird Flu Risk

- Viral recombination enables the creation of hybrid influenza strains.
- H5N1 bird flu:
  - First identified in 1996, caused fatal human infections.
  - New clade 2.3.4.4b emerged in 2020, spreading in wild and domestic birds, spilling over to
    mammals (foxes, seals, cows).
    - A clade is a group of organisms (or viruses) that all share a common ancestor, defined by
      genetics.
- Raises concern for the next pandemic.

## Global Circulation & Spillover

- Maps show rapid spread of H5Nx influenza in wild birds and poultry globally (2021–2024).
- Human infections reported across multiple continents, often fatal.
- Spillover events occur in diverse ecological contexts.

## Genomics for Outbreak Control

- Big data + sequencing enables near real-time tracking of outbreaks.
- Mutations (SNPs) allow identification of transmission chains between patients.

## Sequencing Basics

- Steps:
  1. DNA/RNA extraction,
  2. Library preparation,
  3. Sequencing (short reads),
  4. Data analysis.
- Reference-based read mapping generates consensus genomes.

## Case Study 1: Severe Influenza A Infection

- Patient returned from Chile, admitted to ICU with Influenza A.
- Sequencing is used to:
  - Identify HA/NA subtypes,
  - Check for bird flu (H5N1),
  - Identify zoonotic source,
  - Investigate possible hospital transmission,
  - Integrate genomics + epidemiology in a report.

## Influenza Typing & Analysis

- Molecular typing divides Influenza A into subtypes (H1–H18, N1–N11).
- Consensus sequences can be uploaded to databases (e.g., GISAID).
- BLASTn searches help identify subtype and closest matches.
- Example: subtype H5N1 linked to Southern Elephant Seal and other animals.

## Phylogenetic Analysis

- Tools: BLASTn, Nextclade.
- Steps: upload sequences, compare to reference sets, build phylogenetic trees.
- Metadata (species, geography) informs transmission pathways.

## Recent Outbreaks

- Dairy cow outbreak of H5N1 clade 2.3.4.4b (2023–2024):
  - Spread through cattle, wild birds, and poultry.
  - Multiple US states affected.
  - Raises concern about endemicity in cows and potential further human transmission.

## Limitations & Case Study 2

- Limitations: analyzing only HA/NA segments may miss reassortments.
- Case Study 2: Possible nosocomial (hospital-acquired) H5N1 transmission.
  - Case 1 and Case 2 overlapped in hospital stay.
  - Sequencing requested to determine clade, subtype, and transmission pathway.
