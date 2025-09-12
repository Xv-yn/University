# IOA Notes

## Chlamydia trachomatis Infection Strategy

Chlamydia is an **obligated intracellular bacterium**.

- “Obligate” = must. The organism cannot survive or replicate outside host cells.
- “Intracellular” = it lives inside the host cell rather than floating around outside
  in tissue or blood.
- “Bacterium” = it’s a prokaryotic microorganism (not a virus, though it behaves
  somewhat like one).

1. Entry into the Body
   - Usually transmitted sexually (urogenital tract), but it can also be passed from
     mother to baby during birth → eye or lung infections in the newborn.

2. Elementary Body (EB) — The Infectious Form
   - `Outside of cells, Chlamydia exists as an elementary body`.
   - Think of it like a spore: small, tough, and designed to survive outside cells
     until it finds a new host.
   - It attaches to epithelial cells (the surface cells lining the genital tract, eye,
     or respiratory tract).

3. Cell Entry
   - The `EB binds to receptors` on the host cell surface (e.g. heparan sulfate,
     integrins).
   - The host cell pulls it inside via `endocytosis` → it’s now inside a little bubble
     (a vacuole).

4. Reticulate Body (RB) — The Replicating Form
   - Once inside, the `EB transforms into a reticulate body (RB)`.
   - RBs are larger and metabolically active but not infectious.
   - They divide by `binary fission` (just like other bacteria) inside the vacuole,
     which becomes an inclusion body (a protective bubble).

5. Multiplication and Evasion
   - Chlamydia avoids being digested by the host by `blocking fusion with lysosomes`.
     - It `releases specific inclusion membrane proteins` (Inc proteins) that insert
       into the vacuole membrane, altering its identity so the host cell treats it
       like a normal organelle instead of a pathogen.
   - It `steals nutrients` and ATP from the host.
   - The RBs multiply until the inclusion is packed.
   - It `reduces or alters MHC I expression` to prevent WBC detection.

6. Back to Elementary Bodies
   - After replication, RBs convert back into EBs.
   - These EBs are now ready to infect new cells.

7. Release
   - The host cell either:
     - `Bursts (lysis)` → releasing lots of EBs all at once, or
     - `Exocytosis` → pushes them out more gently.
   - EBs spread to nearby cells → the infection cycle continues.

## Immune Response to Chlamydia

### Innate Immune Response (First Responders)

- Epithelial cell sensing
  - Infected epithelial cells `recognize Chlamydia through pattern recognition`
    receptors (PRRs) like TLR3, TLR4, TLR9, and NOD receptors.
  - They `release pro-inflammatory cytokines` (IL0, TNF-α, IL-6, IL-8).
  - Effect: recruits neutrophils and macrophages to the site.

- Neutrophils
  - First responders, recruited by cytokines from infected epithelial cells.
  - Kill free Chlamydia elementary bodies (EBs) using:
    - Phagocytosis
    - Reactive oxygen species (ROS) (“toxic grenades”)
      - When a neutrophil engulfs a microbe, it activates the enzyme NADPH oxidase
        in its membrane.
      - This enzyme transfers electrons to oxygen, producing superoxide anion (O₂⁻).
      - Superoxide then reacts further to form hydrogen peroxide (H₂O₂) and other
        ROS like hydroxyl radicals (•OH) and hypochlorous acid (HOCl).
      - HOCl, in particular, is made by the enzyme myeloperoxidase (MPO) using H₂O₂
        and chloride ions — it’s basically bleach inside your cells.
    - NETs (DNA traps coated with antimicrobial proteins)
      - The neutrophil disassembles its own nucleus and mixes DNA with antimicrobial
        proteins (like elastase, defensins, MPO).
      - This sticky web of DNA and proteins is then expelled outside the cell.
      - The result is a trap that ensnares bacteria, fungi, and even some parasites,
        holding them in place and exposing them to toxic proteins.
  - Cause pus + inflammation, but can’t reach Chlamydia hiding inside cells.

- Macrophages
  - Phagocytose EBs; present antigens to adaptive immune system.
  - Release cytokines (IL-1, TNF-α, IL-12) → recruit more immune cells.
  - Problem: Chlamydia can survive inside macrophages, turning them into
    “Trojan horses.”

- NK Cells
  - Target infected epithelial cells with low/altered MHC I.
  - Kill using perforin + granzymes (apoptosis of host cell + bacteria inside).
  - Release IFN-γ → boosts macrophage killing ability.

> [!NOTE]
> Limitation: Chlamydia manipulates MHC I → sometimes “invisible” to NK cells.

### Adaptive Immune Response (Specific Defenders)

Antibodies (B cell side):

- Antibodies can neutralize extracellular elementary bodies (EBs) before they enter
  cells.
- But once Chlamydia is inside epithelial cells as reticulate bodies (RBs), antibodies
  can’t touch them.
- So even “perfect” antibodies only help during that brief extracellular stage.

Cytotoxic T Cells (CD8⁺ T cell side):

- Normally, infected cells display microbial peptides on MHC I, flagging themselves
  for destruction.
- Chlamydia interferes with this process:
  - It reduces or alters MHC I expression.
  - Result = infected cells “look normal enough” to escape T cell killing, but
    “not abnormal enough” to fully trigger NK cells either.
  - This “gray zone” means adaptive immunity can’t fully sterilize the infection.
