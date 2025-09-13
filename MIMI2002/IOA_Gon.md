# Neisseria gonorrhoeae Infection Strategy

Neisseria gonorrhoeae is facultative intracellular, meaning that it can survive short
times in cells but mainly thrives extracellularly.

1. Entry Point
   - Spread through sexual contact (genital, rectal, oral).
   - Bacteria land on mucosal surfaces (urethra, cervix, rectum, throat, conjunctiva).

2. Attachment to Epithelial Cells
   - Pili (fimbriae) act like grappling hooks, latching onto epithelial cells.
   - Opa proteins act like glue, binding to receptors such as CEACAMs.
     - These proteins also trigger the cell to "rearrange" themselves.
   - This tight binding prevents the bacteria from being washed away by mucus or
     urine.

> [!NOTE]
> Once N. gonorrhoeae crosses the epithelium into the subepithelial tissue, it doesn’t
> need to worry about being washed away by mucus/urine anymore.
>
> Meaning that:
>
> - It can be floating extracellularly in tissue fluid (where it triggers strong
>   neutrophil responses), or
> - Attaching to host cells (epithelial, immune, endothelial) using pili/Opa if it
>   wants closer interaction.

3. Invasion (Hijacking the Cell’s Machinery)
   - Opa proteins bind to receptors like CEACAMs, which trigger the host’s
     preprogrammed signaling pathways (PI3K, Rho GTPases).
   - These pathways cause the `actin cytoskeleton to rearrange`, making the membrane
     ruffle.
   - The cell, thinking it’s performing a normal uptake process, forms a vacuole that
     unintentionally engulfs the gonococcus (endocytosis-like event).
   - Instead of being sent to the lysosome for destruction, the vacuole is “re-labeled.”
   - The host’s own trafficking system (like a self-driving conveyor belt) carries
     the vacuole across the epithelial cell.
   - At the basolateral surface, the vacuole fuses with the membrane → bacteria are
     released into the underlying tissue (`transcytosis`).
   - Some bacteria remain on the surface, others may linger briefly inside cells, but
     the main strategy is to cross into the tissue and replicate extracellularly.

4. Immune Evasion During Colonization
   - IgA protease cuts up mucosal antibodies, weakening the first line of defense.
     - Gonorrhoeae secretes IgA protease, an enzyme that snips IgA in half at its
       hinge region.
     - Like releasing a cloud of poison that only affects IgA antibodies.

   - LOS (lipooligosaccharide) can be modified (sialylated) → makes it look like
     host molecules → avoids complement killing.
     - Simply put, it adds host-like sialic acid sugars to LOS, literally acting
       as camoflage.

   - Antigenic and phase variation of pili and Opa proteins = constant disguise.
     - Pili can undergo `antigenic variation` (their amino acid sequence changes
       due to recombination of pil genes).
     - `Phase variation` = the bacteria can switch different opa genes on/off. That
       means the bacterium can express different “versions” of Opa at different
       times.

5. Spread and Complications
   - Bacteria can persist in the mucosa → chronic inflammation → scarring.
     - If infection is not cleared, bacteria can persist in the mucosa, and the
       ongoing inflammation leads to scarring and tissue damage.
   - In women: can ascend → pelvic inflammatory disease, infertility.
   - In men: can cause epididymitis.
   - In newborns: eye infection (ophthalmia neonatorum) if exposed during delivery.

## Damage

- Inflammation and Symptoms
  - The infection triggers strong neutrophil recruitment (via cytokines IL-8, TNF-α).
  - Neutrophils release toxic molecules → cause tissue damage → pus (purulent
    discharge) and pain.
    - Neutrophils swarm the site and release:
      - Reactive oxygen species
      - Proteases (enzymes)
      - Antimicrobial peptides
    - This kills many bacteria, but also damages host tissue.
  - This inflammatory reaction explains the urethritis/cervicitis symptoms.
    - The pus (purulent discharge) = dead neutrophils + dead bacteria + tissue
      debris.
    - The pain and burning = inflamed, damaged mucosa from neutrophil attack
  - Analogy
    - It’s like a thief (gonorrhoea) wearing a cloak of invisibility (immune
      evasion).
    - Most of the time, guards (immune system) don’t see him.
    - But if he lingers too long or too many thieves gather, they eventually trip
      alarms.
    - The guards (neutrophils) rush in swinging wildly, hitting both the thief and
      the furniture (host tissue).
    - The chaos = pus + pain.

# Immune Response to Neisseria gonorrhoeae

## Innate Immune Response (First Responders)

1. Epithelial Cell Detection
   - When gonorrhoea attaches and invades, epithelial cells recognize bacterial
     components (LOS, peptidoglycan fragments) via TLRs (esp. TLR2/4) and NOD
     receptors.
   - They release IL-8, TNF-α, IL-1β → alarm signals.

2. Neutrophil Recruitment
   - IL-8 draws huge numbers of neutrophils into the mucosa.
   - Neutrophil recruitment occurs as soon as epithelial cells sense gonorrhoea. As well
     as swarming in after transcytosis.
   - Neutrophils engulf bacteria and release ROS + proteases.
   - Problem: gonorrhoea can sometimes survive inside neutrophils, using them as
     temporary shelters.
   - Result: lots of neutrophil death → pus (purulent discharge).

3. Complement Activation
   - Complement tries to form MAC (membrane attack complex) on gonorrhoea.
   - But LOS sialylation and host-like sugars inhibit complement binding.

4. Macrophages & Dendritic Cells
   - Try to engulf bacteria, but encounter resistance (surface variation + survival
     tricks).
   - Dendritic cell activation is dampened → weak adaptive priming.

## Adaptive Immune Response (Specific Defenders)

1. B Cells / Antibodies
   - Infection does induce IgG and mucosal IgA against pili, Opa, and LOS.
   - BUT:
     - IgA is cut by IgA protease.
     - Antigenic/phase variation makes antibodies rapidly obsolete.
   - So antibodies don’t provide lasting protection.

2. T Cells
   - CD4+ T-cell responses are triggered, but N. gonorrhoeae skews them toward Th17
     (neutrophil recruitment) **instead** of strong Th1 (bactericidal) or Th2 (antibody)
     responses.
     - Gonorrhoea-stimulated DCs release IL-6, IL-23, and TGF-β.
       - Cytokine environment → Th17 bias
       - IL-6 + IL-23 + TGF-β = the classic “recipe” for Th17 differentiation.
     - Th1 responses (IFN-γ, activated macrophages) are great for intracellular
       pathogens (like TB, Chlamydia).
       - But N. gonorrhoeae mostly lives outside cells once it’s crossed the epitheliu
   - It’s like fighting a fire-type boss in Pokémon — but gonorrhoea tricks you
     into only sending out grass-type Pokémon.
   - This means:
     - Lots of inflammation (neutrophils keep coming).
     - But poor development of protective memory.
     - CD8+ T cells play a minor role since gonorrhoea is mainly extracellular
       after crossing the epithelium.
