# Data — Exoskeleton High-Level Control (Gait)

Ce répertoire décrit **où obtenir** les jeux de données de marche humaine et **comment organiser** les fichiers pour reproduire les résultats (génération de trajectoires hanche/genou, estimation période/phase).

⚠️ **Aucune donnée brute n’est redistribuée ici.** Merci de télécharger depuis les sources officielles et de respecter leurs licences/conditions d’usage.

---

## Jeux de données référencés

1) **Embry et al., 2018** — *The Effect of Walking Incline and Speed on Human Leg Kinematics, Kinetics, and EMG.*  
   **DOI**: `10.21227/GK32-E868`  
   **Contenu** : cinématique/kinétique/EMG pour différentes **pentes** et **vitesses**.

2) **Fukuchi et al., 2018 (PeerJ)** — *A public dataset of overground and treadmill walking kinematics and kinetics in healthy individuals.*  
   **DOI**: `10.7717/peerj.4640`  
   **Contenu** : cinématique/kinétique **tapis** et **sol**, sujets sains, large éventail de vitesses.

3) **Moreira et al., 2021 (Scientific Data)** — *Lower limb kinematic, kinetic, and EMG data from young healthy humans during walking at controlled speeds.*  
   **DOI**: `10.1038/s41597-021-00881-3`  
   **URL**: <https://www.nature.com/articles/s41597-021-00881-3>  
   **Contenu** : cinématique/kinétique/EMG à **vitesses contrôlées**, sujets jeunes sains.

> 💡 Ces trois sources sont **complémentaires** (pente, tapis/sol, vitesses contrôlées) et couvrent les besoins pour entraîner/valider un contrôleur haut-niveau (DNN + GPR) sur la marche.

---

## Arborescence recommandée

data/
├─ README.md                 # ce fichier
├─ samples/                  # petits échantillons synthétiques fournis
├─ raw/                      # données brutes téléchargées depuis les DOIs (NON suivies par git)
│  ├─ embry_2018/            # 10.21227/GK32-E868
│  ├─ fukuchi_2018_peerj/    # 10.7717/peerj.4640
│  └─ moreira_2021_sdata/    # 10.1038/s41597-021-00881-3
├─ interim/                  # conversions/normalisations intermédiaires
└─ processed/                # cycles normalisés, keypoints & features prêts pour le modèle


`raw/` est **ignoré par git** (voir `.gitignore`).  
`samples/` contient **des données synthétiques** minimales pour exécuter les notebooks/tests sans données privées.

---

## Schéma HDF5 (datasetV2.h5)

Le fichier `datasetV2.h5` est organisé en **sujets** → **essais** → **côtés** → **angles**, avec métadonnées par sujet et par essai.

### Hiérarchie

/{Subject}/
subjectdetails/
Age # float (années)
Gender # string/float encodé (selon import)
Height # float (m) ou (cm) — préciser l’unité utilisée
Weight # float (kg)
Id # identifiant sujet
{Trial}/
description/
Speed # float (m/s)
Incline # float (degrés) (0 = plat, >0 montée, <0 descente)
left/
angles/
hip/x # float64, shape = (N_cycles, 150)
hip/y # float64, shape = (N_cycles, 150)
knee/x # float64, shape = (N_cycles, 150)
time # float64, shape = (N_cycles, 150) # temps par cycle
time_norm # float64, shape = (N_cycles, 150) # 0→1 ou 0→100%
right/
angles/
hip/x # float64, shape = (N_cycles, 150)
hip/y # float64, shape = (N_cycles, 150)
knee/x # float64, shape = (N_cycles, 150)
time # float64, shape = (N_cycles, 150)
time_norm # float64, shape = (N_cycles, 150)


- **Subjects** : par ex. `AB01`, `AB02`, …  
- **Trials** : libellés de type `s0x8d10`, `s1i7x5`, etc. (codent des conditions vitesse/pente).  
- **Côtés** : `left`, `right`.  
- **Articulations** : `hip`, `knee`.  
- **Axes** : `x`, `y` (ex. sagittal/coronal suivant la convention).  
- **Taille des matrices** : `(N_cycles, 150)` = **N cycles** par essai, **150 points** par cycle (échantillonnage normalisé du cycle de marche).  
- **Métadonnées essai** : `description/Speed` (m/s), `description/Incline` (degrés).  
- **Démographie** sujet : `Age`, `Gender`, `Height`, `Weight`, `Id`.

> Si une autre unité est utilisée (p. ex. Height en cm), la préciser ici et **convertir** dans les scripts de préparation.

---

## Format CSV (processed) — mapping recommandé

Pour l’entraînement/évaluation, on conseille d’exporter vers un **CSV tabulaire** par échantillon de cycle, avec au minimum :

| Colonne        | Type   | Description |
|---             |---     |---|
| `subject_id`   | str    | ex. `AB03` |
| `trial_id`     | str    | ex. `s1x2i10` |
| `side`         | str    | `left` / `right` |
| `joint`        | str    | `hip` / `knee` |
| `axis`         | str    | `x` / `y` |
| `cycle_idx`    | int    | index de cycle (0…N-1) |
| `cycle_pct`    | float  | 0–100 (ou 0–1) |
| `value_deg`    | float  | angle (degrés) |
| `speed_mps`    | float  | `description/Speed` |
| `incline_deg`  | float  | `description/Incline` |
| `age`          | float  | années |
| `gender`       | str/int| encodage à préciser |
| `height_m`     | float  | m (convertir si nécessaire) |
| `weight_kg`    | float  | kg |

> Si vous ne tenez qu’aux profils **sagittaux**, choisir l’axe correspondant (`x` ou `y`) et fixer l’autre champ.
