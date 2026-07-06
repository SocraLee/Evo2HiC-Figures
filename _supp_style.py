"""Shared style + clade utilities for the supp 13 / 19 / 20 set.

Loads `CLAUDE_CLADE_DIR` from paths.py so the reader edits paths in one place.

Style (Nature Comm-style supp figures):
  - Arial, 7pt body
  - 7.5pt bold panel labels (a, b, c, ...)
  - No panel titles (captions live in the manuscript)

Clade override:
  - The DNA Zoo claude_*_clean.txt files (data/claude/) cover 6 named
    clades; GENUS_PATCH assigns clades for the remaining genera.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl

from paths import CLAUDE_CLADE_DIR


GENUS_PATCH = [
    (['canis', 'lycaon', 'urocyon', 'vulpes',
      'leopardus', 'felis', 'panthera', 'prionailurus', 'puma',
      'otocolobus', 'herpailurus', 'neofelis',
      'crocuta', 'hyaena', 'cryptoprocta',
      'ursus', 'tremarctos', 'helarctos',
      'spilogale', 'mustela', 'martes', 'meles', 'taxidea',
      'lontra', 'aonyx', 'enhydra', 'pteronura',
      'otaria', 'arctocephalus', 'mirounga', 'leptonychotes',
      'halichoerus', 'phoca', 'odobenus',
      'mungos', 'urva', 'paradoxurus', 'genetta', 'civetta', 'cryptoprocta',
      'bassariscus', 'nasua', 'potos', 'procyon', 'ailurus',
      'phataginus', 'manis_',
      'myrmecophaga', 'tamandua',
      'tapir', 'rhinoceros', 'equus_',
      'addax', 'tragelaphus', 'syncerus', 'antidorcas', 'nanger',
      'okapia', 'giraffa', 'odocoileus', 'cervus_', 'capreolus',
      'lama_', 'vicugna', 'camelus_', 'sus_', 'phacochoerus',
      'hippopotamus',
      'phocoena', 'peponocephala', 'lagenorhynchus', 'megaptera',
      'balaenoptera', 'stenella', 'steno', 'mesoplodon', 'orcaella',
      'eubalaena', 'physeter', 'cephalorhynchus', 'kogia', 'monodon',
      'delphinapterus', 'globicephala', 'inia', 'sotalia', 'lipotes',
      'pontoporia', 'platanista',
      'sciurus', 'meriones', 'myocastor', 'lagostomus', 'uromys',
      'xerus', 'hydrochoerus', 'pachyuromys', 'cricetus',
      'mastacomys', 'mesocricetus', 'peromyscus', 'oryzomys',
      'cricetulus', 'fukomys', 'rattus', 'mus_',
      'sylvilagus', 'lepus_',
      'pteropus', 'rousettus', 'eidolon', 'corynorhinus', 'myotis',
      'eptesicus', 'nyctalus', 'pipistrellus', 'desmodus',
      'phyllostomus', 'macroglossus', 'cynopterus', 'trachops',
      'noctilio', 'rhinolophus', 'lasiurus', 'natalus', 'epomops',
      'cebuella', 'papio', 'pithecia', 'leontopithecus',
      'cercopithecus', 'symphalangus', 'varecia', 'eulemur',
      'otolemur', 'rhinopithecus', 'macaca', 'callicebus',
      'cebus', 'ateles', 'aotus', 'saimiri', 'sapajus',
      'piliocolobus', 'nomascus', 'pongo', 'hylobates',
      'erythrocebus', 'mandrillus', 'chlorocebus', 'colobus',
      'plecturocebus', 'theropithecus', 'allenopithecus',
      'pan_', 'gorilla', 'homo_',
     ], 'Mammalia'),
    (['setonix', 'dasyurus', 'sminthopsis', 'pseudocheirus',
      'pseudochirops', 'macropus', 'monodelphis', 'thylacinus',
      'wallabia', 'isoodon', 'perameles', 'didelphis', 'sarcophilus',
      'phascolarctos', 'vombatus', 'myrmecobius', 'phascogale',
      'potorous', 'dendrolagus', 'notamacropus', 'osphranter',
      'lagorchestes'], 'Marsupialia'),
    (['gymnogyps', 'eopsaltria', 'gallus_', 'corvus_', 'apus_'], 'Reptilia'),
    (['uta_', 'crocodylus', 'gekko', 'pelodiscus', 'anolis',
      'chrysemys', 'chelonia', 'caretta', 'intellagama',
      'sphaerodactylus'], 'Reptilia'),
    (['acyrthosiphon', 'aedes', 'agapostemon', 'schistocerca',
      'magicicada', 'halictus', 'lasioglossum', 'phlebotomus',
      'lutzomyia', 'tanypteryx', 'pachyrhynchus', 'aphis_',
      'drosophila', 'apis_', 'bombyx', 'tribolium', 'augochlora'],
     'Protostomes'),
    (['arachis', 'rubus', 'arabidopsis', 'oryza_', 'solanum',
      'glycine_', 'medicago'], 'Angiosperms'),
    (['cristatella', 'lottia', 'mytilus', 'crassostrea'], 'Protostomes'),
    (['maccullochella', 'danio_', 'oryzias', 'salmo', 'gadus',
      'oncorhynchus', 'esox', 'tetraodon', 'astyanax'], 'Actinopterygii'),
]


def load_clade_map(cdir: Path = CLAUDE_CLADE_DIR) -> dict:
    """Read claude_*_clean.txt then expand with GENUS_PATCH so every
    species in result/multi/SPC.csv falls into one of the named clades."""
    species2clade: dict[str, str] = {}
    for f in sorted(cdir.iterdir()):
        if f.name.startswith('claude_') and f.name.endswith('_clean.txt'):
            cl = f.name.replace('claude_', '').replace('_clean.txt', '')
            with open(f) as fh:
                for line in fh:
                    s = line.strip()
                    if s:
                        species2clade[s] = cl
    return species2clade


def assign_clade(species: str, base_map: dict) -> str:
    if species in base_map:
        return base_map[species]
    base = species.split('__')[0]
    if base in base_map:
        return base_map[base]
    s_lower = species.lower()
    for keywords, clade in GENUS_PATCH:
        for kw in keywords:
            if kw in s_lower:
                return clade
    return 'Other'


def apply_supp_style():
    mpl.rcParams.update({
        'font.family':       'Arial',
        'font.size':         7,
        'axes.labelsize':    7,
        'axes.titlesize':    7,
        'xtick.labelsize':   7,
        'ytick.labelsize':   7,
        'legend.fontsize':   7,
        'figure.titlesize':  7,
        'lines.linewidth':   0.8,
        'axes.linewidth':    0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size':  2,
        'ytick.major.size':  2,
    })


PANEL_LABEL_KW = dict(fontsize=7.5, fontweight='bold', ha='left', va='top')
OBS_BRACKET_COLOR  = '#0000ff'
PRED_BRACKET_COLOR = '#008000'
DELTA_POS_COLOR = '#fb8072'
DELTA_NEG_COLOR = '#bebada'
