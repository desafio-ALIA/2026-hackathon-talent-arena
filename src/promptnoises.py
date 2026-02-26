#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
promptnoises.py

This script generates 3 corrupted variants of Spanish prompts:

BLOCK 1 — typos
    - Keyboard typos (qwerty mistakes, character omission, abbreviations, random space removal)
        * 1-2 typos max.
    - General normalization rules:
        * ALWAYS removes open question marks (¿)
        * NEVER removes commas
        * Removes accents with a 60% probability

BLOCK 2 — grammatical errors
    - General normalization rules:
        * ALWAYS removes accents
        * ALWAYS removes open question marks (¿)
        * ALWAYS removes commas
    - Applies N grammatical changes in fixed order
        * 3-4 grammatical errors max.

BLOCK 3 — custom changes
    - Uses the same functions in blocks 1 and 2 but selects the rules by weights defined in custom_config.yaml
    - No limit in n of errors (defined by user)
    - Normalization rules defined by user (true/false)

Usage:
    python promptnoises.py --input_json test_1.json --output_json test_output.json --custom_config custom_config.yaml --seed 42
    python promptnoises.py --input_csv  in.csv      --output_csv  out.csv       --custom_config custom_config.yaml --seed 42
"""

# ============================================================
# Imports
# ============================================================

import argparse
import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

try:
    import yaml  # PyYAML
except ImportError as e:
    raise ImportError("Falta PyYAML. Instálalo con: pip install pyyaml") from e


# ============================================================
# General utilities
# ============================================================

def strip_accents(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


def weighted_choice(items: List[Any], weights: List[float]) -> Any:
    if len(items) != len(weights) or not items:
        raise ValueError("weighted_choice: items/weights mismatch or empty.")
    total = float(sum(max(0.0, float(w)) for w in weights))
    if total <= 0:
        return random.choice(items)
    r = random.random() * total
    upto = 0.0
    for item, w in zip(items, weights):
        upto += max(0.0, float(w))
        if upto >= r:
            return item
    return items[-1]


# ============================================================
# Block 1 and 2 lists
# ============================================================

QWERTY_NEIGHBORS = {
    'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy',
    'y': 'tghu', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
    'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm',
    'l': 'kop',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb',
    'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
}

COMMON_PRETERITE_2SG = [
    "dijiste", "hiciste", "pudiste", "quisiste", "supiste", "fuiste", "tuviste",
    "viste", "pusiste", "trajiste", "viniste", "saliste", "llegaste",
    "preguntaste", "contestaste", "intentaste", "probaste", "buscaste",
    "encontraste", "mandaste", "enviaste", "escribiste", "leiste", "creiste",
    "pediste", "notaste", "cambiaste", "borraste", "pegaste", "editaste",
]


# ============================================================
# BLOCK 1 - Typos
# ============================================================

class TypoOps:

    def _qwerty_candidates(self, text: str) -> List[int]:
        chars = list(text)
        return [
            i for i in range(1, len(chars) - 1)
            if chars[i].isalpha() and chars[i].lower() in QWERTY_NEIGHBORS
        ]

    def qwerty_once(self, text: str) -> str:
        chars = list(text)
        cand = self._qwerty_candidates(text)
        if not cand:
            return text
        idx = random.choice(cand)
        base = chars[idx].lower()
        repl = random.choice(QWERTY_NEIGHBORS[base])
        chars[idx] = repl.upper() if chars[idx].isupper() else repl
        return ''.join(chars)

    def omission_once(self, text: str, vowel_bias: float = 0.8) -> str:
        chars = list(text)
        if len(chars) < 3:
            return text
        vowels = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
        vowel_cand = [i for i, c in enumerate(chars) if c in vowels]
        alpha_cand = [i for i, c in enumerate(chars) if c.isalpha()]
        if not alpha_cand:
            return text
        idx = random.choice(vowel_cand) if (vowel_cand and random.random() < vowel_bias) else random.choice(alpha_cand)
        del chars[idx]
        return ''.join(chars)

    def abbr_once(self, text: str, weight_q: float, weight_pq: float) -> str:
        candidates: List[Tuple[str, float]] = []
        if re.search(r'\bque\b', text, flags=re.IGNORECASE):
            candidates.append(("q", max(0.0, float(weight_q))))
        if re.search(r'\bporque\b', text, flags=re.IGNORECASE) or re.search(r'\bpor\s+que\b', text, flags=re.IGNORECASE):
            candidates.append(("pq", max(0.0, float(weight_pq))))
        if not candidates:
            return text

        ops = [c[0] for c in candidates]
        ws = [c[1] for c in candidates]
        op = weighted_choice(ops, ws)

        if op == "q":
            return re.sub(r'\bque\b', 'q', text, flags=re.IGNORECASE, count=1)

        t = re.sub(r'\bpor\s+que\b', 'pq', text, flags=re.IGNORECASE, count=1)
        t = re.sub(r'\bporque\b', 'pq', t, flags=re.IGNORECASE, count=1)
        return t

    def remove_space_once(self, text: str) -> str:
        if ' ' not in text:
            return text

        chars = list(text)
        candidates = [
            i for i in range(1, len(chars) - 1)
            if chars[i] == ' ' and chars[i - 1] != ' ' and chars[i + 1] != ' '
        ]
        if not candidates:
            return text

        idx = random.choice(candidates)
        del chars[idx]
        return ''.join(chars)


def apply_typos_weighted_exact(
    text: str,
    n_typos: int,
    ops: TypoOps,
    typo_type_weights: Dict[str, float],
    vowel_delete_bias: float,
    abbr_q_weight: float,
    abbr_pq_weight: float,
    max_attempts: int = 120
) -> str:

    applied = 0
    attempts = 0
    type_names = ["qwerty", "omission", "abbr", "space_remove"]

    while applied < n_typos and attempts < max_attempts:
        attempts += 1
        before = text

        ws = [
            float(typo_type_weights.get("qwerty", 0.5)),
            float(typo_type_weights.get("omission", 0.3)),
            float(typo_type_weights.get("abbr", 0.2)),
            float(typo_type_weights.get("space_remove", 0.0)),
        ]
        tname = weighted_choice(type_names, ws)

        if tname == "qwerty":
            text = ops.qwerty_once(text)
        elif tname == "omission":
            text = ops.omission_once(text, vowel_bias=vowel_delete_bias)
        elif tname == "abbr":
            text = ops.abbr_once(text, weight_q=abbr_q_weight, weight_pq=abbr_pq_weight)
        else:
            text = ops.remove_space_once(text)

        # fallback if nothing changed
        if text == before:
            fallback = type_names[:]
            random.shuffle(fallback)
            for t2 in fallback:
                cand = text
                if t2 == "qwerty":
                    cand = ops.qwerty_once(text)
                elif t2 == "omission":
                    cand = ops.omission_once(text, vowel_bias=vowel_delete_bias)
                elif t2 == "abbr":
                    cand = ops.abbr_once(text, weight_q=abbr_q_weight, weight_pq=abbr_pq_weight)
                else:
                    cand = ops.remove_space_once(text)

                if cand != text:
                    text = cand
                    break

        if text != before:
            applied += 1
        else:
            break

    return text


def normalize_block1(text: str, accents_drop_prob: float) -> str:
    text = re.sub(r'¿', '', text)
    if random.random() < accents_drop_prob:
        text = strip_accents(text)
    return text


# ============================================================
# BLOCK 2 - Grammatical errors
# ============================================================

GrammarRule = Callable[[str], str]


class GrammarRules:

    def __init__(self):
        # Bidirectional homophones
        self.homophone_pairs = [
            (r'\bhecho\b', 'echo'),
            (r'\becho\b', 'hecho'),

            (r'\bvaya\b', 'valla'),
            (r'\bvalla\b', 'vaya'),

            (r'\bhaber\b', 'a ver'),
            (r'\ba ver\b', 'haber'),

            (r'\bhay\b', 'ay'),
            (r'\bay\b', 'hay'),

            (r'\boye\b', 'olle'),
            (r'\bolle\b', 'oye'),
        ]

        self.porque_pairs = [
            (r'\bporque\b', 'por que'),
            (r'\bpor\s+que\b', 'porque'),
            (r'\bpor qué\b', 'porque'),
            (r'\bporqué\b', 'porque'),
        ]

    def habia_to_habian(self, text: str) -> str:
        t = strip_accents(text)
        if re.search(r'\bhabia\b', t, flags=re.IGNORECASE):
            base = strip_accents(text)
            return re.sub(r'\bhabia\b', 'habian', base, flags=re.IGNORECASE, count=1)
        return text

    def hemos_to_habemos(self, text: str) -> str:
        if re.search(r'\bhemos\b', text, flags=re.IGNORECASE):
            return re.sub(r'\bhemos\b', 'habemos', text, flags=re.IGNORECASE, count=1)
        return text

    def homophones(self, text: str) -> str:
        for pat, repl in self.homophone_pairs:
            if re.search(pat, text, flags=re.IGNORECASE):
                return re.sub(pat, repl, text, flags=re.IGNORECASE, count=1)
        return text

    def porque(self, text: str) -> str:
        for pat, repl in self.porque_pairs:
            if re.search(pat, text, flags=re.IGNORECASE):
                return re.sub(pat, repl, text, flags=re.IGNORECASE, count=1)
        return text

    def seseo_ceceo(self, text: str, max_replacements: int = 2) -> str:
        # Bidirectional changes: s/z/c confusion
        pairs = [
            # seseo
            (r'za', 'sa'), (r'zo', 'so'), (r'zu', 'su'),
            (r'ce', 'se'), (r'ci', 'si'),
            # ceceo
            (r'sa', 'za'), (r'so', 'zo'), (r'su', 'zu'),
            (r'se', 'ce'), (r'si', 'ci'),
        ]

        made = 0
        for pat, repl in pairs:
            if made >= max_replacements:
                break
            if re.search(pat, text, flags=re.IGNORECASE):
                text = re.sub(pat, repl, text, flags=re.IGNORECASE, count=1)
                made += 1
        return text

    def preterite_s(self, text: str) -> str:
        t = strip_accents(text)
        earliest: Optional[Tuple[int, str]] = None
        for v in COMMON_PRETERITE_2SG:
            m = re.search(rf'\b{re.escape(v)}\b', t, flags=re.IGNORECASE)
            if m:
                if earliest is None or m.start() < earliest[0]:
                    earliest = (m.start(), v)
        if earliest is None:
            return text
        verb = earliest[1]
        base = strip_accents(text)
        return re.sub(rf'\b{re.escape(verb)}\b', verb + "s", base, flags=re.IGNORECASE, count=1)

    def drop_initial_h(self, text: str) -> str:
        pattern = re.compile(r'\b([hH])([A-Za-zÁÉÍÓÚÜáéíóúüÑñ])')
        m = pattern.search(text)
        if not m:
            return text
        start, end = m.span(1)
        return text[:start] + text[end:]

    def swap_bv(self, text: str) -> str:
        m = re.search(r'[bB]', text)
        if m:
            i = m.start()
            c = text[i]
            repl = 'v' if c == 'b' else 'V'
            return text[:i] + repl + text[i+1:]
        m = re.search(r'[vV]', text)
        if m:
            i = m.start()
            c = text[i]
            repl = 'b' if c == 'v' else 'B'
            return text[:i] + repl + text[i+1:]
        return text

    def registry(self) -> Dict[str, GrammarRule]:
        return {
            "habia_to_habian": self.habia_to_habian,
            "hemos_to_habemos": self.hemos_to_habemos,
            "homophones": self.homophones,
            "porque": self.porque,
            "seseo_ceceo": self.seseo_ceceo,
            "preterite_s": self.preterite_s,
            "drop_initial_h": self.drop_initial_h,
            "swap_bv": self.swap_bv,
        }


def normalize_block2(text: str) -> str:
    text = strip_accents(text)
    text = re.sub(r'¿', '', text)
    text = re.sub(r',', '', text)
    return text


def apply_grammar_ordered(
    text: str,
    n_changes: int,
    rule_order: List[str],
    rule_registry: Dict[str, GrammarRule]
) -> str:

    applied = 0
    for name in rule_order:
        if applied >= n_changes:
            break
        fn = rule_registry[name]
        new_text = fn(text)
        if new_text != text:
            text = new_text
            applied += 1
    return text


# ============================================================
# BLOCK 3 - Custom
# ============================================================

@dataclass
class CustomConfig:
    """
    Configuración personalizada para el proceso de corrupción de prompts (Bloque 3).
    
    Esta clase permite a los participantes del hackathon ajustar el número de errores
    (tipográficos y gramaticales) y la probabilidad de cada tipo de error. 
    ¡Juega con estos pesos para ver cómo el modelo maneja diferentes niveles de ruido!
    """

    n_typos: int = 2
    n_grammar_changes: int = 2

    typo_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "qwerty": 0.5,
        "omission": 0.3,
        "abbr": 0.2,
        "space_remove": 0.0,  # enable by raising this weight
    })
    vowel_delete_bias: float = 0.9
    abbr_q_weight: float = 0.6
    abbr_pq_weight: float = 0.4

    grammar_rule_weights: Dict[str, float] = field(default_factory=lambda: {
        "habia_to_habian": 1.0,
        "hemos_to_habemos": 0.7,
        "homophones": 0.7,
        "porque": 0.9,
        "seseo_ceceo": 0.4,
        "preterite_s": 0.3,
        "drop_initial_h": 0.3,
        "swap_bv": 0.2
    })

    remove_open_questions: bool = True
    strip_accents: bool = True
    remove_commas: bool = True
    lowercase: bool = True


def load_custom_config(path: Optional[str]) -> CustomConfig:
    if not path:
        return CustomConfig()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return CustomConfig()

    allowed = set(CustomConfig.__dataclass_fields__.keys())
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ValueError(
            f"Unknown keys in custom config: {sorted(list(unknown))}. "
            f"Allowed: {sorted(list(allowed))}"
        )
    return CustomConfig(**data)


def normalize_custom(text: str, cfg: CustomConfig) -> str:

    if cfg.remove_open_questions:
        text = re.sub(r'¿', '', text)
    if cfg.strip_accents:
        text = strip_accents(text)
    if cfg.remove_commas:
        text = re.sub(r',', '', text)
    if cfg.lowercase:
        text = text.lower()
    return text


def apply_grammar_weighted(
    text: str,
    n_changes: int,
    rule_registry: Dict[str, GrammarRule],
    weights_by_rule: Dict[str, float],
    max_attempts: int = 120
) -> str:

    applied = 0
    attempts = 0

    while applied < n_changes and attempts < max_attempts:
        attempts += 1

        applicable: List[Tuple[str, str]] = []
        weights: List[float] = []

        for name, fn in rule_registry.items():
            new_text = fn(text)
            if new_text != text:
                applicable.append((name, new_text))
                weights.append(float(weights_by_rule.get(name, 1.0)))

        if not applicable:
            break

        chosen = weighted_choice(applicable, weights)
        _, new_text = chosen
        text = new_text
        applied += 1

    return text


# ============================================================
# Pipeline
# ============================================================

def process_prompts(
    prompts: List[str],
    custom_cfg: Optional[CustomConfig] = None,
    typos_range: Tuple[int, int] = (1, 2),
    grammar_range: Tuple[int, int] = (3, 4),
    typos_accents_drop_prob: float = 0.60
) -> List[Dict[str, str]]:
    """
    Función principal para generar variantes corruptas de una lista de prompts.
    
    En el contexto del hackathon para evaluar la robustez (Robustness), esta función
    toma los prompts originales y genera tres variantes:
    1. Errores tipográficos (Typos).
    2. Errores gramaticales predefinidos.
    3. Una combinación personalizada según `CustomConfig`.
    
    Args:
        prompts (List[str]): Lista de textos (prompts) originales.
        custom_cfg (CustomConfig, optional): Configuración para el Bloque 3.
        typos_range (Tuple[int, int]): Rango mínimo y máximo de errores tipográficos.
        grammar_range (Tuple[int, int]): Rango mínimo y máximo de errores gramaticales.
        typos_accents_drop_prob (float): Probabilidad de eliminar tildes (0 a 1).
        
    Returns:
        List[Dict[str, str]]: Lista de diccionarios con el prompt original y sus 3 variantes.
    """
    custom_cfg = custom_cfg or CustomConfig()

    typo_ops = TypoOps()
    grammar = GrammarRules()
    reg = grammar.registry()

    block2_order = [
        "habia_to_habian",
        "hemos_to_habemos",
        "homophones",
        "porque",
        "seseo_ceceo",
        "preterite_s",
        "drop_initial_h",
        "swap_bv",
    ]

    out: List[Dict[str, str]] = []

    for prompt in prompts:

        # BLOCK 1
        n_typos_block1 = random.randint(typos_range[0], typos_range[1])

        prompt_typos = apply_typos_weighted_exact(
            prompt,
            n_typos=int(n_typos_block1),
            ops=typo_ops,
            typo_type_weights={"qwerty": 0.55, "omission": 0.4, "abbr": 0.4, "space_remove": 0.5},
            vowel_delete_bias=0.8,
            abbr_q_weight=0.6,
            abbr_pq_weight=0.4
        )
        prompt_typos = normalize_block1(prompt_typos, accents_drop_prob=typos_accents_drop_prob)

        # BLOCK 2
        n_grammar_block2 = random.randint(grammar_range[0], grammar_range[1])

        prompt_grammatical = normalize_block2(prompt)
        prompt_grammatical = apply_grammar_ordered(
            prompt_grammatical,
            n_changes=int(n_grammar_block2),
            rule_order=block2_order,
            rule_registry=reg
        )
        if re.search(r'\bhabia\b', prompt_grammatical, flags=re.IGNORECASE):
            prompt_grammatical = re.sub(r'\bhabia\b', 'habian', prompt_grammatical, flags=re.IGNORECASE)

        # BLOCK 3
        prompt_custom = prompt

        # 3A) Grammar
        if custom_cfg.n_grammar_changes > 0:
            prompt_custom = apply_grammar_weighted(
                prompt_custom,
                n_changes=int(custom_cfg.n_grammar_changes),
                rule_registry=reg,
                weights_by_rule=custom_cfg.grammar_rule_weights
            )

        # 3B) Typos
        if custom_cfg.n_typos > 0:
            prompt_custom = apply_typos_weighted_exact(
                prompt_custom,
                n_typos=int(custom_cfg.n_typos),
                ops=typo_ops,
                typo_type_weights=custom_cfg.typo_type_weights,
                vowel_delete_bias=custom_cfg.vowel_delete_bias,
                abbr_q_weight=custom_cfg.abbr_q_weight,
                abbr_pq_weight=custom_cfg.abbr_pq_weight
            )

        # 3C) Normalization
        prompt_custom = normalize_custom(prompt_custom, cfg=custom_cfg)

        out.append({
            "prompt_original": prompt,
            "prompt_typos": prompt_typos,
            "prompt_grammatical_errors": prompt_grammatical,
            "prompt_custom": prompt_custom
        })

    return out


# ============================================================
# I/O (JSON / CSV)
# ============================================================

def process_json(input_path: str, output_path: str, custom_cfg: Optional[CustomConfig] = None, input_col: str = "prompt"):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [item[input_col] for item in data]
    results = process_prompts(prompts, custom_cfg=custom_cfg)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def process_csv(input_path: str, output_path: str, custom_cfg: Optional[CustomConfig] = None, input_col: str = "prompt"):
    df = pd.read_csv(input_path)
    if input_col not in df.columns:
        raise ValueError(f"CSV must contain a {input_col} column. Found: {list(df.columns)}")
    results = process_prompts(df[input_col].tolist(), custom_cfg=custom_cfg)
    pd.DataFrame(results).to_csv(output_path, index=False)


# ============================================================
# CLI
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate typos, grammatical errors, and a custom corruption per prompt.")
    p.add_argument("--input_json", type=str, default=None)
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--input_csv", type=str, default=None)
    p.add_argument("--output_csv", type=str, default=None)
    p.add_argument("--custom_config", type=str, default=None, help="YAML config for prompt_custom")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--input_col", type=str, default="prompt")
    return p


def main():
    args = build_argparser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    custom_cfg = load_custom_config(args.custom_config)

    if args.input_json and args.output_json and not args.input_csv and not args.output_csv:
        process_json(args.input_json, args.output_json, custom_cfg=custom_cfg, input_col = args.input_col)
        return

    if args.input_csv and args.output_csv and not args.input_json and not args.output_json:
        process_csv(args.input_csv, args.output_csv, custom_cfg=custom_cfg, input_col = args.input_col)
        return

    raise SystemExit("Provide either --input_json/--output_json OR --input_csv/--output_csv (but not both).")


if __name__ == "__main__":
    main()






