# Clean Interpretability Examples

## hotpotqa idx=35

Question: D1NZ is a series based on what oversteering technique?

Answers: ['Drifting']

Plot: `snapshots/interpretability/pair1_llama31_same/cleaned/hotpotqa_clean_top_tokens.png`

ReKV clean top tokens: drifting, ifting, technique, driving, is, where

Evict clean top tokens: ste, all, Dr, driving, ifting, maintaining, intentionally, ers, technique, control

Overlap: {'rekv': {'answer_terms': ['drifting'], 'hit_terms': ['drifting'], 'hit_count': 1, 'answer_term_count': 1, 'recall': 1.0}, 'evict': {'answer_terms': ['drifting'], 'hit_terms': [], 'hit_count': 0, 'answer_term_count': 1, 'recall': 0.0}, 'random': {'answer_terms': ['drifting'], 'hit_terms': [], 'hit_count': 0, 'answer_term_count': 1, 'recall': 0.0}}

## musique idx=40

Question: How many students attend Daniel Thürer's university?

Answers: ['nearly 25,000', 'University of Zurich']

Plot: `snapshots/interpretability/pair1_llama31_same/cleaned/musique_clean_top_tokens.png`

ReKV clean top tokens: 000, nearly, with, students, Zurich, 25

Evict clean top tokens: us, Fern, 185, em, 146, EP, cant, University, er, Law

Overlap: {'rekv': {'answer_terms': ['000', '25', 'nearly', 'of', 'university', 'zurich'], 'hit_terms': ['000', '25', 'nearly', 'zurich'], 'hit_count': 4, 'answer_term_count': 6, 'recall': 0.666667}, 'evict': {'answer_terms': ['000', '25', 'nearly', 'of', 'university', 'zurich'], 'hit_terms': ['university'], 'hit_count': 1, 'answer_term_count': 6, 'recall': 0.166667}, 'random': {'answer_terms': ['000', '25', 'nearly', 'of', 'university', 'zurich'], 'hit_terms': [], 'hit_count': 0, 'answer_term_count': 6, 'recall': 0.0}}

## multifieldqa_en idx=27

Question: Question: What types of sensors are now capable of estimating physical activity levels and physiological outcomes of older adults?

Answers: ['Wearable sensors.']

Plot: `snapshots/interpretability/pair1_llama31_same/cleaned/multifieldqa_en_clean_top_tokens.png`

ReKV clean top tokens: able, sensors, text, and, wearable, Wear

Evict clean top tokens: mo, post, hab, Imp, gest, us, sequential, est, ical, FP

Overlap: {'rekv': {'answer_terms': ['sensors', 'wearable'], 'hit_terms': ['sensors', 'wearable'], 'hit_count': 2, 'answer_term_count': 2, 'recall': 1.0}, 'evict': {'answer_terms': ['sensors', 'wearable'], 'hit_terms': [], 'hit_count': 0, 'answer_term_count': 2, 'recall': 0.0}, 'random': {'answer_terms': ['sensors', 'wearable'], 'hit_terms': [], 'hit_count': 0, 'answer_term_count': 2, 'recall': 0.0}}
