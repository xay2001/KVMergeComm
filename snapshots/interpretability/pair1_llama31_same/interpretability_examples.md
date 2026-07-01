# Interpretability Examples

## hotpotqa idx=6

Question: Thomas Fitch defended John Henry Holliday when he was accused of killing men during the gunfight at the o.k. corral, when was that?

Answers: ['October 25, 1881']

ReKV top tokens: <|begin_of_text|>, <|end_header_id|>,   , assistant, 25, .,  October,   , <|eot_id|>, <|start_header_id|>

Evict top tokens:  Holl,  Ear,  Holl,  Ear,  Cor, Doc, 188,  Frank, 185,  gun

Overlap: {'rekv': {'answer_terms': ['1881', '25', 'october'], 'hit_terms': ['25', 'october'], 'hit_count': 2, 'answer_term_count': 3, 'recall': 0.666667}, 'evict': {'answer_terms': ['1881', '25', 'october'], 'hit_terms': [], 'hit_count': 0, 'answer_term_count': 3, 'recall': 0.0}, 'random': {'answer_terms': ['1881', '25', 'october'], 'hit_terms': [], 'hit_count': 0, 'answer_term_count': 3, 'recall': 0.0}}

## multifieldqa_en idx=0

Question: Question: What is the name of the most active fan club?


Answers: ['South West Ultras fan club.']

ReKV top tokens: <|begin_of_text|>, ., <|end_header_id|>,   , assistant,  club, <|start_header_id|>, <|eot_id|>,   , ,

Evict top tokens: ere, inter,  merger,  goal,  youth, acad,  Welt,  mini, Person, ial

Overlap: {'rekv': {'answer_terms': ['club', 'fan', 'south', 'ultras', 'west'], 'hit_terms': ['club', 'south'], 'hit_count': 2, 'answer_term_count': 5, 'recall': 0.4}, 'evict': {'answer_terms': ['club', 'fan', 'south', 'ultras', 'west'], 'hit_terms': ['fan'], 'hit_count': 1, 'answer_term_count': 5, 'recall': 0.2}, 'random': {'answer_terms': ['club', 'fan', 'south', 'ultras', 'west'], 'hit_terms': ['club'], 'hit_count': 1, 'answer_term_count': 5, 'recall': 0.2}}

## musique idx=1

Question: What university did the author of 1967: The Last Good Year attend?

Answers: ['University of British Columbia', 'The University of British Columbia']

ReKV top tokens: <|begin_of_text|>, <|end_header_id|>,  of, assistant,   ,  Columbia,   , <|eot_id|>, ., <|start_header_id|>

Evict top tokens: 7, 196,  Good, ond,  up,  Cent,  title,  Last,  Canada,  Turning

Overlap: {'rekv': {'answer_terms': ['british', 'columbia', 'of', 'the', 'university'], 'hit_terms': ['columbia', 'of'], 'hit_count': 2, 'answer_term_count': 5, 'recall': 0.4}, 'evict': {'answer_terms': ['british', 'columbia', 'of', 'the', 'university'], 'hit_terms': [], 'hit_count': 0, 'answer_term_count': 5, 'recall': 0.0}, 'random': {'answer_terms': ['british', 'columbia', 'of', 'the', 'university'], 'hit_terms': ['the'], 'hit_count': 1, 'answer_term_count': 5, 'recall': 0.2}}

