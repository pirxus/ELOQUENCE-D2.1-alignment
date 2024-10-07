
## Preprocessing

- train
    - [laughter] [noise] 
    - apostrophes
    - abbreviations: `l._a.`
    - incomplete-

- test
    - (%HESITATION)
    - uppercase
    - abbreviations L. A.
    - (uncertain)
    - (incomplete-)

- dev
    - (%HESITATION)
    - uppercase
    - abbreviations L. A.
    - (uncertain)
    - (incomplete-)

- strategy 1:
    - remove laughter, noise, hesitation
    - lowercase everything
    - convert abbreviations to forms without dots and underscores
    - de-parenthesise uncertain and incomplete words
    - keep incomplete words with a dash?

- strategy 2:
    - strategy one but with truecasing

- strategy 3:
    - figure out the eval and dev sets
    - unify the laughter/noise/hesitation tags
    - do strategy 2 without tag removal
