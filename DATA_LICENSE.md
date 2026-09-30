# Data license

The data files in this repository (`manifests/`, `splits/`, `transcripts/`,
`predictions/`, `artifacts/`, `results/`, `embeddings/*.json`, and the embeddings
release asset) are
derived from the BanglaMUSE corpus and are released under the
[Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

They contain the corpus's gold transcripts and sentiment labels, automatic
transcripts of its audio, features and embeddings computed from its recordings, and
model predictions and statistics. The audio itself is not redistributed.

When using these files, please cite the paper (see `README.md`) and the corpus:

- Md Darun Nayeem, Zarin Rafa, Tasnuva Tasnim Nova, Yasin Rahman, Abdul Mumeet Pathan,
  and Md Masudul Islam. 2026. BanglaMUSE: A multimodal Bangla sentiment dataset of
  text–audio pairs for speech and sentiment analysis. *Data in Brief*, 65:112458.
- Md Darun Nayeem, Zarin Rafa, Tasnuva Tasnim Nova, Yasin Rahman, Abdul Mumeet Pathan,
  and Nusrat Sultana. 2025. A multimodal Bangla text–audio dataset for sentiment
  analysis. Mendeley Data, Version 1. DOI: 10.17632/5yb4jjzrx3.1. CC BY 4.0.

Changes made to the original corpus are described in the paper (Appendix A.1) and
implemented in `scripts/audit_dataset.py`.
