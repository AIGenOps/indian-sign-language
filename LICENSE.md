# Project License

Original project engineering code/documents may be licensed under Apache-2.0 by the project owner.

Third-party model weights, datasets, libraries and source repositories retain their own licenses.
Review each license before redistribution or commercial deployment.

## Known third-party licenses (last checked 2026-09-27)
- Sanket - ISL Translator weights (`ayush2635/sanket-isl-translator` on Hugging
  Face): tagged **MIT** on the Hub as of this check. Re-verify at
  https://huggingface.co/ayush2635/sanket-isl-translator before redistribution,
  since a model card's license tag can be changed by its owner after the fact
  and this file only records what was observed at check time.
- `facebook/timesformer-base-finetuned-k400` (vision encoder backbone) and
  `t5-base` (language decoder backbone): each carries its own Hugging Face
  license; check their respective model cards directly rather than assuming
  they inherit Sanket's MIT tag.
- INCLUDE, ISLTranslate/iSign, CISLR and any other dataset: none of these
  automatically share a license with the model weights or with each other.
  See `docs/17_DATASET_REGISTRY.md`.
- ISLTranslate / iSign dataset (`Exploration-Lab/ISLTranslate` on GitHub and
  `Exploration-Lab/iSign` on Hugging Face): tagged **CC BY-NC-SA 4.0** (Creative
  Commons Attribution-NonCommercial-ShareAlike 4.0 International). Non-commercial
  and share-alike restrictions apply to any benchmark results, evaluations, or
  derived artifacts produced from this dataset, separate from the Sanket model
  weights' MIT license.

