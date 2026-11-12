# Third-party code

`baselines/M2FTrans/M2FTrans_v1` contains the Python source from the local M2FTrans v1 dependency used by FlowBraTS v2.1, including local compatibility/preprocessing edits. It is not claimed to be an unmodified upstream release.

- Upstream: https://github.com/Jun-Jie-Shi/M2FTrans
- Local checkout base commit: `03a0b28ae03ad1a7dea5fb874c7b47c1439caa34`
- License: [MIT](baselines/M2FTrans/LICENSE), retained verbatim.
- No upstream datasets, checkpoints, outputs or Git history are included.

FlowBraTS uses its encoder/decoder, data transforms and evaluation protocol. The legacy `/home/sjj/.../BRATS2020` string passed to the evaluator is an upstream dataset-dispatch identifier, not a data path to open. Historical upstream standalone scripts contain original paths; use the FlowBraTS entry points documented in README.
