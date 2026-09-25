# Oracle Bone Script model

Place the local 109-class classifier at:

```text
models/oracle/best_ge50.pt
```

The model file is intentionally excluded from Git. It can also be stored elsewhere by setting `ORACLE_MODEL_PATH` in
the project root `.env` file.

`best_portable.pt` is the legacy 39-class checkpoint. Keep it as a rollback asset; the
dataset pipeline in `scripts/` transfers its learned features to the 109-class model.
