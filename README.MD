 python -m pip install --upgrade pip setuptools wheel

  pip install -e ".[dev,api,gpu]"

  python -m harting_anomaly.data.catalog

  The above command did something and created a .egg-info folder Understanding required.


  python scripts/prepare_demo_dataset.py \
    --product harting_Baseline-10E-M-s \
    --view top \
    --clean


    python scripts/train_patchcore.py \
    --product harting_Baseline-10E-M-s \
    --view top


python scripts/diagnose_patchcore.py     --product harting_baseline_10e_m_s     --view top     --source /home/tachyon/Anomaly-Detection-Module/data/source.avi     --runtime-frames 20
    