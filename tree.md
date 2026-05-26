.
├── issue/
│   ├── 001-windows-defender-throttles-bulk-npy-load.md
│   └── README.md
├── scripts/
│   ├── __init__.py
│   ├── collect.py
│   ├── demo_recognition.py
│   ├── ingest_public.py
│   ├── setup_argos.py
│   ├── setup_models.py
│   └── setup_piper.py
├── src/
│   ├── asl_live/
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── buffer.py
│   │   │   └── main.py
│   │   ├── recognition/
│   │   │   ├── __init__.py
│   │   │   ├── classifier.py
│   │   │   ├── debounce.py
│   │   │   └── landmarks.py
│   │   ├── train/
│   │   │   ├── __init__.py
│   │   │   ├── augment.py
│   │   │   ├── dataset.py
│   │   │   ├── export.py
│   │   │   ├── metrics.py
│   │   │   ├── model.py
│   │   │   └── train_mlp.py
│   │   ├── translation/
│   │   │   ├── __init__.py
│   │   │   └── translator.py
│   │   ├── tts/
│   │   │   ├── __init__.py
│   │   │   └── speaker.py
│   │   ├── __init__.py
│   │   └── config.py
│   └── asl_live.egg-info/
│       ├── dependency_links.txt
│       ├── PKG-INFO
│       ├── requires.txt
│       ├── SOURCES.txt
│       └── top_level.txt
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_augment.py
│   ├── test_classifier.py
│   ├── test_dataset.py
│   ├── test_debounce.py
│   ├── test_export.py
│   ├── test_landmarks.py
│   ├── test_metrics.py
│   ├── test_model.py
│   ├── test_pipeline.py
│   ├── test_speaker.py
│   └── test_translator.py
├── .gitignore
├── index.html
├── pyproject.toml
├── README.md
└── tree.md
