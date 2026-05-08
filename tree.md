.
├── .claude/
│   ├── CLAUDE.md
│   ├── agents/
│   │   └── ml-python-expert.md
│   ├── docs/
│   │   ├── architecture.md
│   │   ├── tech-stack.md
│   │   ├── decisions/
│   │   │   └── README.md
│   │   └── features/
│   │       ├── feature-1-hand-landmarks.md
│   │       ├── feature-2-data-collection.md
│   │       ├── feature-3-classifier.md
│   │       ├── feature-4-debounce.md
│   │       ├── feature-5-translation.md
│   │       ├── feature-6-tts.md
│   │       ├── feature-7-lcd.md
│   │       ├── feature-8-buttons.md
│   │       ├── feature-9-language-menu.md
│   │       └── feature-10-orchestration.md
│   ├── plans/
│   │   ├── PLAN.md
│   │   ├── current-task.md
│   │   ├── plan_zip.md
│   │   └── phases/
│   │       ├── phase-1-data-collection.md
│   │       └── phase-2-training.md
│   └── settings.local.json
├── issue/
│   ├── 001-windows-defender-throttles-bulk-npy-load.md
│   └── README.md
├── scripts/
│   ├── __init__.py
│   ├── collect.py
│   ├── ingest_public.py
│   └── setup_models.py
├── src/
│   └── asl_live/
│       ├── __init__.py
│       ├── config.py
│       ├── recognition/
│       │   ├── __init__.py
│       │   ├── classifier.py
│       │   └── landmarks.py
│       └── train/
│           ├── __init__.py
│           ├── augment.py
│           ├── dataset.py
│           ├── export.py
│           ├── metrics.py
│           ├── model.py
│           └── train_mlp.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_augment.py
│   ├── test_classifier.py
│   ├── test_dataset.py
│   ├── test_export.py
│   ├── test_landmarks.py
│   ├── test_metrics.py
│   └── test_model.py
├── .gitignore
├── README.md
├── pyproject.toml
└── tree.md
