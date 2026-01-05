# Email RL System

A reinforcement learning system that learns to prioritize, classify, and recommend actions for emails based on human behavior patterns. Trained on the Enron email dataset with a **target of 95% accuracy**.

## What This Does

When a user receives an email, the system predicts:
- **Action**: Reply now, reply later, forward, archive, delete, or create task
- **Priority**: How urgently does this need attention (0-1 score)
- **Task Creation**: Should this become a tracked task item?
- **Context**: What information is relevant for handling this?

The RL agent learns by comparing its predictions against actual user behavior in the Enron dataset.

## Scoring Dimensions

Emails are scored across multiple dimensions:
- **People**: Sender importance, org level, relationship strength
- **Project**: Active project references, deadlines, user's role
- **Topic**: Meeting request, task assignment, decision needed, FYI
- **Task**: Deadlines, action items, deliverables mentioned
- **Action**: Expected response type based on all signals

## Training Pipeline (95% Accuracy Target)

| Stage | Method | Expected Accuracy |
|-------|--------|------------------|
| 1 | Supervised Fine-Tuning (SFT) | 65-70% |
| 2 | Reward Model Training | - |
| 3 | GRPO (DeepSeek algorithm) | 80-85% |
| 4 | DPO (Direct Preference Optimization) | 88-90% |
| 5 | Temporal RLHF (future emails as feedback) | 92-94% |
| 6 | Rejection Sampling Refinement | **95%+** |

## Hardware Requirements

- **Recommended**: Apple Silicon M4 Max with 128GB RAM
- Supports running 70B+ parameter models locally
- Uses MPS (Metal Performance Shaders) and MLX for acceleration

## Quick Start: Your Own Emails

Process your Gmail export and start labeling:

```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Start PostgreSQL
docker compose up -d postgres
# Or: ./scripts/start_db.sh

# 3. Set OpenAI API key (optional, for embeddings)
export OPENAI_API_KEY="sk-..."

# 4. Run the pipeline on your Gmail export
python run_pipeline.py /path/to/your-gmail.mbox

# 5. Start the labeling UI
streamlit run apps/labeling_ui.py
```

The pipeline will:
- Parse your MBOX file
- Import to PostgreSQL
- Discover projects from labels and participant patterns
- Cluster emails by semantic similarity
- Detect high-engagement periods

See [PIPELINE.md](./PIPELINE.md) for detailed documentation.

## Quick Start: Enron Dataset (Research)

```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Download Enron dataset
./scripts/download_enron.sh

# 3. Preprocess emails
python src/preprocess.py

# 4. Run full training pipeline
python src/train_full_pipeline.py --target_accuracy 0.95

# 5. Evaluate
python src/evaluate.py
```

## Documentation

See the [docs/](./docs/) folder for detailed documentation:

1. **[Dataset Setup](./docs/01-enron-dataset.md)** - Download and prepare Enron emails
2. **[System Architecture](./docs/02-architecture.md)** - RL system design
3. **[Feature Extraction](./docs/03-features.md)** - Scoring dimensions
4. **[Training Guide](./docs/04-training.md)** - Basic training
5. **[Advanced Training](./docs/05-advanced-training.md)** - Multi-stage RL for 95% accuracy

## Key Algorithms

- **GRPO** (Group Relative Policy Optimization) - DeepSeek's efficient alternative to PPO
- **DPO** (Direct Preference Optimization) - Anthropic-style direct alignment
- **KTO** (Kahneman-Tversky Optimization) - Works with unpaired preferences
- **Temporal RLHF** - Uses future emails as human feedback signal

## SurrealDB Integration (Optional)

For graph-based queries and efficient data storage, the system supports SurrealDB:

```bash
# Install SurrealDB
brew install surrealdb/tap/surreal

# Start database (using helper script)
./scripts/start_surreal.sh           # Start Enron database (default)
./scripts/start_surreal.sh gmail     # Or start Gmail database

# Initialize schema (in a new terminal while DB is running)
./scripts/init_surreal.sh            # Apply schema to Enron
./scripts/init_surreal.sh gmail      # Apply schema to Gmail

# Import data
python -m db.import_data enron data/train.json data/val.json data/test.json
python -m db.import_data gmail data/gmail_emails.json

# Connect to query data
surreal sql --endpoint http://127.0.0.1:8000 --user root --pass root --ns email --db enron

# Use SurrealDB-backed dataset in Python
from db import create_surreal_dataloaders
train, val, test = create_surreal_dataloaders(database='enron')
```

Benefits:
- **Graph queries**: Communication patterns between users
- **Cached embeddings**: Store embeddings in DB for faster loading
- **Efficient streaming**: Handle datasets larger than RAM
- **Separate databases**: Enron (training) and Gmail (personal validation)

## Project Structure

```
rl-emails/
├── README.md
├── requirements.txt
├── docs/                  # Documentation
├── scripts/               # Setup scripts
│   ├── download_enron.sh  # Download Enron dataset
│   ├── start_surreal.sh   # Start SurrealDB server
│   └── init_surreal.sh    # Initialize database schema
├── src/                   # Source code
│   ├── preprocess.py
│   ├── train.py
│   ├── train_full_pipeline.py
│   └── evaluate.py
├── db/                    # SurrealDB integration
│   ├── schema.surql       # Database schema
│   ├── import_data.py     # Data import script
│   ├── dataset.py         # PyTorch Dataset
│   └── benchmark.py       # Performance comparison
├── data/                  # Dataset (after download)
├── models/                # Downloaded LLMs
└── checkpoints/           # Training checkpoints
```

## License

MIT
