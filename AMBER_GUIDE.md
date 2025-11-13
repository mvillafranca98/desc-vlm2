# Amber - Video Scene Search Assistant

Amber is a GUI assistant that helps you search through your indexed video scenes using natural language queries.

## Features

- **Natural Language Queries**: Ask questions in plain English
- **Time-Based Filtering**: Query by time periods (yesterday, past 3 days, etc.)
- **Relevance Scoring**: Results are scored and ranked by relevance
- **Frame Display**: See up to 3 frames from each matching scene
- **Statistics**: Get summary stats about search results

## Usage

### Starting Amber

```bash
./run_amber.sh
```

Or directly:
```bash
python amber_assistant.py
```

### Example Queries

- "Do you have any records of hyenas in the past 3 days?"
- "Are there any records of lions yesterday?"
- "Show me scenes with blue aprons"
- "Find hyenas from 2 days ago"
- "What scenes contain lions?"

### Time Expressions Supported

- `yesterday`
- `today`
- `past 3 days` / `last 3 days`
- `past week` / `last week`
- `past month` / `last month`
- `X days ago` (e.g., "5 days ago")

### Understanding Results

Each result shows:
- **Relevance Score**: Composite score (0.0-1.0) based on:
  - Vector similarity (50% weight)
  - Keyword matching (30% weight)
  - Exact phrase match (20% weight)
  - Level weighting (caption > chunk > scene)
  - Frame count bonus
- **Vector Similarity**: Raw similarity from LanceDB search
- **Frames**: Up to 3 frames displayed per result

### Statistics Displayed

- Total number of matching records
- Average relevance score
- Highest relevance score
- Time range (if specified)

## Requirements

- Python 3.8+
- tkinter (usually included with Python)
- Pillow (for image display)
- All dependencies from `requirements.txt`

## Troubleshooting

**Amber says "Scene indexer is not enabled"**
- Check that LanceDB credentials are set in environment variables
- Verify the table name is correct

**Frames not displaying**
- Ensure frame files exist in the `frames/` directory
- Check file permissions

**No results found**
- Try broader search terms
- Check if scenes have been indexed recently
- Verify time constraints aren't too restrictive

