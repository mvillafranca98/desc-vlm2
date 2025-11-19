#!/usr/bin/env python3
"""Quick utility to verify LanceDB Cloud connection and tables."""

import os
import sys
import lancedb


def main():
    project_slug = os.getenv("LANCEDB_PROJECT_SLUG", "descvlm2-lnh0lv")
    api_key = os.getenv("LANCEDB_API_KEY")
    if not api_key:
        print("⚠️  Error: LANCEDB_API_KEY not set. Please set it in .env file or environment variables.")
        sys.exit(1)
    region = os.getenv("LANCEDB_REGION", "us-east-1")

    if not project_slug or not api_key:
        print("LanceDB credentials are missing.")
        print("Set LANCEDB_PROJECT_SLUG and LANCEDB_API_KEY before running this script.")
        sys.exit(1)

    print("Connecting to LanceDB Cloud...")
    db = lancedb.connect(uri=f"db://{project_slug}", api_key=api_key, region=region)

    tables = db.table_names()
    if tables:
        print("Available tables:")
        for name in tables:
            print(f" - {name}")
    else:
        print("No tables found in this project yet.")


if __name__ == "__main__":
    main()