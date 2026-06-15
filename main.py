"""
Named Model Resolution — CLI entry point.

Usage:
    uv run python main.py \\
        --catalog-type file \\
        --catalog-path ./sample_data/ \\
        --spec pharma_knowledge_base/gold_layer_datamarts.csv \\
        --configs-dir pharma_knowledge_base/configs/ \\
        --output-dir ./output/

    # SQL connector example (Databricks, Postgres, etc.)
    uv run python main.py \\
        --catalog-type sql \\
        --connection-string "databricks+connector://token@<host>/<http_path>?catalog=hive_metastore&schema=gold" \\
        --schema gold \\
        --spec pharma_knowledge_base/gold_layer_datamarts.csv \\
        --configs-dir pharma_knowledge_base/configs/

    # Process specific datasets only
    uv run python main.py --catalog-type file --catalog-path ./data/ --datasets Gold_Rx_Claims Gold_Patient_Adherence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_connector(args: argparse.Namespace):
    if args.catalog_type == "file":
        from orchestrator.connectors import FileCatalogConnector

        return FileCatalogConnector(args.catalog_path)
    elif args.catalog_type == "sql":
        from sqlalchemy import create_engine

        from orchestrator.connectors import SQLCatalogConnector

        engine = create_engine(args.connection_string)
        return SQLCatalogConnector(engine, schema=args.schema)
    else:
        print(f"Unknown catalog type: {args.catalog_type}", file=sys.stderr)
        sys.exit(1)


def print_result_summary(result) -> None:
    print(f"\n{'─' * 60}")
    print(f"Dataset     : {result.dataset_name}")
    print(f"Table type  : {result.classification.table_type}")
    if result.classification.matched_catalog_entry:
        score = result.classification.catalog_match_score
        print(f"Catalog match: {result.classification.matched_catalog_entry} (score={score:.2f})")

    if result.model_configs:
        print("Model routing (ranked):")
        for mc in result.model_configs:
            print(f"  [{mc.confidence:.2f}] {mc.model_name}")
            for uc in mc.use_cases[:2]:
                print(f"         • {uc}")
            if mc.flagged_unclassified_columns:
                print(f"         unclassified metrics passed through: {mc.flagged_unclassified_columns}")
    else:
        print("Model routing: no model matched (table may be a dimension or unclassifiable)")

    if result.column_profiles:
        flagged = [p for p in result.column_profiles if p.suggested_transforms]
        if flagged:
            print("Transform suggestions:")
            for p in flagged:
                print(f"  {p.name}: {'; '.join(p.suggested_transforms)}")

    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  ⚠  {w}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Data-aware model router")
    parser.add_argument("--catalog-type", choices=["file", "sql"], default="file")
    parser.add_argument("--catalog-path", default=None, help="Directory of CSV/Parquet files (file connector)")
    parser.add_argument("--connection-string", default=None, help="SQLAlchemy connection string (sql connector)")
    parser.add_argument("--schema", default=None, help="Database schema (sql connector)")
    parser.add_argument(
        "--spec",
        default="pharma_knowledge_base/gold_layer_datamarts.csv",
        help="Path to gold_layer_datamarts.csv",
    )
    parser.add_argument(
        "--configs-dir",
        default="pharma_knowledge_base/configs",
        help="Directory containing YAML configs",
    )
    parser.add_argument("--output-dir", default=None, help="Directory to write per-model configs as JSON")
    parser.add_argument("--datasets", nargs="*", default=None, help="Specific dataset names to process")
    parser.add_argument("--run-pipelines", action="store_true", help="Execute model pipelines after routing")

    args = parser.parse_args()

    connector = build_connector(args)

    from orchestrator.router import Router

    router = Router(
        connector=connector,
        spec_path=args.spec,
        configs_dir=args.configs_dir,
    )

    print(f"Routing datasets from: {args.catalog_type} connector")
    results = router.run(datasets=args.datasets)
    print(f"Processed {len(results)} dataset(s)\n")

    for result in results:
        print_result_summary(result)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for result in results:
            payload = {
                "dataset": result.dataset_name,
                "table_type": result.classification.table_type,
                "catalog_match": result.classification.matched_catalog_entry,
                "model_configs": [
                    {
                        "model": mc.model_name,
                        "confidence": mc.confidence,
                        "use_cases": mc.use_cases,
                        "join_keys": mc.join_keys,
                        "dimension_tables": mc.dimension_tables,
                        "flagged_unclassified_columns": mc.flagged_unclassified_columns,
                    }
                    for mc in result.model_configs
                ],
                "warnings": result.warnings,
            }
            out_file = out / f"{result.dataset_name}.json"
            out_file.write_text(json.dumps(payload, indent=2))
        print(f"\nConfigs written to: {args.output_dir}")

    if args.run_pipelines:
        from orchestrator.pipelines import PIPELINE_REGISTRY

        for result in results:
            if not result.model_configs:
                continue
            top_model = result.model_configs[0].model_name
            pipeline_cls = PIPELINE_REGISTRY.get(top_model)
            if pipeline_cls:
                print(f"\nRunning {top_model} pipeline on {result.dataset_name}...")
                pipeline = pipeline_cls()
                output = pipeline.run(connector, result)
                print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
