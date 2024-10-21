import argparse
import pandas as pd
import os
import time

from retriever import ColumnRetriever
from matcher import ColumnMatcher
from utils import get_dataset_paths, process_tables, get_samples
from evaluation import evaluate_matches, convert_to_valentine_format

API_KEY = "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"


class RetrieveMatch:
    def __init__(self, model_type, dataset, serialization):
        self.retriever = ColumnRetriever(
            model_type=model_type, dataset=dataset, serialization=serialization
        )
        self.matcher = ColumnMatcher(api_key=API_KEY)

    def match(self, source_tables_path, target_tables_path, source_path, top_k, cand_k):
        source_table = pd.read_csv(os.path.join(source_tables_path, source_path))
        target_path = source_path.replace("source", "target")
        target_table = pd.read_csv(os.path.join(target_tables_path, target_path))
        source_table, target_table = process_tables(source_table, target_table)

        start_time = time.time()
        source_values = {
            col: get_samples(source_table[col]) for col in source_table.columns
        }
        target_values = {
            col: get_samples(target_table[col]) for col in target_table.columns
        }
        matched_columns = self.retriever.find_matches(
            source_table, target_table, source_values, target_values, top_k
        )
        print("Matched Columns:", matched_columns)

        if cand_k > 1:
            matched_columns = self.matcher.rematch(source_table, matched_columns, top_k)
            print("Refined Matches:", matched_columns)
        runtime = time.time() - start_time

        converted_matches = convert_to_valentine_format(
            matched_columns,
            source_path.replace(".csv", ""),
            target_path.replace(".csv", ""),
        )
        return converted_matches, runtime


def run_retrieve_match(args):
    source_tables_path, target_tables_path, gt_path = get_dataset_paths(args.dataset)
    rema = RetrieveMatch(args.model_type, args.dataset, args.serialization)

    gt_df = pd.read_csv(gt_path)
    columns = [
        "usecase",
        "tablename",
        "top_k",
        "runtime",
        "Precision",
        "F1Score",
        "Recall",
        "PrecisionTop10Percent",
        "RecallAtSizeofGroundTruth",
        "MRR",
        "one2one_Precision",
        "one2one_F1Score",
        "one2one_Recall",
        "one2one_PrecisionTop10Percent",
        "one2one_RecallAtSizeofGroundTruth",
    ]

    if args.dataset not in ["gdc"]:
        results = []
        for source_path in os.listdir(source_tables_path):
            matches, runtime = rema.match(
                source_tables_path,
                target_tables_path,
                source_path,
                args.top_k,
                args.cand_k,
            )
            gt_rows = gt_df[gt_df["source_tab"] == source_path.split(".")[0]]
            ground_truth = [
                (row["source_col"], row["target_col"]) for _, row in gt_rows.iterrows()
            ]

            print("Ground Truth:", ground_truth)
            metrics = evaluate_matches(matches, ground_truth)
            print("Metrics:", metrics)

            metrics.update(
                {
                    "usecase": args.dataset,
                    "tablename": source_path.split(".")[0],
                    "top_k": args.top_k,
                    "runtime": runtime * 1000,
                }
            )

            results.append(metrics)

        all_metrics = pd.DataFrame(results, columns=columns, index=None)
        print("All Metrics:", all_metrics)

        avg_metrics = all_metrics.mean(numeric_only=True)
        print("Average Metrics:", avg_metrics)

        # Save to CSV
        filename = f"{args.dataset}_{args.model_type}_{args.serialization}_{args.top_k}_{args.cand_k}.csv"
        all_metrics.to_csv(filename)


def main():
    parser = argparse.ArgumentParser(
        description="Match columns between source and target tables using pretrained models."
    )
    parser.add_argument(
        "--dataset",
        default="chembl-joinable",
        help="Name of the dataset for model customization",
    )
    parser.add_argument(
        "--model_type",
        default="roberta-zs",
        help="Type of model (roberta-zs, roberta-ft, mpnet-zs, mpnet-ft, arctic-zs, arctic-ft)",
    )
    parser.add_argument(
        "--serialization",
        default="header_values",
        help="Column serialization method (header, header_values_default, header_values_prefix, header_values_repeat)",
    )
    parser.add_argument(
        "--top_k", type=int, default=20, help="Number of top matches to return"
    )
    parser.add_argument(
        "--cand_k",
        type=int,
        default=1,
        help="Number of candidate matches to refine",
    )

    args = parser.parse_args()
    run_retrieve_match(args)


if __name__ == "__main__":
    main()
