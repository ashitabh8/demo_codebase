#!/usr/bin/env python3
"""
Experiment Manager

This script helps you quickly view and compare all experiments in the experiments directory.
It reads the experiment_summary.yaml files and displays them in a formatted table.

Usage:
    python experiment_manager.py [options]

Options:
    --experiments_dir PATH    Path to experiments directory (default: src2/experiments)
    --sort_by FIELD          Sort by field (default: experiment_id)
                             Options: experiment_id, model, dataset, best_val_accuracy, avg_val_std
    --filter_model MODEL     Filter by model name
    --filter_quantization    Show only quantization experiments
    --export CSV_FILE        Export results to CSV file
    --detailed               Show detailed information for each experiment
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime
import sys
from typing import List, Dict, Any
import csv


def load_experiment_summaries(experiments_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all experiment summaries from the experiments directory.
    
    Args:
        experiments_dir: Path to experiments directory
        
    Returns:
        List of experiment summary dictionaries
    """
    summaries = []
    
    if not experiments_dir.exists():
        print(f"Error: Experiments directory not found: {experiments_dir}")
        return summaries
    
    # Iterate through all subdirectories
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        # Skip special directories
        if exp_dir.name.startswith('.') or exp_dir.name in ['spectral_analysis', '__pycache__']:
            continue
        
        # Look for experiment_summary.yaml
        summary_file = exp_dir / "experiment_summary.yaml"
        
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = yaml.safe_load(f)
                    summaries.append(summary)
            except Exception as e:
                print(f"Warning: Could not load {summary_file}: {e}")
        else:
            # If no summary file, create a basic one from config.yaml
            config_file = exp_dir / "config.yaml"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                        summary = {
                            'experiment_id': exp_dir.name,
                            'model': config.get('model', 'Unknown'),
                            'model_variant': config.get('model_variant', 'N/A'),
                            'dataset': Path(config.get('yaml_path', 'Unknown')).stem,
                            'quantization_enabled': str(config.get('quantization', {}).get('enable', False)),
                            'quantization_method': config.get('quantization_method', 'N/A'),
                            'bitwidth_options': 'N/A',
                            'training_method': 'N/A',
                            'validation_function': 'N/A',
                            'num_epochs': config.get(config.get('model', 'ResNet'), {}).get('lr_scheduler', {}).get('train_epochs', 'N/A'),
                            'batch_size': config.get('batch_size', 'N/A'),
                            'learning_rate': 'N/A',
                            'optimizer': config.get(config.get('model', 'ResNet'), {}).get('optimizer', {}).get('name', 'N/A'),
                            'scheduler': config.get(config.get('model', 'ResNet'), {}).get('lr_scheduler', {}).get('name', 'N/A'),
                            'loss_function': config.get('loss_name', 'N/A'),
                            'final_train_accuracy': 'N/A',
                            'final_val_accuracy': 'N/A',
                            'best_val_accuracy': 'N/A',
                            'training_status': 'no_summary',
                        }
                        summaries.append(summary)
                except Exception as e:
                    print(f"Warning: Could not load {config_file}: {e}")
    
    return summaries


def filter_summaries(summaries: List[Dict[str, Any]], 
                     filter_model: str = None,
                     filter_quantization: bool = False) -> List[Dict[str, Any]]:
    """
    Filter experiment summaries based on criteria.
    
    Args:
        summaries: List of experiment summaries
        filter_model: Filter by model name
        filter_quantization: Show only quantization experiments
        
    Returns:
        Filtered list of summaries
    """
    filtered = summaries
    
    if filter_model:
        filtered = [s for s in filtered if s.get('model', '').lower() == filter_model.lower()]
    
    if filter_quantization:
        filtered = [s for s in filtered if str(s.get('quantization_enabled', 'False')).lower() == 'true']
    
    return filtered


def sort_summaries(summaries: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    """
    Sort experiment summaries by a field.
    
    Args:
        summaries: List of experiment summaries
        sort_by: Field to sort by
        
    Returns:
        Sorted list of summaries
    """
    def get_sort_key(summary):
        value = summary.get(sort_by, '')
        # Handle numeric fields
        if sort_by in ['best_val_accuracy', 'final_val_accuracy', 'final_train_accuracy', 'avg_val_std']:
            try:
                return float(value) if value != 'N/A' else -1 if 'accuracy' in sort_by else 999
            except (ValueError, TypeError):
                return -1 if 'accuracy' in sort_by else 999
        return str(value)
    
    # For accuracy fields, sort in descending order (higher is better)
    # For avg_val_std, sort in ascending order (lower is better - more stable)
    reverse = sort_by in ['best_val_accuracy', 'final_val_accuracy', 'final_train_accuracy']
    
    return sorted(summaries, key=get_sort_key, reverse=reverse)


def print_summary_table(summaries: List[Dict[str, Any]], detailed: bool = False):
    """
    Print experiment summaries in a formatted table.
    
    Args:
        summaries: List of experiment summaries
        detailed: Whether to show detailed information
    """
    if not summaries:
        print("No experiments found.")
        return
    
    print("\n" + "=" * 165)
    print(f"EXPERIMENT SUMMARY ({len(summaries)} experiments)")
    print("=" * 165)
    
    if detailed:
        # Detailed view - one experiment per block
        for i, summary in enumerate(summaries, 1):
            print(f"\n{i}. {summary.get('experiment_id', 'Unknown')}")
            print("-" * 100)
            print(f"  Model:                  {summary.get('model', 'N/A')} ({summary.get('model_variant', 'N/A')})")
            print(f"  Dataset:                {summary.get('dataset', 'N/A')}")
            print(f"  Status:                 {summary.get('training_status', 'N/A')}")
            print(f"  Quantization:           {summary.get('quantization_enabled', 'N/A')}")
            if str(summary.get('quantization_enabled', 'False')).lower() == 'true':
                print(f"  Quantization Method:    {summary.get('quantization_method', 'N/A')}")
                print(f"  Bitwidth Options:       {summary.get('bitwidth_options', 'N/A')}")
                print(f"  Training Method:        {summary.get('training_method', 'N/A')}")
                print(f"  Validation Function:    {summary.get('validation_function', 'N/A')}")
            print(f"  Epochs:                 {summary.get('num_epochs', 'N/A')}")
            print(f"  Batch Size:             {summary.get('batch_size', 'N/A')}")
            print(f"  Learning Rate:          {summary.get('learning_rate', 'N/A')}")
            print(f"  Optimizer:              {summary.get('optimizer', 'N/A')}")
            print(f"  Scheduler:              {summary.get('scheduler', 'N/A')}")
            print(f"  Loss Function:          {summary.get('loss_function', 'N/A')}")
            print(f"  Final Train Accuracy:   {summary.get('final_train_accuracy', 'N/A')}")
            print(f"  Final Val Accuracy:     {summary.get('final_val_accuracy', 'N/A')}")
            print(f"  Best Val Accuracy:      {summary.get('best_val_accuracy', 'N/A')}")
            
            # Display bitwidth bin statistics if available
            if 'bitwidth_bin_stats' in summary and summary['bitwidth_bin_stats']:
                bin_stats = summary['bitwidth_bin_stats']
                avg_val_std = summary.get('avg_val_std', 'N/A')
                print(f"  Average Val Std:        {avg_val_std}")
                print(f"  Bitwidth Bin Stats:     ({len(bin_stats)} bins)")
                for bin_stat in bin_stats:
                    print(f"    - BW {bin_stat['bitwidth']:.1f}bit: "
                         f"Acc={bin_stat['mean_acc']:.4f}±{bin_stat['std_acc']:.4f} "
                         f"(min={bin_stat['min_acc']:.4f}, max={bin_stat['max_acc']:.4f}, n={bin_stat['count']})")
            elif summary.get('avg_val_std', 'N/A') != 'N/A':
                print(f"  Average Val Std:        {summary.get('avg_val_std', 'N/A')}")
    else:
        # Compact table view
        # Header
        header = f"{'#':<4} {'Experiment ID':<30} {'Model':<15} {'Dataset':<12} {'Quant':<6} {'Method':<10} {'Bitwidths':<15} {'Epochs':<7} {'Best Val Acc':<13} {'Avg Val Std':<12} {'Status':<12}"
        print(f"\n{header}")
        print("-" * 165)
        
        # Rows
        for i, summary in enumerate(summaries, 1):
            exp_id = summary.get('experiment_id', 'Unknown')[:28]
            model = f"{summary.get('model', 'N/A')[:8]}/{summary.get('model_variant', 'N/A')[:6]}"
            dataset = summary.get('dataset', 'N/A')[:10]
            quant = 'Yes' if str(summary.get('quantization_enabled', 'False')).lower() == 'true' else 'No'
            method = summary.get('quantization_method', 'N/A')[:9]
            bitwidths = str(summary.get('bitwidth_options', 'N/A'))[:13]
            epochs = str(summary.get('num_epochs', 'N/A'))
            
            best_val_acc = summary.get('best_val_accuracy', 'N/A')
            if isinstance(best_val_acc, (int, float)):
                best_val_acc_str = f"{best_val_acc:.4f}"
            else:
                best_val_acc_str = str(best_val_acc)
            
            avg_val_std = summary.get('avg_val_std', 'N/A')
            if isinstance(avg_val_std, (int, float)):
                avg_val_std_str = f"{avg_val_std:.4f}"
            else:
                avg_val_std_str = str(avg_val_std)[:10]
            
            status = summary.get('training_status', 'N/A')[:11]
            
            row = f"{i:<4} {exp_id:<30} {model:<15} {dataset:<12} {quant:<6} {method:<10} {bitwidths:<15} {epochs:<7} {best_val_acc_str:<13} {avg_val_std_str:<12} {status:<12}"
            print(row)
    
    print("\n" + "=" * 165)
    print(f"\nTotal experiments: {len(summaries)}")
    
    # Statistics
    completed = sum(1 for s in summaries if s.get('training_status') == 'completed')
    quantized = sum(1 for s in summaries if str(s.get('quantization_enabled', 'False')).lower() == 'true')
    
    print(f"Completed: {completed}")
    print(f"Quantization enabled: {quantized}")
    
    # Best experiment
    valid_summaries = [s for s in summaries if isinstance(s.get('best_val_accuracy'), (int, float))]
    if valid_summaries:
        best_exp = max(valid_summaries, key=lambda s: s.get('best_val_accuracy', 0))
        print(f"\nBest experiment: {best_exp.get('experiment_id', 'Unknown')}")
        print(f"  Best Val Accuracy: {best_exp.get('best_val_accuracy', 'N/A'):.4f}")
        print(f"  Model: {best_exp.get('model', 'N/A')} ({best_exp.get('model_variant', 'N/A')})")
        if str(best_exp.get('quantization_enabled', 'False')).lower() == 'true':
            print(f"  Quantization: {best_exp.get('quantization_method', 'N/A')} with bitwidths {best_exp.get('bitwidth_options', 'N/A')}")
            if best_exp.get('avg_val_std', 'N/A') != 'N/A':
                print(f"  Average Val Std: {best_exp.get('avg_val_std', 'N/A'):.4f}")
    
    print("=" * 165 + "\n")


def export_to_csv(summaries: List[Dict[str, Any]], csv_file: str):
    """
    Export experiment summaries to CSV file.
    
    Args:
        summaries: List of experiment summaries
        csv_file: Path to output CSV file
    """
    if not summaries:
        print("No experiments to export.")
        return
    
    # Define fields to export
    fields = [
        'experiment_id', 'model', 'model_variant', 'dataset',
        'quantization_enabled', 'quantization_method', 'bitwidth_options',
        'training_method', 'validation_function',
        'num_epochs', 'batch_size', 'learning_rate',
        'optimizer', 'scheduler', 'loss_function',
        'final_train_accuracy', 'final_val_accuracy', 'best_val_accuracy',
        'avg_val_std', 'training_status'
    ]
    
    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summaries)
        
        print(f"Exported {len(summaries)} experiments to {csv_file}")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='View and compare experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View all experiments
  python experiment_manager.py
  
  # View only quantization experiments
  python experiment_manager.py --filter_quantization
  
  # View detailed information
  python experiment_manager.py --detailed
  
  # Sort by best validation accuracy
  python experiment_manager.py --sort_by best_val_accuracy
  
  # Sort by average validation std (lower = more stable)
  python experiment_manager.py --sort_by avg_val_std
  
  # Filter by model
  python experiment_manager.py --filter_model ResNet
  
  # Export to CSV
  python experiment_manager.py --export experiments.csv
        """
    )
    
    parser.add_argument(
        '--experiments_dir',
        type=str,
        default='src2/experiments',
        help='Path to experiments directory (default: src2/experiments)'
    )
    
    parser.add_argument(
        '--sort_by',
        type=str,
        default='experiment_id',
        help='Sort by field (default: experiment_id). Options: experiment_id, best_val_accuracy, avg_val_std, etc.'
    )
    
    parser.add_argument(
        '--filter_model',
        type=str,
        help='Filter by model name'
    )
    
    parser.add_argument(
        '--filter_quantization',
        action='store_true',
        help='Show only quantization experiments'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to CSV file'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed information for each experiment'
    )
    
    args = parser.parse_args()
    
    # Load experiment summaries
    experiments_dir = Path(args.experiments_dir)
    print(f"Loading experiments from: {experiments_dir.absolute()}")
    
    summaries = load_experiment_summaries(experiments_dir)
    
    if not summaries:
        print("No experiments found.")
        return
    
    # Filter summaries
    summaries = filter_summaries(
        summaries,
        filter_model=args.filter_model,
        filter_quantization=args.filter_quantization
    )
    
    # Sort summaries
    summaries = sort_summaries(summaries, args.sort_by)
    
    # Print summary table
    print_summary_table(summaries, detailed=args.detailed)
    
    # Export to CSV if requested
    if args.export:
        export_to_csv(summaries, args.export)


if __name__ == "__main__":
    main()

