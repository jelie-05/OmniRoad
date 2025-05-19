# tools/list_runs.py
import argparse
import yaml
import json
from pathlib import Path
import datetime
import sys
import pandas as pd

def list_runs(experiment_name, show_test=False):
    """List all runs for a given experiment."""
    experiment_dir = Path(f"outputs/{experiment_name}")
    
    if not experiment_dir.exists():
        print(f"No experiment directory found for '{experiment_name}'")
        return
    
    print(f"Runs for experiment '{experiment_name}':")
    print("-" * 120)
    
    if show_test:
        print(f"{'Run ID':<30} {'Date':<20} {'Val mIoU':<10} {'Test mIoU':<10} {'Test Acc':<10} {'Status':<10} {'Notes':<20}")
    else:
        print(f"{'Run ID':<30} {'Date':<20} {'Best mIoU':<10} {'Status':<10} {'Notes':<20}")
    print("-" * 120)
    
    for run_dir in sorted(experiment_dir.glob("run_*")):
        run_id = run_dir.name
        
        # Get creation date
        try:
            timestamp = datetime.datetime.fromtimestamp(run_dir.stat().st_ctime)
            date_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except:
            date_str = "Unknown"
        
        # Get best validation metrics
        best_metrics_path = run_dir / "best_metrics.txt"
        best_miou = "N/A"
        if best_metrics_path.exists():
            with open(best_metrics_path, 'r') as f:
                for line in f:
                    if "mean_iou" in line:
                        best_miou = line.split(":")[-1].strip()
                        best_miou = float(best_miou)
                        best_miou = f"{best_miou:.4f}"
                        break
        
        # Get test metrics if requested
        test_miou = "N/A"
        test_acc = "N/A"
        if show_test:
            test_metrics_path = run_dir / "test_results" / "test_metrics.json"
            if test_metrics_path.exists():
                try:
                    with open(test_metrics_path, 'r') as f:
                        test_data = json.load(f)
                        metrics = test_data.get('metrics', {})
                        test_miou = metrics.get('mean_iou', 'N/A')
                        test_acc = metrics.get('pixel_accuracy', 'N/A')
                        if isinstance(test_miou, (int, float)):
                            test_miou = f"{test_miou:.4f}"
                        if isinstance(test_acc, (int, float)):
                            test_acc = f"{test_acc:.4f}"
                except:
                    pass
        
        # Check if training is complete
        checkpoints_dir = run_dir / "checkpoints"
        test_dir = run_dir / "test_results"
        if checkpoints_dir.exists() and (checkpoints_dir / "best_model.pth").exists():
            if test_dir.exists() and (test_dir / "test_metrics.json").exists():
                status = "Complete+Test"
            else:
                status = "Complete"
        else:
            status = "Incomplete"
        
        # Check for custom notes
        notes_path = run_dir / "notes.txt"
        notes = ""
        if notes_path.exists():
            with open(notes_path, 'r') as f:
                notes = f.read().strip()
        
        # Print run info
        if show_test:
            print(f"{run_id:<30} {date_str:<20} {best_miou:<10} {test_miou:<10} {test_acc:<10} {status:<10} {notes[:20]:<20}")
        else:
            print(f"{run_id:<30} {date_str:<20} {best_miou:<10} {status:<10} {notes[:20]:<20}")
    
    print("-" * 120)

def show_run_config(experiment_name, run_id):
    """Display configuration for a specific run."""
    run_dir = Path(f"outputs/{experiment_name}/{run_id}")
    config_path = run_dir / "config.yaml"
    
    if not config_path.exists():
        print(f"No configuration file found for run '{run_id}'")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Configuration for run '{run_id}':")
    print("-" * 80)
    yaml.dump(config, sys.stdout, default_flow_style=False, sort_keys=False)
    print("-" * 80)

def show_test_results(experiment_name, run_id):
    """Show detailed test results for a specific run."""
    run_dir = Path(f"outputs/{experiment_name}/{run_id}")
    test_dir = run_dir / "test_results"
    
    if not test_dir.exists():
        print(f"No test results found for run '{run_id}'")
        return
    
    # Show test summary
    summary_path = test_dir / "test_summary.txt"
    if summary_path.exists():
        print(f"Test Summary for run '{run_id}':")
        print("-" * 80)
        with open(summary_path, 'r') as f:
            print(f.read())
        print("-" * 80)
    
    # Show detailed metrics
    metrics_path = test_dir / "test_metrics.json"
    if metrics_path.exists():
        print(f"\nDetailed Test Metrics:")
        print("-" * 80)
        with open(metrics_path, 'r') as f:
            test_data = json.load(f)
            
        print(f"Timestamp: {test_data.get('timestamp', 'N/A')}")
        print(f"Test time: {test_data.get('test_time_seconds', 'N/A'):.2f} seconds")
        
        metrics = test_data.get('metrics', {})
        print(f"\nOverall Metrics:")
        print(f"  Test Loss: {metrics.get('test_loss', 'N/A'):.4f}")
        print(f"  Pixel Accuracy: {metrics.get('pixel_accuracy', 'N/A'):.4f}")
        print(f"  Mean Accuracy: {metrics.get('mean_accuracy', 'N/A'):.4f}")
        print(f"  Mean IoU: {metrics.get('mean_iou', 'N/A'):.4f}")
        print(f"  Freq Weighted IoU: {metrics.get('fw_iou', 'N/A'):.4f}")
        
        # Show per-class IoU if available
        if 'iou' in metrics and isinstance(metrics['iou'], list):
            print(f"\nPer-Class IoU:")
            for i, iou in enumerate(metrics['iou']):
                print(f"  Class {i}: {iou:.4f}")
        
        print("-" * 80)

def add_notes(experiment_name, run_id, notes):
    """Add or update notes for a specific run."""
    run_dir = Path(f"outputs/{experiment_name}/{run_id}")
    
    if not run_dir.exists():
        print(f"Run '{run_id}' not found for experiment '{experiment_name}'")
        return False
    
    notes_path = run_dir / "notes.txt"
    
    # Check if notes file already exists
    if notes_path.exists():
        print(f"Existing notes found. Choose an option:")
        print("1. Overwrite existing notes")
        print("2. Append to existing notes")
        print("3. View existing notes and cancel")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "3":
            # Show existing notes and exit
            with open(notes_path, 'r') as f:
                existing_notes = f.read()
            print("\nExisting notes:")
            print("-" * 40)
            print(existing_notes)
            print("-" * 40)
            return False
        
        elif choice == "2":
            # Append to existing notes
            with open(notes_path, 'r') as f:
                existing_notes = f.read()
            
            # Add timestamp for the new addition
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            updated_notes = f"{existing_notes}\n\n[{timestamp}]\n{notes}"
            
            with open(notes_path, 'w') as f:
                f.write(updated_notes)
            
            print(f"Notes appended to run '{run_id}'")
            return True
    
    # Either overwrite or create new notes file
    with open(notes_path, 'w') as f:
        # Add timestamp for new notes
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}]\n{notes}")
    
    print(f"Notes {'updated' if notes_path.exists() else 'added'} for run '{run_id}'")
    return True

def view_notes(experiment_name, run_id):
    """View notes for a specific run."""
    run_dir = Path(f"outputs/{experiment_name}/{run_id}")
    
    if not run_dir.exists():
        print(f"Run '{run_id}' not found for experiment '{experiment_name}'")
        return
    
    notes_path = run_dir / "notes.txt"
    if not notes_path.exists():
        print(f"No notes found for run '{run_id}'")
        return
    
    with open(notes_path, 'r') as f:
        notes = f.read()
    
    print(f"Notes for run '{run_id}':")
    print("-" * 60)
    print(notes)
    print("-" * 60)

def compare_runs(experiment_name, run_ids, include_test=False):
    """Compare metrics between multiple runs."""
    if len(run_ids) < 2:
        print("Please specify at least two run IDs to compare")
        return
    
    # Collect training metrics
    training_metrics_data = {}
    test_metrics_data = {}
    
    for run_id in run_ids:
        run_dir = Path(f"outputs/{experiment_name}/{run_id}")
        if not run_dir.exists():
            print(f"Run '{run_id}' not found for experiment '{experiment_name}'")
            continue
        
        # Training metrics
        best_metrics_path = run_dir / "best_metrics.txt"
        if best_metrics_path.exists():
            metrics = {}
            with open(best_metrics_path, 'r') as f:
                for line in f:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metrics[key.strip()] = value.strip()
            training_metrics_data[run_id] = metrics
        
        # Test metrics
        if include_test:
            test_metrics_path = run_dir / "test_results" / "test_metrics.json"
            if test_metrics_path.exists():
                try:
                    with open(test_metrics_path, 'r') as f:
                        test_data = json.load(f)
                        test_metrics_data[run_id] = test_data.get('metrics', {})
                except:
                    test_metrics_data[run_id] = {}
    
    if not training_metrics_data and not test_metrics_data:
        return
    
    # Print training metrics comparison
    if training_metrics_data:
        print(f"Training Metrics Comparison for experiment '{experiment_name}':")
        print("-" * 120)
        
        # Find all training metrics keys
        all_training_metrics = set()
        for run_metrics in training_metrics_data.values():
            all_training_metrics.update(run_metrics.keys())
        
        # Header
        print(f"{'Metric':<30}", end="")
        for run_id in training_metrics_data.keys():
            print(f"{run_id[-18:]:<20}", end="")
        print()
        print("-" * 120)
        
        # Metrics rows
        for metric in sorted(all_training_metrics):
            print(f"{metric:<30}", end="")
            for run_id, metrics in training_metrics_data.items():
                value = metrics.get(metric, "N/A")
                print(f"{value:<20}", end="")
            print()
        
        print("-" * 120)
    
    # Print test metrics comparison
    if include_test and test_metrics_data:
        print(f"\nTest Metrics Comparison for experiment '{experiment_name}':")
        print("-" * 120)
        
        # Find all test metrics keys (focus on main metrics)
        main_test_metrics = ['test_loss', 'pixel_accuracy', 'mean_accuracy', 'mean_iou', 'fw_iou']
        
        # Header
        print(f"{'Metric':<30}", end="")
        for run_id in test_metrics_data.keys():
            print(f"{run_id[-18:]:<20}", end="")
        print()
        print("-" * 120)
        
        # Metrics rows
        for metric in main_test_metrics:
            print(f"{metric:<30}", end="")
            for run_id, metrics in test_metrics_data.items():
                value = metrics.get(metric, "N/A")
                if isinstance(value, (int, float)):
                    print(f"{value:.4f}{'':15}", end="")
                else:
                    print(f"{value:<20}", end="")
            print()
        
        print("-" * 120)
        
        # Per-class IoU comparison (if available and requested)
        if any('iou' in metrics for metrics in test_metrics_data.values()):
            print(f"\nPer-Class IoU Comparison:")
            print("-" * 120)
            
            # Find maximum number of classes
            max_classes = max(len(metrics.get('iou', [])) for metrics in test_metrics_data.values())
            
            # Header
            print(f"{'Class':<10}", end="")
            for run_id in test_metrics_data.keys():
                print(f"{run_id[-18:]:<20}", end="")
            print()
            print("-" * 120)
            
            # Per-class rows
            for class_idx in range(max_classes):
                print(f"Class {class_idx:<4}", end="")
                for run_id, metrics in test_metrics_data.items():
                    iou_list = metrics.get('iou', [])
                    if class_idx < len(iou_list):
                        iou_value = iou_list[class_idx]
                        print(f"{iou_value:.4f}{'':15}", end="")
                    else:
                        print(f"{'N/A':<20}", end="")
                print()
            
            print("-" * 120)

def compare_test_results(experiment_name, run_ids):
    """Compare only test results between multiple runs."""
    compare_runs(experiment_name, run_ids, include_test=True)

def export_comparison_csv(experiment_name, run_ids, output_file, include_test=False):
    """Export comparison results to CSV file."""
    if len(run_ids) < 2:
        print("Please specify at least two run IDs to export")
        return
    
    all_data = {}
    
    # Collect training metrics
    for run_id in run_ids:
        run_dir = Path(f"outputs/{experiment_name}/{run_id}")
        if not run_dir.exists():
            continue
        
        run_data = {'run_id': run_id}
        
        # Training metrics
        best_metrics_path = run_dir / "best_metrics.txt"
        if best_metrics_path.exists():
            with open(best_metrics_path, 'r') as f:
                for line in f:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        try:
                            run_data[f"train_{key.strip()}"] = float(value.strip())
                        except:
                            run_data[f"train_{key.strip()}"] = value.strip()
        
        # Test metrics
        if include_test:
            test_metrics_path = run_dir / "test_results" / "test_metrics.json"
            if test_metrics_path.exists():
                try:
                    with open(test_metrics_path, 'r') as f:
                        test_data = json.load(f)
                        metrics = test_data.get('metrics', {})
                        for key, value in metrics.items():
                            if key != 'iou':  # Skip per-class IoU for CSV simplicity
                                run_data[f"test_{key}"] = value
                except:
                    pass
        
        all_data[run_id] = run_data
    
    # Create DataFrame and save to CSV
    if all_data:
        df = pd.DataFrame.from_dict(all_data, orient='index')
        df.to_csv(output_file, index=False)
        print(f"Comparison exported to {output_file}")
    else:
        print("No data found to export")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage experiment runs")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List runs command
    list_parser = subparsers.add_parser("list", help="List all runs for an experiment")
    list_parser.add_argument("experiment", type=str, help="Name of the experiment")
    list_parser.add_argument("--test", action="store_true", help="Include test results in listing")
    
    # Show config command
    config_parser = subparsers.add_parser("config", help="Show configuration for a specific run")
    config_parser.add_argument("experiment", type=str, help="Name of the experiment")
    config_parser.add_argument("run_id", type=str, help="Run identifier")
    
    # Show test results command
    test_parser = subparsers.add_parser("test", help="Show test results for a specific run")
    test_parser.add_argument("experiment", type=str, help="Name of the experiment")
    test_parser.add_argument("run_id", type=str, help="Run identifier")
    
    # Add notes command
    notes_parser = subparsers.add_parser("notes", help="Add or update notes for a specific run")
    notes_parser.add_argument("experiment", type=str, help="Name of the experiment")
    notes_parser.add_argument("run_id", type=str, help="Run identifier")
    notes_parser.add_argument("--text", type=str, help="Notes text (if not provided, will prompt for input)")
    
    # View notes command
    view_notes_parser = subparsers.add_parser("view-notes", help="View notes for a specific run")
    view_notes_parser.add_argument("experiment", type=str, help="Name of the experiment")
    view_notes_parser.add_argument("run_id", type=str, help="Run identifier")
    
    # Compare runs command
    compare_parser = subparsers.add_parser("compare", help="Compare metrics between runs")
    compare_parser.add_argument("experiment", type=str, help="Name of the experiment")
    compare_parser.add_argument("run_ids", type=str, nargs="+", help="Run identifiers to compare")
    compare_parser.add_argument("--test", action="store_true", help="Include test results in comparison")
    
    # Compare test results command
    compare_test_parser = subparsers.add_parser("compare-test", help="Compare test results between runs")
    compare_test_parser.add_argument("experiment", type=str, help="Name of the experiment")
    compare_test_parser.add_argument("run_ids", type=str, nargs="+", help="Run identifiers to compare")
    
    # Export comparison command
    export_parser = subparsers.add_parser("export", help="Export comparison to CSV")
    export_parser.add_argument("experiment", type=str, help="Name of the experiment")
    export_parser.add_argument("run_ids", type=str, nargs="+", help="Run identifiers to compare")
    export_parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    export_parser.add_argument("--test", action="store_true", help="Include test results in export")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_runs(args.experiment, show_test=args.test)
    elif args.command == "config":
        show_run_config(args.experiment, args.run_id)
    elif args.command == "test":
        show_test_results(args.experiment, args.run_id)
    elif args.command == "notes":
        if args.text:
            add_notes(args.experiment, args.run_id, args.text)
        else:
            # Interactive mode for adding notes
            print(f"Enter notes for run '{args.run_id}' (press Ctrl+D or Ctrl+Z on a new line to finish):")
            notes_lines = []
            try:
                while True:
                    line = input()
                    notes_lines.append(line)
            except EOFError:
                notes = "\n".join(notes_lines)
                add_notes(args.experiment, args.run_id, notes)
    elif args.command == "view-notes":
        view_notes(args.experiment, args.run_id)
    elif args.command == "compare":
        compare_runs(args.experiment, args.run_ids, include_test=args.test)
    elif args.command == "compare-test":
        compare_test_results(args.experiment, args.run_ids)
    elif args.command == "export":
        export_comparison_csv(args.experiment, args.run_ids, args.output, include_test=args.test)
    else:
        parser.print_help()