# tools/list_runs.py
import argparse
import yaml
from pathlib import Path
import datetime
import sys

def list_runs(experiment_name):
    """List all runs for a given experiment."""
    experiment_dir = Path(f"outputs/{experiment_name}")
    
    if not experiment_dir.exists():
        print(f"No experiment directory found for '{experiment_name}'")
        return
    
    print(f"Runs for experiment '{experiment_name}':")
    print("-" * 80)
    print(f"{'Run ID':<30} {'Date':<20} {'Best mIoU':<10} {'Status':<10} {'Notes':<20}")
    print("-" * 80)
    
    for run_dir in sorted(experiment_dir.glob("run_*")):
        run_id = run_dir.name
        
        # Get creation date
        try:
            timestamp = datetime.datetime.fromtimestamp(run_dir.stat().st_ctime)
            date_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except:
            date_str = "Unknown"
        
        # Get best metrics
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
        
        # Get configuration
        config_path = run_dir / "config.yaml"
        config_exists = config_path.exists()
        
        # Check if training is complete
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.exists() and (checkpoints_dir / "best_model.pth").exists():
            status = "Complete"
        else:
            status = "Incomplete"
        
        # Check for custom notes (could be added manually to a notes.txt file)
        notes_path = run_dir / "notes.txt"
        notes = ""
        if notes_path.exists():
            with open(notes_path, 'r') as f:
                notes = f.read().strip()
        
        # Print run info
        print(f"{run_id:<30} {date_str:<20} {best_miou:<10} {status:<10} {notes[:20]:<20}")
    
    print("-" * 80)

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

def compare_runs(experiment_name, run_ids):
    """Compare metrics between multiple runs."""
    if len(run_ids) < 2:
        print("Please specify at least two run IDs to compare")
        return
    
    # Collect metrics
    metrics_data = {}
    for run_id in run_ids:
        run_dir = Path(f"outputs/{experiment_name}/{run_id}")
        if not run_dir.exists():
            print(f"Run '{run_id}' not found for experiment '{experiment_name}'")
            continue
        
        best_metrics_path = run_dir / "best_metrics.txt"
        if not best_metrics_path.exists():
            print(f"No metrics found for run '{run_id}'")
            continue
        
        metrics = {}
        with open(best_metrics_path, 'r') as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    metrics[key.strip()] = value.strip()
        
        metrics_data[run_id] = metrics
    
    if not metrics_data:
        return
    
    # Find all metrics keys
    all_metrics = set()
    for run_metrics in metrics_data.values():
        all_metrics.update(run_metrics.keys())
    
    # Print comparison table
    print(f"Metrics comparison for experiment '{experiment_name}':")
    print("-" * 100)
    
    # Header
    print(f"{'Metric':<30}", end="")
    for run_id in metrics_data.keys():
        print(f"{run_id[:18]:<30}", end="")
    print()
    print("-" * 100)
    
    # Metrics rows
    for metric in sorted(all_metrics):
        print(f"{metric:<30}", end="")
        for run_id, metrics in metrics_data.items():
            value = metrics.get(metric, "N/A")
            # Format numeric values with correct format specifier
            if metric == "mean_iou" or "iou" in metric.lower():
                try:
                    value_float = float(value)
                    print(f"{value_float:.4f:<30}", end="")
                except (ValueError, TypeError):
                    print(f"{value:<30}", end="")
            else:
                print(f"{value:<30}", end="")
        print()
    
    print("-" * 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage experiment runs")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List runs command
    list_parser = subparsers.add_parser("list", help="List all runs for an experiment")
    list_parser.add_argument("experiment", type=str, help="Name of the experiment")
    
    # Show config command
    config_parser = subparsers.add_parser("config", help="Show configuration for a specific run")
    config_parser.add_argument("experiment", type=str, help="Name of the experiment")
    config_parser.add_argument("run_id", type=str, help="Run identifier")
    
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
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_runs(args.experiment)
    elif args.command == "config":
        show_run_config(args.experiment, args.run_id)
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
        compare_runs(args.experiment, args.run_ids)
    else:
        parser.print_help()

