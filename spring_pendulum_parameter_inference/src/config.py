"""
Configuration file for the Tangled Fibers project.

Centralizes all file paths, parameters, and settings to ensure consistency
across notebooks and scripts. Includes utilities for directory management.
"""

from pathlib import Path
import h5py
import shutil

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directory
DATA_ROOT = PROJECT_ROOT / "data"

# Plot output directory
PLOTS_ROOT = PROJECT_ROOT / "plots"

# ============================================================================
# UTILITIES
# ============================================================================

def initialize_h5(path: Path, overwrite = False, verbose = True):
    """Create or clear an HDF5 file at the given path."""
    if path.exists():
        if overwrite:
            if verbose: print(f'Writing to file (overwrite): {path}')
            with h5py.File(path, 'w'): pass
        else:
            if verbose: print(f'Writing to file (append): {path}')
            with h5py.File(path, 'a'): pass
    else:
        if verbose: print(f'Writing to file: {path}')
        with h5py.File(path, 'a'): pass

def prepare_directory(path: Path, clear = False, verbose = True):
    """
    Ensure the directory at `path` exists. If `clear=True`, remove all contents.
    """
    if path.exists() and path.is_dir():
        if clear:
            if verbose: print(f'Clearing {len(list(path.iterdir()))} files in directory: {path}')
            for item in path.iterdir():
                if item.is_file():
                    if verbose: print(f' -> Removing file: {item.relative_to(path)}')
                    item.unlink()
                elif item.is_dir():
                    if verbose: print(f' -> Removing directory: {item.relative_to(path)}')
                    shutil.rmtree(item, ignore_errors=True)
    else:
        if verbose: print(f'Creating directory: {path}')
        path.mkdir(parents=True, exist_ok=True)

def print_directory_tree(root_path, include_files=True, ignore_dotfiles=True, 
                        ignore_patterns=None, ignore_exact=None, _prefix="", _is_last=True):
    """
    Print directory structure in tree format with ignore options.
    
    Args:
        root_path (str or Path): Root directory to start from
        include_files (bool): Whether to include files (True) or only directories (False)
        ignore_dotfiles (bool): Whether to ignore files/directories starting with '.'
        ignore_patterns (list): List of substrings - ignore items containing any of these
        ignore_exact (list): List of exact names to ignore
        _prefix (str): Internal parameter for formatting (don't use directly)
        _is_last (bool): Internal parameter for formatting (don't use directly)
    """
    root_path = Path(root_path)
    
    # Set defaults for ignore lists
    if ignore_patterns is None:
        ignore_patterns = []
    if ignore_exact is None:
        ignore_exact = []
    
    if not root_path.exists():
        print(f"Error: Path '{root_path}' does not exist")
        return
    
    if not root_path.is_dir():
        print(f"Error: Path '{root_path}' is not a directory")
        return
    
    def should_ignore(item_name):
        """Check if an item should be ignored based on ignore rules."""
        # Check dotfiles
        if ignore_dotfiles and item_name.startswith('.'):
            return True
        
        # Check exact matches
        if item_name in ignore_exact:
            return True
        
        # Check patterns (substring matching)
        for pattern in ignore_patterns:
            if pattern in item_name:
                return True
        
        return False
    
    # Print the root directory name
    if _prefix == "":
        print(f"{root_path.name}/")
    
    try:
        # Get all items in the directory
        items = list(root_path.iterdir())
        
        # Filter out ignored items
        items = [item for item in items if not should_ignore(item.name)]
        
        # Filter based on include_files parameter
        if include_files:
            # Sort: directories first, then files, both alphabetically
            directories = sorted([item for item in items if item.is_dir()])
            files = sorted([item for item in items if item.is_file()])
            all_items = directories + files
        else:
            # Only directories
            all_items = sorted([item for item in items if item.is_dir()])
        
        # Print each item
        for i, item in enumerate(all_items):
            is_last_item = (i == len(all_items) - 1)
            
            # Determine the connector character
            if is_last_item:
                connector = "└── "
                next_prefix = _prefix + "    "
            else:
                connector = "├── "
                next_prefix = _prefix + "│   "
            
            # Print the item
            if item.is_dir():
                print(f"{_prefix}{connector}{item.name}/")
                # Recursively print subdirectory with same ignore rules
                print_directory_tree(item, include_files, ignore_dotfiles, 
                                   ignore_patterns, ignore_exact, next_prefix, is_last_item)
            else:
                print(f"{_prefix}{connector}{item.name}")
                
    except PermissionError:
        print(f"{_prefix}[Permission Denied]")
    except Exception as e:
        print(f"{_prefix}[Error: {e}]")