"""
Cross-validation strategies.
"""
import numpy as np

class UniformSplitter:
    """
    Systematic cross-validation splitter that maintains uniform spacing.
    
    This splitter generates train/test splits by selecting every k-th point 
    (with different offsets), preserving time step regularity required by FFT-based 
    methods. Ensures uniform sampling across folds.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits to generate. Each split uses 1/n_splits of the points 
        for testing and the rest for training.
    random_state : int or None, default=None
        Not used, included for compatibility with sklearn API.
    """
    
    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state
    
    def split(self, X, y=None, groups=None):
        """
        Generate systematic train/test splits.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Input data to split (used for indexing only).
        y : Ignored
            Included for API compatibility.
        groups : Ignored
            Included for API compatibility.
        
        Yields
        ------
        train_indices : ndarray of shape (n_train_samples,)
            Indices for training data.
        test_indices : ndarray of shape (n_test_samples,)
            Indices for testing data.
        """
        n_samples = len(X)
        
        for i in range(self.n_splits):
            # Take every n_splits-th point, starting from offset i
            train_indices = np.arange(i, n_samples, self.n_splits)
            test_indices = np.setdiff1d(np.arange(n_samples), train_indices)
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Return number of splits.
        
        Returns
        -------
        n_splits : int
            Number of CV splits.
        """
        return self.n_splits