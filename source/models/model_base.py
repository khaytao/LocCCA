from abc import ABC, abstractmethod

class LocalizationModel(ABC):
    """
    Abstract base class for localization models.
    All localization models should inherit from this class and implement the abstract methods.
    """
    
    @abstractmethod
    def preprocess(self, data_dir: str):
        """
        Preprocess data from the given directory path.
        
        Args:
            data_dir (str): Path to directory containing input data files
            
        Returns:
            Preprocessed data ready for model processing
        """
        pass
        
    @abstractmethod 
    def process(self, data):
        """
        Process the preprocessed data through the model.
        
        Args:
            data: Preprocessed data to be processed by the model
            
        Returns:
            Model output/predictions
        """
        pass
