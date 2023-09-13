class FIL1:
    """
    Contains information about a signal processing filter.
    
    Attributes
    ----------
        noOfCoefficients : int
            Number of filter coefficients [1]
        decimationFactor : int
            Filter decimation factor [1]
        coefficients : np.array
            Filter coefficients [1]
                    
    """
    def __init__(self, noOfCoefficients, decimationFactor, coefficients):
        """
        Initialise the FIL1 class.
        
        """

        self.NoOfCoefficients = noOfCoefficients
        self.DecimationFactor = decimationFactor
        self.Coefficients = coefficients
