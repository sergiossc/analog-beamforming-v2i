import numpy as np

class PartitionedArray:
    def __init__(self, size, element_spacing, wavelength, formfactory):
        self.formfactory = formfactory
        self.size = size
        if self.formfactory == 'UPA':
            self.ura = np.ones((int(np.sqrt(size)), int(np.sqrt(size))))
        else:
            if self.formfactory == 'ULA':
                self.ula = np.ones((size, 1))
            else:
                raise Exception('"formfactor" parameter should be "UPA" or "ULA".')
        self.element_spacing = element_spacing
        self.wave_length = wavelength
