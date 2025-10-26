'''
This is a rate-model of a patch of the mouse olfactory bulb with G glomeruli.
Each glomerulus has: OSN input O (sensory drive), an external tufted cell E, a mitral cell M, and a granule-like unit Gv.
Two pooled axonic interneuron populations — Se (external short-axon pools)
and Sd (deep short-axon pools) — provide interglomerular inhibition (Se) and
disinhibition via inhibition of granule cells (Sd). Dynamics are first-order linear
low-pass with a rectified nonlinearity (ReLU).
'''

import numpy as np
