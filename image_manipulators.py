import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def normalize(matrix):
    """Map the elements of `matrix` to the interval [-1, 1)."""
    a = np.amin(matrix)
    b = np.amax(matrix)
    width = b - a
    return (matrix - a) / width * 2 - 1

# Originally from https://gist.github.com/fmder/e28813c1e8721830ff9c which references Simard.
def elasticDeformation(image, prefilter=False):

    alpha = np.random.uniform(1.1, 1.8)
    sigma = np.random.uniform(8.0, 11.0)
    assert image.shape == (576,)
    shape = (24,24)
    image = image.reshape(shape)

    # Construct random vector field with components taken from the interval [0, 1)
    dx = np.random.rand(*shape)
    dy = np.random.rand(*shape)

    # Smooth the vector field by applying a Gaussian filter, then scale by `alpha`.
    dx = normalize(gaussian_filter(dx, sigma, mode='constant')) * alpha
    dy = normalize(gaussian_filter(dy, sigma, mode='constant')) * alpha

    # Construct the set of coordinates at which `image` will be sampled when constructing the output.
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    coordinates = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    return map_coordinates(image, coordinates, order=1, prefilter=prefilter).reshape((576,))

def simardElasticDeformation(image, sigma=4, alpha=24, \
    gf_args={'order': 0, 'mode': 'reflect', 'cval': 0.0, 'truncate': 4.0}, \
    mc_args={'order': 1, 'mode': 'constant', 'cval': 0.0, 'prefilter': False, }):
    """
    Author: Ben Price

    Description:
    Applies a random deformation to `image`. These deformations are meant to
    simulate the slightly different ways a human hand might draw an
    alphanumeric character. See [Simard, 2003] for more information.

    Arguments:
        `image`: 2D array of floats representing an image of a handwritten character
        `sigma`: radius of Gaussian filter used to smooth the random vector field
        `alpha`: scaling factor to apply to the normalized vector field
        `gf_args`: dictionary of keyword arguments to be passed to `gaussian_filter`
        'mc_args`: dictionary of keyword arguments to be passed to `map_coordinates`

    Returns: Deformed `image`.

    ==========================================================================
    References:
    P.Y. Simard, D. Steinkraus, and J.C. Platt. Best practices for
        convolutional neural networks applied to visual document analysis.
        In Seventh International Conference on Document Analysis and
        Recognition, 2003.
    """
    original_shape = (576,)
    shape = (24,24)
    assert image.shape == original_shape
    image = image.reshape(shape)

    randomField = lambda: np.random.rand(*shape) * 2 - 1
    smooth = lambda x: gaussian_filter(x, sigma, **gf_args)
    scale = lambda x: x / np.linalg.norm(x) * alpha

    dx, dy = scale(smooth(randomField())), scale(smooth(randomField()))

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    coordinates = np.stack((y + dy, x + dx))

    return map_coordinates(image, coordinates, **mc_args).reshape(original_shape)

def composeImages(image1, image2): #, replacement_threshold=0.3, opacity=0.4):
    """Replaces pixels with values less than `replacement_threshold`
    from `image1` with pixels from `image2`."""

    # The following hard-coded values have been eye-balled by me;
    # they appear to produce compositions which look like those given in the test set.
    replacement_threshold = np.random.uniform(0.3, 0.45)
    opacity = np.random.uniform(0.2, 0.6)

    # If a[i,j] <= replacement_threshold, then a[i,j] = b[i,j] * opacity
    replacement_indices = (image1 <= replacement_threshold)
    image1[replacement_indices] = image2[replacement_indices] * opacity
    return image1
