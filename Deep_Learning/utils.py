import string
from PIL import Image

IMG_SIZE = 150


def scatter_pixels(img_file):
    """Return the URL of a scatter plot of the supplied image

    The image will be rendered square and black on white. Adapt the
    code if you want something else.
    """
    # Use simple chart encoding. To make things really simple
    # use a square image where each X or Y position corresponds
    # to a single encode value.
    w = IMG_SIZE
    img = Image.open(img_file).resize((w, w)).convert("1")
    pels = img.load()
    black_pels = [(x, y) for x in range(w) for y in range(w)
                  if pels[x, y] == 0]
    sqside = 3.0
    
    # invert Y coordinate with w-y
    return [t[0] for t in black_pels],[w - t[1] for t in black_pels]


def pack_data(x,y):
    """
    pack 2d data to 1d vector
    """
    one_d_data = []
    for i in range(len(x)):
        one_d_data.append(x[i])
        one_d_data.append(y[i])
        
    return one_d_data
        
def unpack_1d_data(one_d_data):
    """
    unpack 1d data to 2d vector
    """
    x = []
    y = []
    for i in range(len(one_d_data)):
        if i%2==0:
            x.append(one_d_data[i])
        else:
            y.append(one_d_data[i])
    return x,y


