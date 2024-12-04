# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import os


def image_grid(
    images,
    folder: str,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")
    

    if rows is None:
        rows = len(images)
        cols = 1

    os.makedirs(folder, exist_ok=True)

    for idx, im in enumerate(images):
        fig_single, ax_single = plt.subplots()  # Create a new figure for each image
        
        if rgb:
            ax_single.imshow(im[..., :3])
        else:
            ax_single.imshow(im[..., 3])
        
        ax_single.axis('off')  # Turn off axis
        
        # Save each figure as a separate image
        filename = f"{folder}/view_{idx:02d}.png"
        fig_single.savefig(filename, bbox_inches='tight', pad_inches=0)
        
        plt.close(fig_single)  # Close each figure to prevent memory issues

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            ax.imshow(im[..., :3])
        else:
            ax.imshow(im[..., 3])
        ax.axis('off')

    plt.show()  # Display all images as a grid in a single figure
    plt.close(fig)  # Close the grid figure after displaying