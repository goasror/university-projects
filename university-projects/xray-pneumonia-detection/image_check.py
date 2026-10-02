from PIL import Image, ImageStat


def is_xray_image(image: Image.Image) -> bool:
    """
    Detector — works with dark, bright, and high-contrast X-rays.
    """
    width, height = image.size
    aspect_ratio = height / width
    if not (0.6 <= aspect_ratio <= 2.2):  # allow more variation
        return False

    # Convert to grayscale
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    mean_intensity = stat.mean[0]
    std_intensity = stat.stddev[0]

    # RGB uniformity check
    if image.mode == "RGB":
        r, g, b = image.split()
        diff_rg = abs(ImageStat.Stat(r).mean[0] - ImageStat.Stat(g).mean[0])
        diff_rb = abs(ImageStat.Stat(r).mean[0] - ImageStat.Stat(b).mean[0])

        if diff_rg > 35 or diff_rb > 35:
            return False

    # Contrast must exist (flat images are not X-rays)
    if std_intensity < 2:  # almost pure white/gray → reject
        return False

    # Brightness lower bound (cannot be too dark)
    if mean_intensity < 5:  # very dark → reject
        return False

    # Remove upper brightness limit → bright X-rays pass
    return True
