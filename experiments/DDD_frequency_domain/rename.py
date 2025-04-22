import os


def rename_files_in_directory(directory):
  # List all files in the directory
  files = os.listdir(directory)

  # Sort the files to ensure consistency in naming
  files.sort()

  # Separate files into categories
  captions = []
  images = []
  masks = []
  for f in files:
    if f.endswith("_captions.txt"):
      captions.append(f)
    elif f.endswith("_mask.png"):
      masks.append(f)
    elif f.endswith(".png"):
      images.append(f)

  # Ensure the lists are of the same length and correspond correctly
  if len(captions) != len(images) or len(images) != len(masks):
    print("Mismatch in the number of captions, images, or masks.")
    return

  # Rename files sequentially
  for i, (caption, image, mask) in enumerate(
      zip(captions, images, masks), start=1
  ):
    # Determine new names
    new_caption_name = f"{i}_captions.txt"
    new_image_name = f"{i}.png"
    new_mask_name = f"{i}_masked.png"

    # Rename the files
    os.rename(
        os.path.join(directory, caption),
        os.path.join(directory, new_caption_name)
    )
    os.rename(
        os.path.join(directory, image), os.path.join(directory, new_image_name)
    )
    os.rename(
        os.path.join(directory, mask), os.path.join(directory, new_mask_name)
    )

    print(
        f"Renamed: {caption} -> {new_caption_name}, {image} -> {new_image_name}, {mask} -> {new_mask_name}"
    )


# Example usage
directory_path = "./people/"
rename_files_in_directory(directory_path)
