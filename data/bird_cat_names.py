import os

# Path to your classes directory
classes_dir = "data/v5/dataset/train"
output_header = "bird_category_name.hpp"

# Read and sort class folder names
class_names = sorted([
    name for name in os.listdir(classes_dir)
    if os.path.isdir(os.path.join(classes_dir, name))
])

# Generate C++ array string
header_guard = "#pragma once\n"
array_name = "static const char *classification_cat_names[] = {\n"
entries = [f'    "{name}",' for name in class_names]
array_end = "\n};\n"

# Combine and write to file
with open(output_header, "w") as f:
    f.write(header_guard)
    f.write(array_name)
    f.write("\n".join(entries))
    f.write(array_end)

print(f"Generated {output_header} with {len(class_names)} class entries.")
