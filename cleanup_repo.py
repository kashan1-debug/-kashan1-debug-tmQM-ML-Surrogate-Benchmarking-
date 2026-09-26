import os

# 1. List of extra plot files to delete
files_to_delete = [
    "coulomb_parity_plot.png",
    "enhanced_parity_plot.png",
    "final_parity_plot.png",
    "nonlinear_parity_plot.png",
]

# Delete extra files if they exist
print("--- Deleting Extra Files ---")
for file_name in files_to_delete:
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Deleted: {file_name}")
    else:
        print(f"File not found (already deleted): {file_name}")

# 2. Rename files containing ' (1)' to clean standard names
print("\n--- Renaming Files with '(1)' ---")
for file_name in os.listdir("."):
    if " (1)" in file_name:
        new_name = file_name.replace(" (1)", "")
        os.rename(file_name, new_name)
        print(f"Renamed: '{file_name}' -> '{new_name}'")

print("\nCleanup Complete!")
