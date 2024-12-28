import magic  # Install with `pip install python-magic`

file_path = "4.875N52.375E-200001-sfc.nc"
file_type = magic.Magic(mime=True).from_file(file_path)
print(f"File type: {file_type}")