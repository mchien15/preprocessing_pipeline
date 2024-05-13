if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <image_folder> <output_folder>"
    exit 1
fi

image_folder="$1"
output_folder="$2"

for image_path in "$image_folder"/*; do
    if [ -f "$image_path" ]; then
        ./test -idir="$image_path" -odir="$output_folder"
    fi
done