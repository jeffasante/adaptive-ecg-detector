# Download real, messy, clinically validated data
!wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
!unzip -q 1.0.0.zip

if [ -d "data" ]; then 
    echo "Database directory 'data' already exists."
else
    echo "Directory 'data' not found. Creating and moving database..."
    mkdir -p data
    mv physionet.org data
fi
