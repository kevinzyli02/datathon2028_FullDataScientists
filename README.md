# datathon2028_FullDataScientists

To Use: 

Download virtual environment using command prompt:

cd /__insert file location__

then use the folloiwng prompt to install the necessary packages

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

# create requirements
cat > requirements.txt << 'EOF'
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
pyyaml>=6.0
EOF

pip install -r requirements.txt
