export LAUNCHER="cd /Documents/sink;
apt-get update && apt-get install -y ca-certificates;
update-ca-certificates;

rm -rf /cache/venv;
python3 -m venv /cache/venv;
source /cache/venv/bin/activate;

pip install --no-cache-dir numpy==1.24.4 pandas==2.0.0;
pip install --no-cache-dir hf_transfer rdkit levenshtein wandb;
pip install --no-cache-dir -e .;
pip install --no-cache-dir trl==0.14.0 transformers==4.48.2 gdown==4.7.1;
pip list
"

srun apptainer exec --nv \
    --bind /scratch \
    --bind /work \
    --mount type=bind,src=${LLM_MODEL_DIR},dst=/LLM_models \
    --mount type=bind,src="$(dirname "$(pwd)")",dst=/Documents \
    --mount type=bind,src=${CACHE_DIR},dst=/cache \
    --mount type=bind,src=${PROXY_CA},dst=/certs/cacert.pem \
    ${CONTAINER_PATH} \
    bash -c "$LAUNCHER"
