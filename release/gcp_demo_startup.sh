#!/usr/bin/env bash
set -euo pipefail

LOG_FILE=/var/log/mist-demo-startup.log
exec > >(tee -a "$LOG_FILE") 2>&1

metadata_value() {
  local key="$1"
  local fallback="${2:-}"
  local value
  value="$(curl -fs -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/${key}" 2>/dev/null || true)"
  if [[ -n "$value" ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$fallback"
  fi
}

REPO_URL="$(metadata_value REPO_URL "https://github.com/schwallergroup/mist.git")"
REPO_DIR="/opt/mist"
BRANCH="$(metadata_value BRANCH "main")"
FIGSHARE_URL="$(metadata_value FIGSHARE_URL "")"
EXTRACT_DEMO_FIXTURES="$(metadata_value EXTRACT_DEMO_FIXTURES "0")"
DEMO_SAMPLE_ROWS="$(metadata_value DEMO_SAMPLE_ROWS "50")"

echo "[$(date -Iseconds)] Starting MiST demo VM bootstrap"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y curl file git python3 python3-venv unzip

if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" fetch origin "$BRANCH"
  git -C "$REPO_DIR" checkout "$BRANCH"
  git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

python3 <<'PY'
from pathlib import Path

repo_dir = Path("/opt/mist")
forward_path = repo_dir / "src/open_r1/tasks/reactions/forward.py"
forward_text = forward_path.read_text()
if "_has_local_reaction_files" not in forward_text:
    forward_text = forward_text.replace(
        "    custom_metrics: Dict[str, Any] = {}\n"
        "    random_log: Dict[str, Any] = {}\n"
        "    printed_sample_prompt: bool = False\n",
        "    custom_metrics: Dict[str, Any] = {}\n"
        "    random_log: Dict[str, Any] = {}\n"
        "    printed_sample_prompt: bool = False\n\n"
        "    @staticmethod\n"
        "    def _has_local_reaction_files(dataset_dir: str) -> bool:\n"
        "        required_files = [\n"
        "            \"src-train.txt\",\n"
        "            \"tgt-train.txt\",\n"
        "            \"src-test.txt\",\n"
        "            \"tgt-test.txt\",\n"
        "        ]\n"
        "        return all(\n"
        "            os.path.exists(os.path.join(dataset_dir, filename))\n"
        "            for filename in required_files\n"
        "        )\n",
    )
    forward_text = forward_text.replace(
        "        download_data(self.dataset_id_or_path)\n",
        "        if not self._has_local_reaction_files(self.dataset_id_or_path):\n"
        "            download_data(self.dataset_id_or_path)\n",
    )
    forward_path.write_text(forward_text)

demo_dir = repo_dir / "demo/rxnpred_tiny"
demo_dir.mkdir(parents=True, exist_ok=True)
train_pairs = [
    ("CCO.C", "CCOC"),
    ("CC.C", "CCC"),
    ("CCO.N", "CCON"),
    ("CCC.O", "CCCO"),
    ("CCN.C", "CCNC"),
    ("CO.C", "COC"),
    ("CCCl.O", "OCCCl"),
    ("CCBr.N", "NCCBr"),
    ("CCF.O", "OCCF"),
    ("CCS.C", "CCSC"),
    ("CCO.CC", "CCOCC"),
    ("CCN.CC", "CCNCC"),
    ("CCC.N", "CCCN"),
    ("CCCC.O", "CCCCO"),
    ("CO.N", "CON"),
    ("CCO.Cl", "CCOCl"),
    ("CCN.O", "CCNO"),
    ("CCCO.C", "CCCOC"),
    ("CC(C)O.C", "CC(C)OC"),
    ("CC(C)N.C", "CC(C)NC"),
    ("CCOC.C", "CCOCC"),
    ("CCCN.C", "CCCNC"),
    ("CCOCC.O", "CCOCCO"),
    ("CC(=O)O.C", "COC(C)=O"),
    ("CC(=O)N.C", "CNC(C)=O"),
    ("CCCO.N", "CCCON"),
    ("CCS.O", "OCCS"),
    ("COC.C", "COCC"),
    ("COC.N", "CONC"),
    ("CC(C)Cl.O", "OCC(C)Cl"),
    ("CC(C)Br.N", "NCC(C)Br"),
    ("CC(C)F.O", "OCC(C)F"),
    ("CCCOC.C", "CCCOCC"),
    ("CCNCC.O", "OCCNCC"),
    ("CCOCN.C", "CCOCNC"),
    ("CCSC.N", "CCSCN"),
    ("CCCOCC.O", "CCCOCCO"),
    ("CCNCO.C", "CCNCOC"),
    ("CCOCCl.N", "NCCOCCl"),
    ("CCBr.C", "CCCBr"),
]
test_pairs = [
    ("CO.C", "COC"),
    ("CC.N", "CCNC"),
    ("CCO.C", "CCOC"),
    ("CCC.N", "CCCN"),
    ("CCO.O", "CCOO"),
    ("CCCl.N", "NCCCl"),
    ("CCBr.O", "OCCBr"),
    ("CCS.N", "NCCS"),
    ("CCF.N", "NCCF"),
    ("CCOC.N", "CCOCN"),
]
(demo_dir / "src-train.txt").write_text(
    "\n".join(src for src, _ in train_pairs) + "\n"
)
(demo_dir / "tgt-train.txt").write_text(
    "\n".join(tgt for _, tgt in train_pairs) + "\n"
)
(demo_dir / "src-test.txt").write_text(
    "\n".join(src for src, _ in test_pairs) + "\n"
)
(demo_dir / "tgt-test.txt").write_text(
    "\n".join(tgt for _, tgt in test_pairs) + "\n"
)

run_demo = repo_dir / "demo/run_demo.py"
if not run_demo.exists():
    run_demo.write_text(
        "import json\n"
        "from pathlib import Path\n\n"
        "from open_r1.tasks.reactions.forward import ForwardReaction\n\n\n"
        "def main():\n"
        "    dataset_dir = Path(__file__).resolve().parent / 'rxnpred_tiny'\n"
        "    task = ForwardReaction(dataset_id_or_path=str(dataset_dir))\n"
        "    dataset = task.load()\n"
        "    prompts = [dataset['test'][0]['problem'], dataset['test'][1]['problem']]\n"
        "    solutions = [dataset['test'][0]['solution'], dataset['test'][1]['solution']]\n"
        "    completions = [\n"
        "        '<think>A plausible product is COC based on the reagents.</think><answer>COC</answer>',\n"
        "        '<think>A plausible product is CCNC based on the reagents.</think><answer>CCNC</answer>',\n"
        "    ]\n"
        "    rewards = task.accuracy_reward(completions, solutions, prompts)\n"
        "    summary = {\n"
        "        'train_examples': len(dataset['train']),\n"
        "        'test_examples': len(dataset['test']),\n"
        "        'solutions': solutions,\n"
        "        'rewards': rewards,\n"
        "    }\n"
        "    print(json.dumps(summary, indent=2))\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

make_kinetic_tiny = repo_dir / "demo/make_kinetic_tiny.py"
if not make_kinetic_tiny.exists():
    make_kinetic_tiny.write_text(
        "import pickle\n"
        "from pathlib import Path\n\n"
        "import numpy as np\n\n"
        "TRAIN_SIZE = 40\n"
        "VAL_SIZE = 10\n"
        "TIME_STEPS = 4\n"
        "CHANNELS = 12\n\n"
        "def build_x1(num_examples: int, offset: float) -> np.ndarray:\n"
        "    rows = []\n"
        "    for i in range(num_examples):\n"
        "        base = offset + i * 0.01\n"
        "        rows.append([base + 0.10, base + 0.20, base + 0.30, base + 0.40])\n"
        "    return np.array(rows, dtype=float)\n\n"
        "def build_x2(num_examples: int, offset: float) -> np.ndarray:\n"
        "    tensor = np.zeros((num_examples, TIME_STEPS, CHANNELS), dtype=float)\n"
        "    for i in range(num_examples):\n"
        "        scale = offset + i * 0.02\n"
        "        for t in range(TIME_STEPS):\n"
        "            time_value = round(t / (TIME_STEPS - 1), 3)\n"
        "            tensor[i, t, 0] = time_value\n"
        "            tensor[i, t, 3] = time_value\n"
        "            tensor[i, t, 6] = time_value\n"
        "            tensor[i, t, 9] = time_value\n"
        "            for run in range(4):\n"
        "                substrate_col = run * 3 + 1\n"
        "                product_col = run * 3 + 2\n"
        "                start = max(0.15, 1.0 - 0.05 * run - 0.02 * i)\n"
        "                decay = min(0.75, 0.16 * t + 0.015 * i + 0.02 * run + scale)\n"
        "                substrate = max(0.0, round(start - decay, 3))\n"
        "                product = min(1.0, round(1.0 - substrate, 3))\n"
        "                tensor[i, t, substrate_col] = substrate\n"
        "                tensor[i, t, product_col] = product\n"
        "    return tensor\n\n"
        "def build_y(num_examples: int, start_class: int) -> np.ndarray:\n"
        "    return np.array([[(start_class + i) % 20] for i in range(num_examples)], dtype=int)\n\n"
        "def build_kinetic_files():\n"
        "    return {\n"
        "        'x1_train_M1_M20_train_val_test_set_part_0.pkl': build_x1(TRAIN_SIZE, 0.00),\n"
        "        'x2_train_M1_M20_train_val_test_set_part_0.pkl': build_x2(TRAIN_SIZE, 0.00),\n"
        "        'y_train_M1_M20_train_val_test_set_part_0.pkl': build_y(TRAIN_SIZE, 0),\n"
        "        'x1_val_M1_M20_train_val_test_set_part_0.pkl': build_x1(VAL_SIZE, 0.40),\n"
        "        'x2_val_M1_M20_train_val_test_set_part_0.pkl': build_x2(VAL_SIZE, 0.10),\n"
        "        'y_val_M1_M20_train_val_test_set_part_0.pkl': build_y(VAL_SIZE, 4),\n"
        "    }\n\n"
        "def ensure_kinetic_tiny(output_dir: Path) -> Path:\n"
        "    output_dir.mkdir(parents=True, exist_ok=True)\n"
        "    for filename, array in build_kinetic_files().items():\n"
        "        path = output_dir / filename\n"
        "        with path.open('wb') as handle:\n"
        "            pickle.dump(array, handle)\n"
        "    return output_dir\n\n"
        "if __name__ == '__main__':\n"
        "    ensure_kinetic_tiny(Path(__file__).resolve().parent / 'kinetic_tiny')\n"
    )

run_fixture_smoke = repo_dir / "demo/run_fixture_smoke.py"
if not run_fixture_smoke.exists():
    run_fixture_smoke.write_text(
        "import json\n"
        "from pathlib import Path\n\n"
        "from open_r1.tasks import CHEMTASKS\n"
        "from make_kinetic_tiny import ensure_kinetic_tiny\n\n"
        "def summarize_dataset(task):\n"
        "    dataset = task.load()\n"
        "    summary = {\n"
        "        'train_examples': len(dataset['train']),\n"
        "        'test_examples': len(dataset['test']),\n"
        "    }\n"
        "    if len(dataset['train']):\n"
        "        summary['train_columns'] = sorted(dataset['train'].column_names)\n"
        "        summary['train_solution_example'] = dataset['train'][0]['solution']\n"
        "    if len(dataset['test']):\n"
        "        summary['test_solution_example'] = dataset['test'][0]['solution']\n"
        "    return summary\n\n"
        "def main():\n"
        "    demo_dir = Path(__file__).resolve().parent\n"
        "    datasets_dir = demo_dir / 'datasets'\n"
        "    kinetic_dir = ensure_kinetic_tiny(demo_dir / 'kinetic_tiny')\n"
        "    task_configs = {\n"
        "        'rxnpred': demo_dir / 'rxnpred_tiny',\n"
        "        'iupacsm': datasets_dir / 'CRLLM-PubChem-compounds1M.sample.csv',\n"
        "        'iupacsm_with_tags': datasets_dir / 'CRLLM-PubChem-compounds1M-simple.sample.csv',\n"
        "        'canonic': datasets_dir / 'CRLLM-PubChem-compounds1M.sample.csv',\n"
        "        'canonmc': datasets_dir / 'CRLLM-PubChem-compounds1M.sample.csv',\n"
        "        'smi_permute': datasets_dir / 'CRLLM-PubChem-compounds1M-very_very_simple.sample.csv',\n"
        "        'smhydrogen': datasets_dir / 'CRLLM-PubChem-compounds1M_hydrogen.sample.csv',\n"
        "        'kinetic': kinetic_dir,\n"
        "    }\n"
        "    summary = {}\n"
        "    for task_name, dataset_path in task_configs.items():\n"
        "        task = CHEMTASKS[task_name](dataset_id_or_path=str(dataset_path))\n"
        "        summary[task_name] = {'dataset_path': str(dataset_path), **summarize_dataset(task)}\n"
        "    print(json.dumps(summary, indent=2))\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
PY

python3 -m venv /opt/mist-demo-venv
source /opt/mist-demo-venv/bin/activate
pip install --upgrade pip
pip install -r "$REPO_DIR/dev-requirements.txt"

cd "$REPO_DIR"
PYTHONPATH=src python demo/run_demo.py | tee /var/log/mist-demo-output.json

if [[ -n "$FIGSHARE_URL" ]]; then
  echo "[$(date -Iseconds)] Fetching Figshare archive metadata from $FIGSHARE_URL"
  mkdir -p /var/log/mist-figshare
  python3 <<'PY'
import json
import re
import urllib.request


def fetch(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/FIGSHARE_URL"
req = urllib.request.Request(metadata_url, headers={"Metadata-Flavor": "Google"})
with urllib.request.urlopen(req, timeout=10) as response:
    figshare_url = response.read().decode("utf-8", errors="replace")

match = re.search(r"/(\d+)(?:$|[/?#])", figshare_url)
article_id = match.group(1) if match else ""

if article_id:
    api_urls = [
        f"https://api.figshare.com/v2/articles/{article_id}",
        f"https://api.figshare.com/v2/articles/{article_id}/files",
    ]
    for api_url in api_urls:
        print(f"FIGSHARE_API_URL {api_url}")
        try:
            status, body = fetch(api_url)
            print(f"FIGSHARE_API_STATUS {status}")
            print(body[:4000])
        except Exception as exc:
            print(f"FIGSHARE_API_ERROR {api_url} {exc!r}")
        print("FIGSHARE_API_END")
PY
  curl -L -D /var/log/mist-figshare/headers.txt -o /var/log/mist-figshare/article.bundle "$FIGSHARE_URL" || true
  file /var/log/mist-figshare/article.bundle | tee /var/log/mist-figshare/filetype.txt || true
  unzip -l /var/log/mist-figshare/article.bundle > /var/log/mist-figshare/unzip-list.txt 2>&1 || true
  tar -tf /var/log/mist-figshare/article.bundle > /var/log/mist-figshare/tar-list.txt 2>&1 || true
  python3 <<'PY'
import hashlib
import json
import io
import os
import subprocess
import tarfile
import urllib.request
import zipfile
import pickle
import csv
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - best effort on the VM
    np = None

files = [
    (
        "code.zip",
        "https://ndownloader.figshare.com/files/54861755",
        20229065,
        "2a93762a87db6a3e9065cbed16c87abf",
    ),
    (
        "datasets.zip",
        "https://ndownloader.figshare.com/files/55028555",
        2359373145,
        "774bae0908abb81a419adaa0cce83b2d",
    ),
    (
        "models.zip",
        "https://ndownloader.figshare.com/files/55032923",
        5419112931,
        "6b8702e8a52a3b30463a4101dfff28b5",
    ),
]
workdir = Path("/var/log/mist-figshare/files")
workdir.mkdir(parents=True, exist_ok=True)
summary = []
for name, url, expected_size, expected_md5 in files:
    path = workdir / name
    print(f"FIGSHARE_FILE_DOWNLOAD {name} {url}")
    download = subprocess.run(
        [
            "curl",
            "-L",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "-A",
            "Mozilla/5.0",
            "-o",
            str(path),
            url,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if download.returncode != 0:
        raise RuntimeError(
            f"Failed to download {name}: {download.stderr[:2000]}"
        )
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    listing = subprocess.run(
        ["unzip", "-l", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    summary.append(
        {
            "name": name,
            "size": path.stat().st_size,
            "expected_size": expected_size,
            "md5": digest.hexdigest(),
            "expected_md5": expected_md5,
            "zip_ok": listing.returncode == 0,
            "zip_preview": listing.stdout.splitlines()[:60],
            "zip_error": listing.stderr[:2000],
        }
    )
print("FIGSHARE_FILE_SUMMARY_START")
print(json.dumps(summary, indent=2))
print("FIGSHARE_FILE_SUMMARY_END")

metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/EXTRACT_DEMO_FIXTURES"
req = urllib.request.Request(metadata_url, headers={"Metadata-Flavor": "Google"})
try:
    with urllib.request.urlopen(req, timeout=10) as response:
        extract_demo_fixtures = response.read().decode("utf-8", errors="replace").strip()
except Exception:
    extract_demo_fixtures = "0"

sample_rows_url = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/DEMO_SAMPLE_ROWS"
sample_req = urllib.request.Request(sample_rows_url, headers={"Metadata-Flavor": "Google"})
try:
    with urllib.request.urlopen(sample_req, timeout=10) as response:
        demo_sample_rows = int(response.read().decode("utf-8", errors="replace").strip())
except Exception:
    demo_sample_rows = 50

if extract_demo_fixtures == "1":
    datasets_zip = workdir / "datasets.zip"
    output_dir = Path("/var/log/mist-demo-fixtures")
    if output_dir.exists():
        subprocess.run(["rm", "-rf", str(output_dir)], check=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"created": [], "missing": []}

    def first_match(names, predicate):
        for name in names:
            if predicate(name):
                return name
        return None

    with zipfile.ZipFile(datasets_zip) as zf:
        names = zf.namelist()

        # Real rxnpred slice
        rxnpred_dir = output_dir / "rxnpred_real_tiny"
        rxnpred_dir.mkdir(parents=True, exist_ok=True)
        for filename in ["src-train.txt", "tgt-train.txt", "src-test.txt", "tgt-test.txt"]:
            member = first_match(
                names,
                lambda n, filename=filename: n.endswith(filename) and "USPTO_480k_clean" in n,
            )
            if member is None:
                manifest["missing"].append({"asset": filename, "match": "USPTO_480k_clean"})
                continue
            lines = zf.read(member).decode("utf-8", errors="replace").splitlines()
            sample_lines = [line for line in lines if line.strip()][:demo_sample_rows]
            target = rxnpred_dir / filename
            target.write_text("\n".join(sample_lines) + "\n", encoding="utf-8")
            manifest["created"].append(
                {
                    "asset": f"rxnpred_real_tiny/{filename}",
                    "source_zip_member": member,
                    "n_rows": len(sample_lines),
                }
            )

        # CSV-based tasks
        csv_targets = [
            "CRLLM-PubChem-compounds1M.csv",
            "CRLLM-PubChem-compounds1M_hydrogen.csv",
            "CRLLM-PubChem-compounds1M-simple.csv",
            "CRLLM-PubChem-compounds1M-very_very_simple.csv",
        ]
        for basename in csv_targets:
            member = first_match(names, lambda n, basename=basename: n.endswith("/" + basename) or n.endswith(basename))
            if member is None:
                manifest["missing"].append({"asset": basename, "match": basename})
                continue
            data = zf.read(member).decode("utf-8", errors="replace").splitlines()
            reader = csv.reader(data)
            rows = list(reader)
            sample_rows = rows[: demo_sample_rows + 1]
            target = output_dir / basename.replace(".csv", ".sample.csv")
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows(sample_rows)
            manifest["created"].append(
                {
                    "asset": target.name,
                    "source_zip_member": member,
                    "n_rows": max(len(sample_rows) - 1, 0),
                    "header": sample_rows[0] if sample_rows else [],
                }
            )

        # Kinetic slice
        kinetic_dir = output_dir / "kinetic_tiny"
        kinetic_dir.mkdir(parents=True, exist_ok=True)
        kinetic_files = [
            "x1_train_M1_M20_train_val_test_set_part_0.pkl",
            "x2_train_M1_M20_train_val_test_set_part_0.pkl",
            "y_train_M1_M20_train_val_test_set_part_0.pkl",
            "x1_val_M1_M20_train_val_test_set_part_0.pkl",
            "x2_val_M1_M20_train_val_test_set_part_0.pkl",
            "y_val_M1_M20_train_val_test_set_part_0.pkl",
        ]
        for basename in kinetic_files:
            member = first_match(names, lambda n, basename=basename: n.endswith("/" + basename) or n.endswith(basename))
            if member is None:
                manifest["missing"].append({"asset": basename, "match": basename})
                continue
            raw = io.BytesIO(zf.read(member))
            obj = pickle.load(raw)
            if hasattr(obj, "__getitem__"):
                try:
                    sample_obj = obj[:8]
                except Exception:
                    sample_obj = obj
            else:
                sample_obj = obj
            target = kinetic_dir / basename
            with target.open("wb") as handle:
                pickle.dump(sample_obj, handle)
            entry = {
                "asset": f"kinetic_tiny/{basename}",
                "source_zip_member": member,
            }
            if hasattr(sample_obj, "shape"):
                entry["shape"] = list(sample_obj.shape)
            elif isinstance(sample_obj, list):
                entry["length"] = len(sample_obj)
            manifest["created"].append(entry)

    tar_path = Path("/var/log/mist-demo-fixtures.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(output_dir, arcname="demo-fixtures")

    encoded = tar_path.read_bytes().hex()
    print("DEMO_FIXTURES_MANIFEST_START")
    print(json.dumps(manifest, indent=2))
    print("DEMO_FIXTURES_MANIFEST_END")
    print("DEMO_FIXTURES_HEX_START")
    for start in range(0, len(encoded), 512):
        print(encoded[start : start + 512])
    print("DEMO_FIXTURES_HEX_END")

    repo_demo_datasets = Path("/opt/mist") / "demo" / "datasets"
    repo_demo_datasets.mkdir(parents=True, exist_ok=True)
    for source_name in [
        "CRLLM-PubChem-compounds1M.sample.csv",
        "CRLLM-PubChem-compounds1M_hydrogen.sample.csv",
        "CRLLM-PubChem-compounds1M-very_very_simple.sample.csv",
    ]:
        source = output_dir / source_name
        if source.exists():
            (repo_demo_datasets / source_name).write_bytes(source.read_bytes())
    source = output_dir / "CRLLM-PubChem-compounds1M.sample.csv"
    if source.exists():
        (repo_demo_datasets / "CRLLM-PubChem-compounds1M-simple.sample.csv").write_bytes(source.read_bytes())
PY
fi

python3 "$REPO_DIR/demo/make_kinetic_tiny.py"
cd "$REPO_DIR"
PYTHONPATH=src python demo/run_fixture_smoke.py | tee /var/log/mist-fixture-smoke.json

echo "[$(date -Iseconds)] MiST demo completed"
