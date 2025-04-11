

# Building an LLMOPS Pipeline with Airflow on Google Kubernetes Engine: A Comprehensive Guide

Machine Learning Operations (MLOps) is revolutionizing how we develop, deploy, and maintain machine learning models in production. By blending DevOps practices with machine learning workflows, MLOps ensures scalability, reproducibility, and efficiency. In this article, we’ll walk through a hands-on guide to building an end-to-end MLOps pipeline using Apache Airflow on Google Kubernetes Engine (GKE). You’ll learn how to automate data downloading, data preparation, model fine-tuning, and deployment of a large language model (LLM) using a Directed Acyclic Graph (DAG). We’ll include every piece of code from the process, explain its role, and highlight how this pipeline fosters collaboration between platform and machine learning engineers.

## Why MLOps and Airflow?

MLOps bridges the gap between model development and production deployment, ensuring models are continuously updated and monitored. Platform engineers provide the infrastructure—cloud resources, container orchestration, and data storage—while MLOps engineers focus on model training, deployment, and monitoring. Apache Airflow, an open-source workflow orchestration platform, is ideal for managing these complex pipelines. Its DAGs define tasks and dependencies in Python, making it flexible and powerful for automating machine learning workflows.

In this guide, we’ll:
- Download a movie review dataset from Kaggle.
- Prepare the data for training.
- Fine-tune the Gemma-2-9b-it LLM using Low-Rank Adaptation (LoRA).
- Deploy the model on GKE with vLLM for high-performance inference.
- Orchestrate the pipeline with an Airflow DAG running daily.

The pipeline uses `gcloud` commands for transparency, avoiding high-level abstractions like Terraform, so you can understand each step from both platform and ML perspectives.

## Target Audience

This guide is for:
- Machine Learning Engineers
- Platform Engineers
- Data Scientists
- Data Engineers
- DevOps Engineers
- Platform Architects
- Customer Engineers

It assumes familiarity with GKE and ML concepts but provides detailed explanations for each step.

## Step 1: Setting Up Google Cloud

Before diving into the pipeline, you need a Google Cloud project. Here’s how to set it up.

### Code: Verify Authentication and Project

```bash
gcloud auth list
```

**Explanation**: This command lists the authenticated Google Cloud accounts in Cloud Shell, ensuring you’re logged in. The output shows the active account, e.g., `<my_account>@<my_domain.com>`.

```bash
gcloud config list project
```

**Explanation**: This confirms the active project ID. If it’s not set, you can update it with:

```bash
gcloud config set project <PROJECT_ID>
```

**Explanation**: This sets the specified project ID as the active project, ensuring all subsequent `gcloud` commands target the correct project.

### Prerequisites

1. Sign in to the [Google Cloud Console](http://console.cloud.google.com/) and create a project. Enable billing (new users get a $300 free trial).
2. Activate Cloud Shell, a browser-based command-line environment with `gcloud` preinstalled.
3. Sign up for [Kaggle](https://www.kaggle.com/) and [Hugging Face](https://huggingface.co/). Generate API tokens:
   - Kaggle: Download `kaggle.json` from the API section.
   - Hugging Face: Create a read token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. Accept the license for the Gemma-2-9b-it model at [https://huggingface.co/google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it).

## Step 2: Creating the Infrastructure

The pipeline runs on GKE, with data stored in Google Cloud Storage (GCS) and container images in Artifact Registry. We’ll set up two node pools: one for training (with a single Nvidia L4 GPU) and one for inference (with two Nvidia L4 GPUs).

### Code: Set Environment Variables and Enable APIs

```bash
# Set environment variables
export CODELAB_PREFIX=mlops-airflow
export PROJECT_NUMBER=$(gcloud projects list --filter="${DEVSHELL_PROJECT_ID}" --format="value(PROJECT_NUMBER)")
SUFFIX=$(echo $RANDOM | md5sum | head -c 4; echo;)
export CLUSTER_NAME=${CODELAB_PREFIX}
export CLUSTER_SA=sa-${CODELAB_PREFIX}
export BUCKET_LOGS_NAME=${CODELAB_PREFIX}-logs-${SUFFIX}
export BUCKET_DAGS_NAME=${CODELAB_PREFIX}-dags-${SUFFIX}
export BUCKET_DATA_NAME=${CODELAB_PREFIX}-data-${SUFFIX}
export REPO_NAME=${CODELAB_PREFIX}-repo
export REGION=us-central1
export PROJECT_ID=${DEVSHELL_PROJECT_ID}

# Enable Google APIs
gcloud config set project ${PROJECT_ID}
gcloud services enable \
container.googleapis.com \
cloudbuild.googleapis.com \
artifactregistry.googleapis.com \
storage.googleapis.com
```

**Explanation**: These commands define environment variables for naming resources (e.g., cluster, buckets) and enable APIs for GKE, Cloud Build, Artifact Registry, and GCS. The `SUFFIX` ensures unique bucket names, and `REGION` specifies where resources are deployed (choose a region supporting Nvidia L4 GPUs).

### Code: Create VPC and GKE Cluster

```bash
# Create a VPC for the GKE cluster
gcloud compute networks create mlops --subnet-mode=auto

# Create an IAM Service Account
gcloud iam service-accounts create ${CLUSTER_SA} --display-name="SA for ${CLUSTER_NAME}"
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} \
  --member "serviceAccount:${CLUSTER_SA}@${DEVSHELL_PROJECT_ID}.iam.gserviceaccount.com" \
  --role roles/container.defaultNodeServiceAccount

# Create a GKE cluster
gcloud container clusters create ${CLUSTER_NAME} \
  --zone ${REGION}-a \
  --num-nodes=4 \
  --network=mlops \
  --create-subnetwork name=mlops-subnet \
  --enable-ip-alias \
  --addons GcsFuseCsiDriver \
  --workload-pool=${DEVSHELL_PROJECT_ID}.svc.id.goog \
  --no-enable-insecure-kubelet-readonly-port \
  --service-account=${CLUSTER_SA}@${DEVSHELL_PROJECT_ID}.iam.gserviceaccount.com

# Create training node pool with 1 Nvidia L4 GPU
gcloud container node-pools create training \
  --accelerator type=nvidia-l4,count=1,gpu-driver-version=latest \
  --project=${PROJECT_ID} \
  --location=${REGION}-a \
  --node-locations=${REGION}-a \
  --cluster=${CLUSTER_NAME} \
  --machine-type=g2-standard-12 \
  --num-nodes=1

# Create inference node pool with 2 Nvidia L4 GPUs
gcloud container node-pools create inference \
  --accelerator type=nvidia-l4,count=2,gpu-driver-version=latest \
  --project=${PROJECT_ID} \
  --location=${REGION}-a \
  --node-locations=${REGION}-a \
  --cluster=${CLUSTER_NAME} \
  --machine-type=g2-standard-24 \
  --num-nodes=1

# Download Kubernetes credentials
gcloud container clusters get-credentials ${CLUSTER_NAME} --location ${REGION}-a
```

**Explanation**: 
- The VPC (`mlops`) isolates the GKE cluster’s network.
- A service account (`sa-mlops-airflow`) is created with permissions to manage GKE nodes.
- The GKE cluster is configured with 4 nodes, GCS Fuse for storage integration, and workload identity for secure access.
- Two node pools are added: `training` (g2-standard-12 with 1 L4 GPU) for model fine-tuning and `inference` (g2-standard-24 with 2 L4 GPUs) for serving the model.
- Kubernetes credentials are downloaded to allow `kubectl` commands.

### Code: Create Artifact Registry

```bash
# Create Artifact Registry
gcloud artifacts repositories create ${REPO_NAME} \
  --repository-format=docker \
  --location=${REGION}
gcloud artifacts repositories add-iam-policy-binding ${REPO_NAME} \
  --member=serviceAccount:${CLUSTER_SA}@${DEVSHELL_PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/artifactregistry.reader \
  --location=${REGION}
```

**Explanation**: This creates a Docker repository (`mlops-airflow-repo`) in Artifact Registry to store container images and grants the service account read access.

### Code: Create GCS Buckets

```bash
gcloud storage buckets create gs://${BUCKET_LOGS_NAME} --location=${REGION}
gcloud storage buckets create gs://${BUCKET_DAGS_NAME} --location=${REGION}
gcloud storage buckets create gs://${BUCKET_DATA_NAME} --location=${REGION}
```

**Explanation**: Three GCS buckets are created for logs (`mlops-airflow-logs-<suffix>`), DAGs (`mlops-airflow-dags-<suffix>`), and data (`mlops-airflow-data-<suffix>`), all in the specified region.

### Code: Kubernetes Manifests

Create a `manifests` directory and define the following YAML files:

```bash
mkdir manifests
cd manifests
```

**mlops-sa.yaml**

```yaml
apiVersion: v1
kind: ServiceAccount
automountServiceAccountToken: true
metadata:
  name: airflow-mlops-sa
  namespace: airflow
  labels:
    tier: airflow
```

**Explanation**: Defines a Kubernetes service account (`airflow-mlops-sa`) in the `airflow` namespace for Airflow to interact with GKE resources.

**pv-dags.yaml**

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: airflow-dags
spec:
  accessModes:
  - ReadWriteMany
  capacity:
    storage: 5Gi
  storageClassName: standard
  mountOptions:
    - implicit-dirs
  csi:
    driver: gcsfuse.csi.storage.gke.io
    volumeHandle: BUCKET_DAGS_NAME
    volumeAttributes:
      gcsfuseLoggingSeverity: warning
```

**Explanation**: Creates a persistent volume (`airflow-dags`) backed by the GCS bucket for DAGs, allowing multiple pods to read and write with 5Gi storage.

**pv-logs.yaml**

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: airflow-logs
spec:
  accessModes:
  - ReadWriteMany
  capacity:
    storage: 100Gi
  storageClassName: standard
  mountOptions:
    - implicit-dirs
  csi:
    driver: gcsfuse.csi.storage.gke.io
    volumeHandle: BUCKET_LOGS_NAME
    volumeAttributes:
      gcsfuseLoggingSeverity: warning
```

**Explanation**: Similar to `pv-dags.yaml`, but for logs with 100Gi storage.

**pvc-dags.yaml**

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: airflow-dags
  namespace: airflow
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
  volumeName: airflow-dags
  storageClassName: standard
```

**Explanation**: Binds the `airflow-dags` persistent volume to a claim in the `airflow` namespace.

**pvc-logs.yaml**

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: airflow-logs
  namespace: airflow
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  volumeName: airflow-logs
  storageClassName: standard
```

**Explanation**: Binds the `airflow-logs` persistent volume.

**namespace.yaml**

```yaml
kind: Namespace
apiVersion: v1
metadata:
  name: airflow
  labels:
    name: airflow
```

**Explanation**: Creates the `airflow` namespace for all Airflow-related resources.

**sa-role.yaml**

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: airflow
  name: airflow-deployment-role
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["create", "get", "list", "watch", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["services"]
  verbs: ["create", "get", "list", "watch", "patch", "update", "delete"]
```

**Explanation**: Defines a role allowing management of deployments and services in the `airflow` namespace.

**sa-rolebinding.yaml**

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: airflow-deployment-rolebinding
  namespace: airflow
subjects:
- kind: ServiceAccount
  name: airflow-worker
  namespace: airflow
roleRef:
  kind: Role
  name: airflow-deployment-role
  apiGroup: rbac.authorization.k8s.io
```

**Explanation**: Binds the `airflow-worker` service account to the `airflow-deployment-role`.

**inference.yaml**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-deployment
  namespace: airflow
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gemma-server
  template:
    metadata:
      labels:
        app: gemma-server
        ai.gke.io/model: gemma-2-9b-it
        ai.gke.io/inference-server: vllm
      annotations:
        gke-gcsfuse/volumes: "true"
    spec:
      serviceAccountName: airflow-mlops-sa
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
      - key: "on-demand"
        value: "true"
        operator: "Equal"
        effect: "NoSchedule"
      containers:
      - name: inference-server
        image: vllm/vllm-openai:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: "2"
          limits:
            nvidia.com/gpu: "2"
        command: ["/bin/sh", "-c"]
        args:
        - |
          python3 -m vllm.entrypoints.api_server --model=/modeldata/fine_tuned_model --tokenizer=/modeldata/fine_tuned_model --tensor-parallel-size=2
        volumeMounts:
        - mountPath: /dev/shm
          name: dshm
        - name: gcs-fuse-csi-ephemeral
          mountPath: /modeldata
          readOnly: true
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory
      - name: gcs-fuse-csi-ephemeral
        csi:
          driver: gcsfuse.csi.storage.gke.io
          volumeAttributes:
            bucketName: BUCKET_DATA_NAME
            mountOptions: "implicit-dirs,file-cache:enable-parallel-downloads:true,file-cache:max-parallel-downloads:-1"
            fileCacheCapacity: "20Gi"
            fileCacheForRangeRead: "true"
            metadataStatCacheCapacity: "-1"
            metadataTypeCacheCapacity: "-1"
            metadataCacheTTLSeconds: "-1"
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4
```

**Explanation**: Defines a deployment for the inference server using vLLM, requesting 2 Nvidia L4 GPUs, mounting the fine-tuned model from GCS, and running on GPU-enabled nodes.

**inference-service.yaml**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-service
  namespace: airflow
spec:
  selector:
    app: gemma-server
  type: LoadBalancer
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
```

**Explanation**: Exposes the inference deployment as a LoadBalancer service on port 8000.

### Code: Apply Kubernetes Resources

```bash
# Create the namespace in GKE
kubectl apply -f namespace.yaml

# Update bucket names in YAMLs
sed -i "s/BUCKET_DAGS_NAME/${BUCKET_DAGS_NAME}/g" pv-dags.yaml
sed -i "s/BUCKET_LOGS_NAME/${BUCKET_LOGS_NAME}/g" pv-logs.yaml
sed -i "s/BUCKET_DATA_NAME/${BUCKET_DATA_NAME}/g" inference.yaml

# Apply Kubernetes manifests
kubectl apply -f pv-dags.yaml
kubectl apply -f pv-logs.yaml
kubectl apply -f pvc-dags.yaml
kubectl apply -f pvc-logs.yaml
kubectl apply -f mlops-sa.yaml
kubectl apply -f sa-role.yaml
kubectl apply -f sa-rolebinding.yaml
```

**Explanation**: Creates the `airflow` namespace, updates YAML files with bucket names, and applies the manifests to set up persistent storage and RBAC.

### Code: Assign IAM Roles

```bash
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} \
  --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${DEVSHELL_PROJECT_ID}.svc.id.goog/subject/ns/airflow/sa/airflow-scheduler" \
  --role "roles/storage.objectUser"
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} \
  --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${DEVSHELL_PROJECT_ID}.svc.id.goog/subject/ns/airflow/sa/airflow-triggerer" \
  --role "roles/storage.objectUser"
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} \
  --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${DEVSHELL_PROJECT_ID}.svc.id.goog/subject/ns/airflow/sa/airflow-worker" \
  --role "roles/storage.objectUser"
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} \
  --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${DEVSHELL_PROJECT_ID}.svc.id.goog/subject/ns/airflow/sa/airflow-worker" \
  --role "roles/container.developer"
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} \
  --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${DEVSHELL_PROJECT_ID}.svc.id.goog/subject/ns/airflow/sa/airflow-mlops-sa" \
  --role "roles/artifactregistry.reader"
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} \
  --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${DEVSHELL_PROJECT_ID}.svc.id.goog/subject/ns/airflow/sa/airflow-webserver" \
  --role "roles/storage.objectUser"
```

**Explanation**: Grants storage and container permissions to Airflow components (scheduler, triggerer, worker, webserver, and custom service account) using workload identity federation for secure access.

## Step 3: Installing Airflow on GKE

Airflow orchestrates the pipeline using a Helm chart.

### Code: Airflow Helm Values

**values.yaml**

```yaml
config:
  webserver:
    expose_config: true
webserver:
  service:
    type: LoadBalancer
  podAnnotations:
    gke-gcsfuse/volumes: "true"
executor: KubernetesExecutor
extraEnv: |
  - name: AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL
    value: "30"
logs:
  persistence:
    enabled: true
    existingClaim: "airflow-logs"
dags:
  persistence:
    enabled: true
    existingClaim: "airflow-dags"
scheduler:
  podAnnotations:
    gke-gcsfuse/volumes: "true"
triggerer:
  podAnnotations:
    gke-gcsfuse/volumes: "true"
workers:
  podAnnotations:
    gke-gcsfuse/volumes: "true"
```

**Explanation**: Configures Airflow to:
- Expose the webserver as a LoadBalancer.
- Use KubernetesExecutor for task execution.
- Persist logs and DAGs to GCS-backed volumes.
- Refresh the DAG directory every 30 seconds.
- Enable GCS Fuse for pod storage access.

### Code: Deploy Airflow

```bash
helm repo add apache-airflow https://airflow.apache.org
helm repo update
helm upgrade --install airflow apache-airflow/airflow --namespace airflow -f values.yaml
```

**Explanation**: Adds the Airflow Helm repository and deploys Airflow 2 in the `airflow` namespace with the specified configuration.

## Step 4: Configuring Airflow

Access the Airflow UI to set up connections and variables.

### Code: Get Airflow UI IP

```bash
kubectl -n airflow get svc/airflow-webserver --output jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

**Explanation**: Retrieves the external IP of the Airflow webserver. Navigate to `http://<EXTERNAL-IP>:8080` and log in with `admin/admin`.

### Configuration Steps

1. **Create a GCP Connection**:
   - Go to Admin → Connections → Add a new record.
   - Set:
     - Connection ID: `google_cloud_default`
     - Connection Type: Google Cloud
   - Save.

2. **Create Variables**:
   - Go to Admin → Variables → Add a new record.
   - Add:
     - Key: `BUCKET_DATA_NAME`, Value: `<output of echo $BUCKET_DATA_NAME>`
     - Key: `GCP_PROJECT_ID`, Value: `<output of echo $DEVSHELL_PROJECT_ID>`
     - Key: `HF_TOKEN`, Value: `<your Hugging Face token>`
     - Key: `KAGGLE_USERNAME`, Value: `<your Kaggle username>`
     - Key: `KAGGLE_KEY`, Value: `<API key from kaggle.json>`
   - Save each pair.

**Explanation**: The GCP connection enables Airflow to interact with Google Cloud services. Variables store configuration details used by the DAG, such as bucket names and API tokens.

## Step 5: Building the Pipeline Containers

The pipeline consists of four tasks, each containerized and stored in Artifact Registry.

### Task 1: Data Download

This task downloads the Rotten Tomatoes dataset from Kaggle and uploads it to GCS.

#### Code: Directory Setup

```bash
cd .. ; mkdir 1-dataset-download
cd 1-dataset-download
```

**Explanation**: Creates a directory for the data download task.

#### Code: Python Script

**dataset-download.py**

```python
import os
import kagglehub
from google.cloud import storage

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
BUCKET_DATA_NAME = os.getenv("BUCKET_DATA_NAME")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

# Download latest version
path = kagglehub.dataset_download("priyamchoksi/rotten-tomato-movie-reviews-1-44m-rows")
print("Path to dataset files:", path)
destination_blob_name = "rotten_tomatoes_movie_reviews.csv"
source_file_name = f"{path}/{destination_blob_name}"
upload_blob(BUCKET_DATA_NAME, source_file_name, destination_blob_name)
```

**Explanation**: 
- Authenticates with Kaggle using environment variables.
- Downloads the dataset (`rotten_tomatoes_movie_reviews.csv`).
- Uploads it to the GCS bucket specified by `BUCKET_DATA_NAME`.

#### Code: Dockerfile

**Dockerfile**

```dockerfile
FROM python:3.13.0-slim-bookworm
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY dataset-download.py .
CMD ["python", "dataset-download.py"]
```

**Explanation**: Builds a lightweight Python container, installs dependencies, copies the script, and runs it.

#### Code: Requirements

**requirements.txt**

```
google-cloud-storage==2.19.0
kagglehub==0.3.4
```

**Explanation**: Specifies dependencies for GCS access and Kaggle API.

#### Code: Build and Push Container

```bash
gcloud builds submit --tag ${REGION}-docker.pkg.dev/${DEVSHELL_PROJECT_ID}/${REPO_NAME}/data
```

**Explanation**: Builds the container image and pushes it to Artifact Registry.

### Task 2: Data Preparation

This task processes the dataset into a format suitable for fine-tuning.

#### Code: Directory Setup

```bash
cd .. ; mkdir 2-data-preparation
cd 2-data-preparation
```

**Explanation**: Creates a directory for the data preparation task.

#### Code: Python Script

**data-preparation.py**

```python
import os
import pandas as pd
import gcsfs
import json
from datasets import Dataset

# Environment variables
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BUCKET_DATA_NAME = os.getenv("BUCKET_DATA_NAME")
DATASET_NAME = os.getenv("DATASET_NAME", "rotten_tomatoes_movie_reviews.csv")
PREPARED_DATASET_NAME = os.getenv("PREPARED_DATA_URL", "prepared_data.jsonl")
DATASET_LIMIT = int(os.getenv("DATASET_LIMIT", "100"))  # Process a limited number of rows
DATASET_URL = f"gs://{BUCKET_DATA_NAME}/{DATASET_NAME}"
PREPARED_DATASET_URL = f"gs://{BUCKET_DATA_NAME}/{PREPARED_DATASET_NAME}"

# Load the dataset
print(f"Loading dataset from {DATASET_URL}...")

def transform(data):
    """Transforms a row of the DataFrame into the desired format for fine-tuning."""
    question = f"Review analysis for movie '{data['id']}'"
    context = data['reviewText']
    answer = data['scoreSentiment']
    template = "Question: {question}\nContext: {context}\nAnswer: {answer}"
    return {'text': template.format(question=question, context=context, answer=answer)}

try:
    df = pd.read_csv(DATASET_URL, nrows=DATASET_LIMIT)
    print(f"Dataset loaded successfully.")
    # Drop rows with NaN values in relevant columns
    df = df.dropna(subset=['id', 'reviewText', 'scoreSentiment'])
    # Apply transformation to the DataFrame
    transformed_data = df.apply(transform, axis=1).tolist()
    # Convert transformed data to a DataFrame and then to a Hugging Face Dataset
    transformed_df = pd.DataFrame(transformed_data)
    dataset = Dataset.from_pandas(transformed_df)
    # Save the prepared dataset to JSON lines format
    with gcsfs.GCSFileSystem(project=GCP_PROJECT_ID).open(PREPARED_DATASET_URL, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Prepared dataset saved to {PREPARED_DATASET_URL}")
except Exception as e:
    print(f"Error during data loading or preprocessing: {e}")
    import traceback
    print(traceback.format_exc())
```

**Explanation**:
- Loads the CSV from GCS into a Pandas DataFrame, limiting to `DATASET_LIMIT` rows (100 for testing, 1000 in the DAG).
- Drops rows with missing values in `id`, `reviewText`, or `scoreSentiment`.
- Transforms each row into a question-answering format (e.g., `Question: Review analysis for movie 'id'\nContext: reviewText\nAnswer: scoreSentiment`).
- Saves the result as `prepared_data.jsonl` in GCS.

#### Code: Dockerfile

**Dockerfile**

```dockerfile
FROM python:3.13.0-slim-bookworm
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY data-preparation.py .
CMD ["python", "data-preparation.py"]
```

**Explanation**: Similar to the data download Dockerfile, but for the preparation script.

#### Code: Requirements

**requirements.txt**

```
datasets==3.1.0
gcsfs==2024.9.0
pandas==2.2.3
```

**Explanation**: Includes libraries for dataset handling, GCS access, and data manipulation.

#### Code: Build and Push Container

```bash
gcloud builds submit --tag ${REGION}-docker.pkg.dev/${DEVSHELL_PROJECT_ID}/${REPO_NAME}/data-prep
```

**Explanation**: Builds and pushes the data preparation image.

### Task 3: Model Fine-Tuning

This task fine-tunes the Gemma-2-9b-it model using LoRA.

#### Code: Directory Setup

```bash
cd .. ; mkdir 3-fine-tuning
cd 3-fine-tuning
```

**Explanation**: Creates a directory for the fine-tuning task.

#### Code: Python Script

**finetuning.py**

```python
import os
import torch
import bitsandbytes
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from google.cloud import storage

# Environment variables
BUCKET_DATA_NAME = os.environ["BUCKET_DATA_NAME"]
PREPARED_DATA_URL = os.getenv("PREPARED_DATA_URL", "prepared_data.jsonl")
new_model = os.getenv("NEW_MODEL_NAME", "fine_tuned_model")
model_name = os.getenv("MODEL_ID", "google/gemma-2-9b-it")
save_model_path = os.getenv("MODEL_PATH", "./output")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer loaded successfully!")

# Load dataset
EOS_TOKEN = tokenizer.eos_token
dataset = load_dataset(
    "json", data_files=f"gs://{BUCKET_DATA_NAME}/{PREPARED_DATA_URL}", split="train")
print(dataset)

# LoRA parameters
lora_r = int(os.getenv("LORA_R", "8"))
lora_alpha = int(os.getenv("LORA_ALPHA", "16"))
lora_dropout = float(os.getenv("LORA_DROPOUT", "0.1"))

# TrainingArguments parameters
num_train_epochs = int(os.getenv("EPOCHS", 1))
fp16 = False
bf16 = False
per_device_train_batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "1"))
per_device_eval_batch_size = 1
gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_strategy = os.getenv("CHECKPOINT_SAVE_STRATEGY", "steps")
save_total_limit = int(os.getenv("CHECKPOINT_SAVE_TOTAL_LIMIT", "5"))
save_steps = int(os.getenv("CHECKPOINT_SAVE_STEPS", "1000"))
logging_steps = 50

# SFT parameters
max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "512"))
packing = False

# Load base model
print(f"Loading base model started")
model = AutoModelForCausalLM.from_pretrained(
    attn_implementation="eager",
    pretrained_model_name_or_path=model_name,
    torch_dtype=torch.float16,
)
model.config.use_cache = False
model.config.pretraining_tp = 1
print("Loading base model completed")

# Configure fine-tuning with LoRA
print(f"Configuring fine tuning started")
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# Set training parameters
training_arguments = SFTConfig(
    bf16=bf16,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
    dataset_text_field="text",
    disable_tqdm=True,
    fp16=fp16,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    group_by_length=group_by_length,
    log_on_each_node=False,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    max_grad_norm=max_grad_norm,
    max_seq_length=max_seq_length,
    max_steps=max_steps,
    num_train_epochs=num_train_epochs,
    optim=optim,
    output_dir=save_model_path,
    packing=packing,
    per_device_train_batch_size=per_device_train_batch_size,
    save_strategy=save_strategy,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    warmup_ratio=warmup_ratio,
    weight_decay=weight_decay,
)

# Initialize the SFTTrainer
print(f"Creating trainer started")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
print(f"Creating trainer completed")

# Finetune the model
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning completed.")

# Save the fine-tuned model
print("Saving new model started")
trainer.model.save_pretrained(new_model)
print("Saving new model completed")

# Merge LoRA weights with the base model
print(f"Merging the new model with base model started")
base_model = AutoModelForCausalLM.from_pretrained(
    low_cpu_mem_usage=True,
    pretrained_model_name_or_path=model_name,
    return_dict=True,
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(
    model=base_model,
    model_id=new_model,
)
model = model.merge_and_unload()
print(f"Merging the new model with base model completed")

accelerator = Accelerator()
print(f"Accelerate unwrap model started")
unwrapped_model = accelerator.unwrap_model(model)
print(f"Accelerate unwrap model completed")
print(f"Save unwrapped model started")
unwrapped_model.save_pretrained(
    is_main_process=accelerator.is_main_process,
    save_directory=save_model_path,
    save_function=accelerator.save,
)
print(f"Save unwrapped model completed")
print(f"Save new tokenizer started")
if accelerator.is_main_process:
    tokenizer.save_pretrained(save_model_path)
print(f"Save new tokenizer completed")

# Upload the model to GCS
def upload_to_gcs(bucket_name, model_dir):
    """Uploads a directory to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    for root, _, files in os.walk(model_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            gcs_file_path = os.path.relpath(local_file_path, model_dir)
            blob = bucket.blob(os.path.join(new_model, gcs_file_path))
            blob.upload_from_filename(local_file_path)

upload_to_gcs(BUCKET_DATA_NAME, save_model_path)
print(f"Fine-tuned model {new_model} successfully uploaded to GCS.")
```

**Explanation**:
- Loads the Gemma-2-9b-it model and tokenizer from Hugging Face.
- Configures LoRA with parameters like `lora_r=8` for efficient fine-tuning.
- Loads the prepared dataset (`prepared_data.jsonl`) from GCS.
- Trains the model using `SFTTrainer` with FP16 quantization for 1 epoch.
- Merges LoRA weights with the base model and saves the result to GCS.

#### Code: Dockerfile

**Dockerfile**

```dockerfile
# Using the NVIDIA CUDA base image
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

# Install necessary system packages
RUN apt-get update && \
    apt-get -y --no-install-recommends install python3-dev gcc python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the container
COPY requirements.txt .

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy your finetune script into the container
COPY finetuning.py .

# Set the environment variable to ensure output is flushed
ENV PYTHONUNBUFFERED 1
ENV MODEL_ID "google/gemma-2-9b-it"
ENV GCS_BUCKET "finetuning-data-bucket"

# Set the command to run the finetuning script with CUDA device
CMD ["python3", "finetuning.py"]
```

**Explanation**: Uses an Nvidia CUDA base image for GPU support, installs Python dependencies, and runs the fine-tuning script.

#### Code: Requirements

**requirements.txt**

```
accelerate==1.1.1
bitsandbytes==0.45.0
datasets==3.1.0
gcsfs==2024.9.0
peft==v0.13.2
torch==2.5.1
transformers==4.47.0
trl==v0.11.4
```

**Explanation**: Includes libraries for model training, LoRA, and GCS access.

#### Code: Build and Push Container

```bash
gcloud builds submit --tag ${REGION}-docker.pkg.dev/${DEVSHELL_PROJECT_ID}/${REPO_NAME}/finetuning:latest
```

**Explanation**: Builds and pushes the fine-tuning image.

## Step 6: Defining the Airflow DAG

The DAG orchestrates the four tasks: data download, preparation, fine-tuning, and model serving.

### Code: DAG Definition

**mlops-dag.py**

```python
import yaml
from os import path
from datetime import datetime
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes import client, config
from kubernetes.client import models
from kubernetes.client.rest import ApiException

GCP_PROJECT_ID = Variable.get("GCP_PROJECT_ID")
BUCKET_DATA_NAME = Variable.get("BUCKET_DATA_NAME")
HF_TOKEN = Variable.get("HF_TOKEN")
KAGGLE_USERNAME = Variable.get("KAGGLE_USERNAME")
KAGGLE_KEY = Variable.get("KAGGLE_KEY")
JOB_NAMESPACE = Variable.get("JOB_NAMESPACE", default_var="airflow")

def model_serving():
    config.load_incluster_config()
    k8s_apps_v1 = client.AppsV1Api()
    k8s_core_v1 = client.CoreV1Api()
    while True:
        try:
            k8s_apps_v1.delete_namespaced_deployment(
                namespace="airflow",
                name="inference-deployment",
                body=client.V1DeleteOptions(
                    propagation_policy="Foreground", grace_period_seconds=5
                )
            )
        except ApiException:
            break
    print("Deployment inference-deployment deleted")
    with open(path.join(path.dirname(__file__), "inference.yaml")) as f:
        dep = yaml.safe_load(f)
        resp = k8s_apps_v1.create_namespaced_deployment(
            body=dep, namespace="airflow")
        print(f"Deployment created. Status='{resp.metadata.name}'")
    while True:
        try:
            k8s_core_v1.delete_namespaced_service(
                namespace="airflow",
                name="llm-service",
                body=client.V1DeleteOptions(
                    propagation_policy="Foreground", grace_period_seconds=5
                )
            )
        except ApiException:
            break
    print("Service llm-service deleted")
    with open(path.join(path.dirname(__file__), "inference-service.yaml")) as f:
        dep = yaml.safe_load(f)
        resp = k8s_core_v1.create_namespaced_service(
            body=dep, namespace="airflow")
        print(f"Service created. Status='{resp.metadata.name}'")

with DAG(
    dag_id="mlops-dag",
    start_date=datetime(2024, 11, 1),
    schedule_interval="@daily",
    catchup=False
) as dag:
    # DAG Step 1: Fetch raw data to GCS Bucket
    dataset_download = KubernetesPodOperator(
        task_id="dataset_download_task",
        namespace=JOB_NAMESPACE,
        image="us-central1-docker.pkg.dev/{{ var.value.GCP_PROJECT_ID }}/mlops-airflow-repo/dataset-download:latest",
        name="dataset-download",
        service_account_name="airflow-mlops-sa",
        env_vars={
            "KAGGLE_USERNAME": KAGGLE_USERNAME,
            "KAGGLE_KEY": KAGGLE_KEY,
            "BUCKET_DATA_NAME": BUCKET_DATA_NAME
        }
    )

    # DAG Step 2: Run GKEJob for data preparation
    data_preparation = KubernetesPodOperator(
        task_id="data_pipeline_task",
        namespace=JOB_NAMESPACE,
        image="us-central1-docker.pkg.dev/{{ var.value.GCP_PROJECT_ID }}/mlops-airflow-repo/data-preparation:latest",
        name="data-preparation",
        service_account_name="airflow-mlops-sa",
        env_vars={
            "GCP_PROJECT_ID": GCP_PROJECT_ID,
            "BUCKET_DATA_NAME": BUCKET_DATA_NAME,
            "DATASET_LIMIT": "1000",
            "HF_TOKEN": HF_TOKEN
        }
    )

    # DAG Step 3: Run GKEJob for fine tuning
    fine_tuning = KubernetesPodOperator(
        task_id="fine_tuning_task",
        namespace=JOB_NAMESPACE,
        image="us-central1-docker.pkg.dev/{{ var.value.GCP_PROJECT_ID }}/mlops-airflow-repo/finetuning:latest",
        name="fine-tuning",
        service_account_name="airflow-mlops-sa",
        startup_timeout_seconds=600,
        container_resources=models.V1ResourceRequirements(
            requests={"nvidia.com/gpu": "1"},
            limits={"nvidia.com/gpu": "1"}
        ),
        env_vars={
            "BUCKET_DATA_NAME": BUCKET_DATA_NAME,
            "HF_TOKEN": HF_TOKEN
        }
    )

    # DAG Step 4: Run GKE Deployment for model serving
    model_serving = PythonOperator(
        task_id="model_serving",
        python_callable=model_serving
    )

    dataset_download >> data_preparation >> fine_tuning >> model_serving
```

**Explanation**:
- Defines a daily DAG starting November 1, 2024, with no catch-up runs.
- **Task 1**: `dataset_download` runs the data download container, passing Kaggle credentials and bucket name.
- **Task 2**: `data_preparation` runs the data preparation container, limiting to 1000 rows.
- **Task 3**: `fine_tuning` runs the fine-tuning container, requesting 1 GPU.
- **Task 4**: `model_serving` deploys the inference server by deleting and recreating the deployment and service defined in `inference.yaml` and `inference-service.yaml`.
- Dependencies ensure tasks run sequentially: `dataset_download → data_preparation → fine_tuning → model_serving`.

### Code: Upload DAG and Manifests

```bash
gcloud storage cp mlops-dag.py gs://${BUCKET_DAGS_NAME}
gcloud storage cp manifests/inference.yaml gs://${BUCKET_DAGS_NAME}
gcloud storage cp manifests/inference-service.yaml gs://${BUCKET_DAGS_NAME}
```

**Explanation**: Uploads the DAG and Kubernetes manifests to the DAGs bucket, making them available to Airflow.

## Step 7: Running and Testing the Pipeline

In the Airflow UI:
1. Unpause the `mlops-dag`.
2. Trigger it manually to run the pipeline.

### Code: Test the Model

```bash
export MODEL_ENDPOINT=$(kubectl -n airflow get svc/llm-service --output jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -X POST http://${MODEL_ENDPOINT}:8000/generate -H "Content-Type: application/json" -d @- <<EOF
{
    "prompt": "Question: Review analysis for movie 'dangerous_men_2015'",
    "temperature": 0.1,
    "top_p": 1.0,
    "max_tokens": 128
}
EOF
```

**Explanation**:
- Retrieves the external IP of the `llm-service`.
- Sends a POST request to the inference endpoint with a prompt about the movie `dangerous_men_2015`.
- Expected output: A response like `Answer: POSITIVE`, based on the fine-tuned model’s analysis.

## Why Fine-Tuning Over RAG?

The pipeline fine-tunes the LLM instead of using Retrieval-Augmented Generation (RAG). Here’s why:
- **Fine-Tuning**: Creates a specialized model tailored to the dataset, simplifying inference and reducing latency. It’s ideal for static datasets like movie reviews.
- **RAG**: Retrieves external data for each query, suitable for dynamic knowledge bases but adds complexity with microservices and databases.

Fine-tuning aligns with Airflow’s focus on workflow orchestration, keeping the pipeline streamlined.

## Taking It to Production

For production:
- Add a web frontend with [Gradio](https://www.gradio.app/).
- Enable monitoring with GKE’s [automatic application monitoring](https://cloud.google.com/kubernetes-engine/docs/how-to/configure-automatic-application-monitoring) or Airflow’s [Prometheus exporter](https://cloud.google.com/stackdriver/docs/managed-prometheus/exporters/airflow).
- Scale with larger GPUs or [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) for distributed training.
- Use [Google Cloud Composer](https://cloud.google.com/composer), a managed Airflow service, to simplify maintenance.

## Conclusion

You’ve built a robust MLOps pipeline that automates data ingestion, preparation, model fine-tuning, and deployment using Airflow on GKE. This setup not only streamlines machine learning workflows but also fosters collaboration between platform and ML engineers, breaking down silos and accelerating deployment. Explore the [Airflow documentation](https://airflow.apache.org/) or Google Cloud’s MLOps resources to enhance your pipeline further.

---

This article includes every code snippet from the document, with detailed explanations to clarify their purpose and functionality. It balances technical depth with accessibility, ensuring readers can follow the pipeline’s construction and understand its components. Let me know if you need adjustments, such as a different tone, additional focus areas, or further elaboration on any section!
