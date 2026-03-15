# Diatoms Classification API — Documentação

API REST para classificação de diatomáceas usando modelos CNN (ResNet50V2).
Construída com **FastAPI**, autenticação via **Google OAuth2**, armazenamento de imagens no **Cloudflare R2** e banco de dados no **Cloudflare D1**.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Configuração do Ambiente](#configuração-do-ambiente)
- [Rodando sem Docker](#rodando-sem-docker)
- [Rodando com Docker](#rodando-com-docker)
- [Autenticação](#autenticação)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Auth](#auth)
  - [Processamento de Imagem](#processamento-de-imagem)
  - [Imagens](#imagens)
  - [Modelos e Predição](#modelos-e-predição)
  - [Histórico](#histórico)
- [Erros](#erros)

---

## Visão Geral

| Item | Detalhe |
|---|---|
| **Base URL** | `http://localhost:7860` |
| **Autenticação** | Bearer token (Google ID Token) |
| **Formatos aceitos** | PNG, JPEG, BMP, TIFF (máx. 10 MB) |
| **Modelos disponíveis** | `model_7k`, `model_10k`, `model_22k` |
| **Documentação interativa** | `GET /docs` (Swagger UI) |

**Fluxo completo de uso:**

```
Login Google → Tratar imagem → Upload → Predição → Histórico
```

---

## Configuração do Ambiente

### 1. Pré-requisitos

- Python 3.12+
- Arquivo `api/.env` preenchido com todas as credenciais

### 2. Criar o `api/.env`

Copie o exemplo e preencha com suas credenciais reais:

```bash
cp api/.env.example api/.env
```

Conteúdo do `api/.env`:

```env
GOOGLE_CLIENT_ID=seu_client_id.apps.googleusercontent.com
R2_ACCESS_KEY_ID=sua_access_key
R2_SECRET_ACCESS_KEY=sua_secret_key
CLOUDFLARE_ACCOUNT_ID=seu_account_id
D1_DATABASE_ID=seu_database_id
D1_API_TOKEN=seu_api_token
```

> O arquivo `api/.env` está no `.gitignore` e nunca será versionado.

---

## Rodando sem Docker

Indicado para **desenvolvimento local**. Qualquer alteração no código é refletida imediatamente sem rebuild.

### 1. Instalar dependências

Na raiz do repositório, com o venv ativado:

```bash
./setup.sh
```

Ou manualmente:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r api/requirements.txt
```

### 2. Ativar o venv

```bash
source venv/bin/activate
```

### 3. Iniciar a API

Execute **a partir da raiz do repositório**:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload
```

A flag `--reload` reinicia automaticamente ao salvar qualquer arquivo.

### 4. Verificar que está rodando

```
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:7860
```

> Os avisos de CUDA são normais em máquinas sem GPU. O TensorFlow usa CPU automaticamente.

### 5. Testar

```bash
curl http://localhost:7860/health
# {"success":true,"message":"API is healthy","data":{"status":"ok"}}
```

Ou acesse `http://localhost:7860/docs` para a interface Swagger.

---

## Rodando com Docker

Indicado para **deploy e testes de produção**. Reproduz exatamente o ambiente que será usado no Hugging Face Spaces.

### Pré-requisitos

- Docker instalado ([instruções de instalação Linux](https://docs.docker.com/engine/install/ubuntu/))
- Arquivos `.keras` dos modelos presentes localmente em `CNN/models/`

Verificar se os modelos estão no lugar:

```bash
find CNN/models -name "*.keras"
# CNN/models/modelo_7k/fineTuned_model_7k/Diatom_Classifier_FineTuned_Model_7k.keras
# CNN/models/modelo_10k/fineTuned_model_10k/Diatom_Classifier_FineTuned_Model_10k.keras
# CNN/models/modelo_22k/fineTuned_model_22k/Diatom_Classifier_FineTuned_Model_22k.keras
```

### 1. Build da imagem

Execute na raiz do repositório:

```bash
docker build -t diatoms-api .
```

A primeira execução demora (~5–15 min) pois baixa Python 3.11 e instala TensorFlow, rembg, OpenCV, etc. Execuções seguintes são muito mais rápidas pelo cache de camadas.

### 2. Rodar o container

```bash
docker run --env-file api/.env -p 7860:7860 diatoms-api
```

| Flag | Significado |
|---|---|
| `--env-file api/.env` | Injeta as variáveis do `.env` no container |
| `-p 7860:7860` | Mapeia a porta do container para a máquina local |

### 3. Rodar em background

```bash
# Iniciar em background
docker run -d --env-file api/.env -p 7860:7860 --name diatoms diatoms-api

# Ver logs em tempo real
docker logs -f diatoms

# Parar
docker stop diatoms

# Remover container (não apaga a imagem)
docker rm diatoms
```

### 4. Testar

```bash
curl http://localhost:7860/health
# {"success":true,"message":"API is healthy","data":{"status":"ok"}}
```

### 5. Comandos úteis

```bash
# Listar imagens Docker locais
docker images

# Listar containers rodando
docker ps

# Remover a imagem (para rebuild limpo)
docker rmi diatoms-api

# Rebuild sem cache
docker build --no-cache -t diatoms-api .
```

---

## Autenticação

A maioria dos endpoints requer autenticação via **Google ID Token**.

### Fluxo

1. O cliente faz login com Google (frontend usa `@react-oauth/google`)
2. Google retorna um **ID Token** (JWT)
3. O cliente envia o token para `POST /api/auth/login`
4. A API valida o token com a biblioteca `google-auth` e retorna o perfil do usuário
5. Para chamadas subsequentes, o cliente inclui o token no header:

```
Authorization: Bearer <google_id_token>
```

### Endpoints que não requerem autenticação

- `GET /health`
- `POST /api/auth/login`
- `GET /api/models`

---

## Endpoints

> **Contrato vigente (resumo):** respostas de sucesso usam envelope `{ "success": true, "message": "...", "data": ... }`. Respostas de erro usam `{ "success": false, "message": "...", "code": ..., "error": "...", "detail": ... }`.

### Health Check

#### `GET /health`

Verifica se a API está no ar. Usado por serviços de monitoramento.

**Respostas:**

| Código | Situação |
|---|---|
| `200 OK` | API operacional |

```json
{
  "success": true,
  "message": "API is healthy",
  "data": {
    "status": "ok"
  }
}
```

---

### Auth

#### `POST /api/auth/login`

Valida um Google ID Token e cria ou recupera o usuário no banco.

**Request body:**

```json
{
  "token": "eyJhbGciOiJSUzI1NiIs..."
}
```

**Respostas:**

| Código | Situação |
|---|---|
| `200 OK` | Login realizado, retorna perfil do usuário |
| `401 Unauthorized` | Token inválido ou expirado |

`200 OK`:
```json
{
  "success": true,
  "message": "Login successful",
  "data": {
    "id": "116234567890123456789",
    "email": "usuario@gmail.com",
    "name": "Nome Sobrenome",
    "created_at": "2026-02-28T18:00:00"
  }
}
```

`401 Unauthorized`:
```json
{"success": false, "message": "Request failed", "code": 401, "error": "Invalid Google token", "detail": "Invalid Google token"}
```

---

#### `GET /api/auth/me`

Retorna o perfil do usuário autenticado.

**Header requerido:** `Authorization: Bearer <token>`

**Respostas:**

| Código | Situação |
|---|---|
| `200 OK` | Retorna perfil do usuário autenticado |
| `401 Unauthorized` | Token ausente ou inválido |
| `404 Not Found` | Usuário não encontrado no banco |

`200 OK`:
```json
{
  "success": true,
  "message": "User profile retrieved",
  "data": {
    "id": "116234567890123456789",
    "email": "usuario@gmail.com",
    "name": "Nome Sobrenome",
    "created_at": "2026-02-28T18:00:00"
  }
}
```

`model_used` segue exatamente os IDs dos modelos da API: `"model_7k"`, `"model_10k"`, `"model_22k"`.

`401 Unauthorized`:
```json
{"success": false, "message": "Request failed", "code": 401, "error": "Invalid or expired token", "detail": "Invalid or expired token"}
```

`404 Not Found`:
```json
{"success": false, "message": "Request failed", "code": 404, "error": "User not found", "detail": "User not found"}
```

---

### Processamento de Imagem

#### `POST /api/process/treat`

Recebe uma imagem bruta, aplica o pipeline de tratamento e salva o resultado no R2 + D1.

**Pipeline de tratamento:**
1. Remoção de fundo (`rembg`)
2. Equalização adaptativa de contraste (CLAHE)
3. Redimensionamento e padding para 400×400 px

**Header requerido:** `Authorization: Bearer <token>`

**Request:** `multipart/form-data`

| Campo | Tipo | Descrição |
|---|---|---|
| `file` | arquivo | Imagem da diatomácea (PNG/JPEG/BMP/TIFF, máx. 10 MB) |

**Exemplo com curl:**

```bash
curl -X POST http://localhost:7860/api/process/treat \
  -H "Authorization: Bearer <token>" \
  -F "file=@/caminho/para/imagem.png"
```

**Respostas:**

| Código | Situação |
|---|---|
| `200 OK` | Imagem tratada e salva com sucesso |
| `401 Unauthorized` | Token ausente ou inválido |
| `413 Request Entity Too Large` | Arquivo maior que 10 MB |
| `422 Unprocessable Entity` | Formato não suportado ou falha no processamento |

`200 OK`:
```json
{
  "success": true,
  "message": "Image treated and saved successfully",
  "data": {
    "image_id": "550e8400-e29b-41d4-a716-446655440000",
    "url": "https://r2.cloudflare.com/...?X-Amz-Expires=3600&..."
  }
}
```

| Campo | Descrição |
|---|---|
| `image_id` | UUID da imagem salva — usar nas próximas chamadas |
| `url` | URL temporária (1h) para visualizar/baixar a imagem tratada |

`401 Unauthorized`:
```json
{"success": false, "message": "Request failed", "code": 401, "error": "Invalid or expired token", "detail": "Invalid or expired token"}
```

`413 Request Entity Too Large`:
```json
{"success": false, "message": "Request failed", "code": 413, "error": "File exceeds the 10 MB limit", "detail": "File exceeds the 10 MB limit"}
```

`422 Unprocessable Entity`:
```json
{"success": false, "message": "Request failed", "code": 422, "error": "Unsupported file type: image/gif", "detail": "Unsupported file type: image/gif"}
```

---

### Imagens

#### `POST /api/images`

Salva uma imagem já tratada diretamente no R2 + D1 (sem aplicar o pipeline).

**Header requerido:** `Authorization: Bearer <token>`

**Request:** `multipart/form-data`

| Campo | Tipo | Descrição |
|---|---|---|
| `file` | arquivo | Imagem (PNG/JPEG/BMP/TIFF, máx. 10 MB) |

**Respostas:**

| Código | Situação |
|---|---|
| `200 OK` | Imagem salva com sucesso |
| `401 Unauthorized` | Token ausente ou inválido |
| `413 Request Entity Too Large` | Arquivo maior que 10 MB |
| `422 Unprocessable Entity` | Formato não suportado |

`200 OK`:
```json
{
  "success": true,
  "message": "Image uploaded successfully",
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "116234567890123456789",
    "r2_key": "users/116.../images/550e....png",
    "original_name": "diatoma.png",
    "created_at": "2026-02-28T18:00:00",
    "url": "https://r2.cloudflare.com/...?X-Amz-Expires=3600&..."
  }
}
```

`model_used` segue exatamente os IDs dos modelos da API: `"model_7k"`, `"model_10k"`, `"model_22k"`.

`401 Unauthorized`:
```json
{"success": false, "message": "Request failed", "code": 401, "error": "Invalid or expired token", "detail": "Invalid or expired token"}
```

`413 Request Entity Too Large`:
```json
{"success": false, "message": "Request failed", "code": 413, "error": "File exceeds the 10 MB limit", "detail": "File exceeds the 10 MB limit"}
```

`422 Unprocessable Entity`:
```json
{"success": false, "message": "Request failed", "code": 422, "error": "Unsupported file type: image/gif", "detail": "Unsupported file type: image/gif"}
```

---

#### `GET /api/images`

Lista todas as imagens salvas pelo usuário autenticado.

**Header requerido:** `Authorization: Bearer <token>`

**Exemplo com curl:**

```bash
curl http://localhost:7860/api/images \
  -H "Authorization: Bearer <token>"
```

**Respostas:**

| Código | Situação |
|---|---|
| `200 OK` | Lista de imagens do usuário (array vazio se não houver nenhuma) |
| `401 Unauthorized` | Token ausente ou inválido |

`200 OK`:
```json
{
  "success": true,
  "message": "Images retrieved successfully",
  "data": [
    {
      "id": "550e8400-...",
      "user_id": "116...",
      "r2_key": "users/.../images/....png",
      "original_name": "diatoma.png",
      "created_at": "2026-02-28T18:00:00",
      "url": "https://r2.cloudflare.com/..."
    }
  ]
}
```

`401 Unauthorized`:
```json
{"success": false, "message": "Request failed", "code": 401, "error": "Invalid or expired token", "detail": "Invalid or expired token"}
```

---

#### `DELETE /api/images/{image_id}`

Remove uma imagem do R2 e do D1.

**Header requerido:** `Authorization: Bearer <token>`

**Exemplo com curl:**

```bash
curl -X DELETE http://localhost:7860/api/images/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer <token>"
```

**Respostas:**

| Código | Situação |
|---|---|
| `200 OK` | Imagem removida com sucesso (com confirmação no body) |
| `401 Unauthorized` | Token ausente ou inválido |
| `404 Not Found` | Imagem não encontrada ou não pertence ao usuário |

`200 OK`:
```json
{
  "success": true,
  "message": "Image deleted successfully",
  "data": {
    "success": true,
    "image_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Image deleted successfully"
  }
}
```

`401 Unauthorized`:
```json
{"success": false, "message": "Request failed", "code": 401, "error": "Invalid or expired token", "detail": "Invalid or expired token"}
```

`404 Not Found`:
```json
{"success": false, "message": "Request failed", "code": 404, "error": "Image not found", "detail": "Image not found"}
```

---

### Modelos e Predição

#### `GET /api/models`

Lista os modelos CNN disponíveis. Não requer autenticação.

**Respostas:**

| Código | Situação |
|---|---|
| `200 OK` | Lista de modelos disponíveis |

`200 OK`:
```json
{
  "success": true,
  "message": "Models retrieved successfully",
  "data": [
    {
      "id": "model_7k",
      "description": "trained with ~7k curated images and dynamic augmentation (best model)"
    },
    {
      "id": "model_10k",
      "description": "trained with ~10k pure images and dynamic augmentation"
    },
    {
      "id": "model_22k",
      "description": "trained with ~22k curated images and 3x augmentations (raw augmentation, images from model_7k x3)"
    }
  ]
}
```

---

#### `POST /api/predict`

Executa inferência em uma imagem salva e persiste o resultado no D1.

**Header requerido:** `Authorization: Bearer <token>`

**Request body:**

```json
{
  "image_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "model_22k"
}
```

| Campo | Tipo | Valores aceitos |
|---|---|---|
| `image_id` | string | UUID de uma imagem salva do usuário |
| `model` | string | `"model_7k"`, `"model_10k"`, `"model_22k"` |

**Exemplo com curl:**

```bash
curl -X POST http://localhost:7860/api/predict \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"image_id": "550e8400-...", "model": "model_22k"}'
```

**Respostas:**

| Código | Situação |
|---|---|
| `200 OK` | Inferência realizada com sucesso |
| `401 Unauthorized` | Token ausente ou inválido |
| `404 Not Found` | `image_id` não encontrado ou não pertence ao usuário |
| `502 Bad Gateway` | Falha de acesso ao objeto no R2 |
| `503 Service Unavailable` | Modelo solicitado não está carregado |

`200 OK`:
```json
{
  "success": true,
  "message": "Prediction completed successfully",
  "data": {
    "predicted_class": "Navicula",
    "confidence": 0.9732,
    "probabilities": {
      "Encyonema": 0.0041,
      "Eunotia": 0.0089,
      "Gomphonema": 0.0063,
      "Navicula": 0.9732,
      "Pinnularia": 0.0075
    }
  }
}
```

| Campo | Descrição |
|---|---|
| `predicted_class` | Gênero da diatomácea classificada |
| `confidence` | Probabilidade da classe predita (0–1) |
| `probabilities` | Probabilidade para cada um dos 5 gêneros |

`401 Unauthorized`:
```json
{"success": false, "message": "Request failed", "code": 401, "error": "Invalid or expired token", "detail": "Invalid or expired token"}
```

`404 Not Found`:
```json
{"success": false, "message": "Request failed", "code": 404, "error": "Image not found", "detail": "Image not found"}
```

`502 Bad Gateway`:
```json
{"success": false, "message": "Request failed", "code": 502, "error": "R2 error: <ErrorCode>", "detail": "R2 error: <ErrorCode>"}
```

`503 Service Unavailable`:
```json
{"success": false, "message": "Request failed", "code": 503, "error": "Model not loaded", "detail": "Model not loaded"}
```

---

### Histórico

#### `GET /api/history`

Retorna todas as classificações do usuário, da mais recente para a mais antiga.

**Header requerido:** `Authorization: Bearer <token>`

**Respostas:**

| Código | Situação |
|---|---|
| `200 OK` | Lista de classificações (array vazio em `items` se não houver nenhuma) |
| `401 Unauthorized` | Token ausente ou inválido |

`200 OK`:
```json
{
  "success": true,
  "message": "History retrieved successfully",
  "data": {
    "items": [
      {
        "id": "7d3f9a1b-...",
        "user_id": "116...",
        "image_id": "550e8400-...",
        "model_used": "model_22k",
        "predicted_class": "Navicula",
        "confidence": 0.9732,
        "probabilities": {
          "Encyonema": 0.0041,
          "Eunotia": 0.0089,
          "Gomphonema": 0.0063,
          "Navicula": 0.9732,
          "Pinnularia": 0.0075
        },
        "created_at": "2026-02-28T18:00:00",
        "image_url": "https://r2.cloudflare.com/..."
      }
    ],
    "total": 1
  }
}
```

`401 Unauthorized`:
```json
{"success": false, "message": "Request failed", "code": 401, "error": "Invalid or expired token", "detail": "Invalid or expired token"}
```

---

#### `GET /api/history/{classification_id}`

Retorna uma classificação específica com a URL presignada da imagem.

**Header requerido:** `Authorization: Bearer <token>`

**Respostas:**

| Código | Situação |
|---|---|
| `200 OK` | Retorna a classificação com URL da imagem |
| `401 Unauthorized` | Token ausente ou inválido |
| `404 Not Found` | Classificação não encontrada |

`200 OK`:
```json
{
  "success": true,
  "message": "Classification retrieved successfully",
  "data": {
    "id": "7d3f9a1b-...",
    "user_id": "116...",
    "image_id": "550e8400-...",
    "model_used": "model_22k",
    "predicted_class": "Navicula",
    "confidence": 0.9732,
    "probabilities": {
      "Encyonema": 0.0041,
      "Eunotia": 0.0089,
      "Gomphonema": 0.0063,
      "Navicula": 0.9732,
      "Pinnularia": 0.0075
    },
    "created_at": "2026-02-28T18:00:00",
    "image_url": "https://r2.cloudflare.com/..."
  }
}
```

`401 Unauthorized`:
```json
{"success": false, "message": "Request failed", "code": 401, "error": "Invalid or expired token", "detail": "Invalid or expired token"}
```

`404 Not Found`:
```json
{"success": false, "message": "Request failed", "code": 404, "error": "Classification not found", "detail": "Classification not found"}
```

---

## Erros

Todos os erros seguem o formato base:

```json
{
  "success": false,
  "message": "<mensagem>",
  "code": 400,
  "error": "Mensagem de erro resumida",
  "detail": "Mensagem descritiva do erro ou lista de validação"
}
```

> Observação: para erros de validação (`422`), `message` é `Validation failed` e `detail` contém a lista de erros do FastAPI/Pydantic.

| Código | Significado |
|---|---|
| `401` | Token ausente, inválido ou expirado |
| `404` | Recurso não encontrado |
| `413` | Arquivo maior que 10 MB |
| `422` | Formato de arquivo não suportado ou body inválido |
| `502` | Erro de integração com R2 (upstream) |
| `503` | Modelo CNN não carregado |
| `500` | Erro interno do servidor |

---

## Classes de Diatomáceas

Os modelos classificam entre 5 gêneros:

| Gênero | Descrição |
|---|---|
| Encyonema |
| Eunotia | 
| Gomphonema |
| Navicula |
| Pinnularia |
