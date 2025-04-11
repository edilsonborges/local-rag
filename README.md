# LocalRAG - Assistente SIDAGO Local

Este é um sistema de Recuperação Aumentada por Geração (RAG) que funciona localmente, usando SQLite para armazenamento e Ollama para a geração de texto.

## Características

- Base de dados SQLite local (sem necessidade de PostgreSQL)
- Servidor LLM local usando Ollama com o modelo hermes-3-llama-3.2-3b@q4_k_m
- Interface web com chat em tempo real
- Upload de documentos PDF via drag-and-drop
- Processamento e indexação automática de documentos
- Busca semântica em documentos processados

## Requisitos

- Python 3.8+
- Ollama (instalado automaticamente durante o primeiro uso)
- Espaço em disco suficiente para o modelo LLM (~4GB)

## Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/localrag.git
   cd localrag
   ```

2. Crie um ambiente virtual e instale as dependências:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Execute o servidor:
   ```bash
   uvicorn localrag:app --reload --host 0.0.0.0 --port 8000
   ```

4. Acesse a interface web em http://localhost:8000

## Como usar

1. **Adicionar documentos**:
   - Arraste e solte arquivos PDF na área designada
   - Ou clique no botão "Escolher arquivo" para selecionar um PDF

2. **Fazer perguntas**:
   - Digite sua pergunta na caixa de texto na parte inferior
   - Pressione Enter ou clique em "Enviar"
   - O sistema irá buscar informações relevantes nos documentos e gerar uma resposta

## Como funciona

1. **Processamento de documentos**:
   - Os PDFs são convertidos em texto
   - O texto é dividido em chunks menores
   - Cada chunk é transformado em um embedding de vetor
   - Os embeddings são armazenados no SQLite

2. **Busca semântica**:
   - Quando você faz uma pergunta, ela é convertida em um vetor
   - O sistema encontra os chunks mais similares no banco de dados
   - Os chunks relevantes são usados como contexto para o LLM

3. **Geração de resposta**:
   - O modelo LLM (hermes-3-llama-3.2-3b@q4_k_m) recebe a pergunta e o contexto
   - O modelo gera uma resposta baseada apenas nas informações fornecidas
   - A resposta é transmitida em tempo real via streaming

## Arquitetura

- **Frontend**: HTML/CSS/JavaScript puro
- **Backend**: FastAPI
- **Banco de dados**: SQLite
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)
- **LLM**: hermes-3-llama-3.2-3b@q4_k_m via Ollama

## Solução de Problemas

1. **Erro ao iniciar o servidor LLM**:
   - Verifique se o Ollama está instalado corretamente
   - Execute `ollama list` para verificar se o modelo foi baixado
   - Execute `ollama pull hermes-3-llama-3.2-3b@q4_k_m` manualmente se necessário

2. **Erros de memória**:
   - O modelo LLM requer cerca de 4GB de RAM
   - Feche outros aplicativos para liberar memória

3. **Processamento lento de PDFs**:
   - Documentos grandes podem levar tempo para processar
   - O progresso é mostrado na interface

## Limitações

- O modelo LLM é pequeno (3B parâmetros) e pode ter limitações na qualidade das respostas
- A busca vetorial em SQLite não é otimizada como em bancos de dados específicos para vetores
- Documentos muito grandes podem causar lentidão no processamento

## Requisitos Python

Veja o arquivo `requirements.txt` para a lista completa de dependências.