# GT Empower — reconhecimento facial e relatórios

Aplicação web **real**, com React 19 + TypeScript + Tailwind CSS, Python/FastAPI + **DeepFace**, e PostgreSQL. Nenhum dado de exemplo é gerado pelo sistema. O projeto não inclui Spring Boot: o próprio Python expõe a API para evitar uma segunda camada desnecessária entre o DeepFace e o banco.

## Iniciar com Docker (recomendado)

Requisitos: Docker Desktop com Docker Compose, conexão à internet na primeira execução e memória livre suficiente para TensorFlow/RetinaFace.

```bash
cd gt-empower
docker compose up --build
```

Abra **http://localhost:5173**. API: **http://localhost:8000/docs**. Status: **http://localhost:8000/api/health**. O primeiro reconhecimento pode ser lento: DeepFace faz download dos pesos dos modelos e inicializa o TensorFlow. Se a máquina não conseguir baixar os pesos ou não tiver memória suficiente, a API retorna **erro explícito**, sem fabricar correspondências. A instalação das dependências do DeepFace pode exigir alguns minutos e os downloads podem ser volumosos.

## Iniciar sem Docker (desenvolvimento)

Use Python **3.11** para maior compatibilidade com TensorFlow.

```bash
# terminal 1, na raiz do projeto
cd backend
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
export TF_USE_LEGACY_KERAS=1    # Windows PowerShell: $env:TF_USE_LEGACY_KERAS='1'
uvicorn app.main:app --reload --port 8000

# terminal 2
cd frontend
npm install
npm run dev
```

No modo local, sem `DATABASE_URL`, a aplicação utiliza SQLite em `backend/storage/gt_empower.db` como **alternativa de desenvolvimento**, mantendo as mesmas funções. No Docker, utiliza PostgreSQL com volume persistente.

## Fluxo do primeiro teste real

1. Peça a **primeira foto de rosto autorizada** (um rosto visível, de preferência frontal), abra **Pessoas** e envie a foto. O DeepFace detecta o rosto; o sistema mostra se já há cadastro com similaridade acima do limiar.
2. Preencha o **nome** e a **categoria** (Membro, Visitante ou Criança). Clique em **Confirmar cadastro**; rostos já existentes são bloqueados.
3. Envie uma **segunda foto da mesma pessoa**. O sistema detecta o rosto, calcula embedding com Facenet512 e compara a distância de cosseno ao limiar configurado. O resultado mostrado é calculado, não simulado.
4. Crie uma reunião com a data, envie fotografias e faça a conferência. Vincule desconhecidos a cadastros, ignore falsos positivos ou confirme e crie novos registros a partir de um rosto detectado.
5. Em **Relatórios**, selecione a reunião; o total é a quantidade de pessoas **distintas, identificadas e não ignoradas**, não a quantidade de detecções; edite e copie o texto.

## Endpoints

`GET /api/health`, `GET /api/people`, `POST /api/people/inspect`, `POST /api/people`, `POST /api/people/verify`, `GET /api/meetings`, `POST /api/meetings`, `GET /api/meetings/{id}`, `POST /api/meetings/{id}/photos`, `PATCH /api/sightings/{id}`, `POST /api/sightings/{id}/enroll`, `GET /api/files/{filename}`.

## Limitações e cautelas

- O limiar inicial de cosseno `0.30` é **configurável** (`FACE_COSINE_THRESHOLD`), não representa certeza ou acurácia garantida. Iluminação, ângulo, idade e qualidade influenciam o desempenho. Valide em dados autorizados e confira manualmente, sobretudo com crianças.
- Reconhecimento por IA **não deve decidir sozinho** quem esteve presente; o relatório contabiliza pessoas confirmadas/identificadas com deduplicação por `person_id`. Uma pessoa ainda não cadastrada e não revisada **não** é adicionada ao total. Fotografias sem detecção podem gerar erro em vez de contagem zero.
- Uma pessoa pode aparecer mais de uma vez; os rostos detectados ficam em histórico com imagem, área, associação e status de revisão. Fotos são conservadas em volume local. Editar o texto de relatório é uma edição temporária no navegador; não modifica o histórico nem persiste como documento.
- Para produção, **não publique os serviços diretamente na internet** sem autenticação, perfis de acesso, TLS, backups, retenção/eliminação dos dados, logs de auditoria, consentimento/base legal e avaliação de impacto conforme a LGPD. Os arquivos de faces contêm **dados biométricos sensíveis**. O exemplo não implementa essas proteções de implantação.
- Mude a senha do PostgreSQL no `docker-compose.yml` antes de usar, configure políticas de backup e ajuste limite de upload no seu proxy reverso.
