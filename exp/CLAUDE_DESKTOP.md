# Instruções completas para o Claude Code no desktop (RTX 3060)

Caminhos: servidor `/root/xai-sdv-artifact/exp/CLAUDE_DESKTOP.md`; GitHub https://github.com/ojaneri/xai-sdv-artifact/blob/exp/e1-perception/exp/CLAUDE_DESKTOP.md

Cole este arquivo inteiro como primeiro prompt da sessão, ou diga ao Claude:
"Leia exp/CLAUDE_DESKTOP.md e siga-o."

---

## 1. Contexto

Você roda no desktop Windows do autor (Osvaldo Janeri Filho): RTX 3060 12 GB, 16 GB de RAM.
A missão é executar os experimentos que respondem ao desk reject do IEEE OJVT
(OJVT-2026-09-0870). O editor disse que o componente AEB do artigo é "ilustrativo, não
avaliado experimentalmente". Cada experimento troca um placeholder `\tbd{...}` do
manuscrito por um valor **medido**.

| Exp. | O quê | Runbook | Precisa |
|---|---|---|---|
| E1 | SMIRK: detector + patch adversarial + saliência/OOD + custo | `exp/E1/RUNBOOK.md` | 3060 |
| E2 | CARLA: fusão câmera+LiDAR sob ataque, janela TTC 4–3 s | `exp/E2/RUNBOOK.md` | 3060, disco conforme o passo 0 do runbook |
| E5 | Bancada CAN: IDS + TreeSHAP com a GPU ocupada | `exp/E5/README.md` | CANable + Nucleo/ESP32 (chegam até 17/out) |
| E6 | Custo em AWS ARM e ARM+GPU | `exp/E6/README.md` | créditos AWS (pedido em andamento) |

**Ordem:** E1 primeiro. O E2 só começa depois que o E1 estiver no passo 7 ou tiver
falhado e a falha estiver documentada. Os passos do E5 que exigem hardware só rodam
quando o usuário disser que a bancada está montada. O E6 só roda com autorização
explícita do usuário.

## 2. Endereços e caminhos

**Repositório de trabalho:** `exp/` contém os runbooks.
- GitHub (público): https://github.com/ojaneri/xai-sdv-artifact, branch `exp/e1-perception`.
- Espelho no servidor: `/root/xai-sdv-artifact`, clonável com
  `ssh://root@janeri2.janeri.com.br:2222/root/xai-sdv-artifact`.

**Servidor:** `janeri2.janeri.com.br` (23.81.118.139), SSH na **porta 2222**, usuário
`root`. No `ssh` a porta vai em `-p 2222`; no `scp`, em `-P 2222`. O disco do servidor
está 87% cheio: só arquivos pequenos vão para lá.

| Caminho no servidor | Para quê | Sua permissão |
|---|---|---|
| `/root/xai-results/` | destino dos resultados | escrever, só em subpastas novas |
| `/root/xai-sdv-artifact/` | espelho das instruções | só leitura |
| qualquer outro caminho | — | **não tocar** |

**No desktop:**
- Clone em `%USERPROFILE%\xai-sdv-artifact`, ou onde o usuário indicar.
- Arquivos pesados (datasets, pesos, vídeos, CARLA) ficam em `exp/<E>/work/`. Essa pasta
  está no `.gitignore`.
- Cada experimento tem seu venv em `exp/<E>/.venv`.

## 3. Setup (uma vez)

```powershell
nvidia-smi                                   # tem de listar a RTX 3060
git --version; python --version              # Python 3.11 ou 3.12
ssh -p 2222 -o BatchMode=yes root@janeri2.janeri.com.br "echo ok"
```

Se o último comando falhar, peça ao usuário para autorizar a chave. **Não** tente senha e
**não** gere nem copie chaves sozinho. O comando que o usuário deve rodar é:

```powershell
ssh-keygen -t ed25519
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh -p 2222 root@janeri2.janeri.com.br "cat >> ~/.ssh/authorized_keys"
```

Depois, clone o repositório e configure a autoria:

```powershell
git clone -b exp/e1-perception https://github.com/ojaneri/xai-sdv-artifact.git
cd xai-sdv-artifact
git config user.name  "Osvaldo Janeri Filho"
git config user.email "janeri@gmail.com"
```

## 4. Regras que não se negociam

1. **Nunca invente um número.** Todo valor em `results/` vem de um comando que você rodou
   nesta sessão, e o JSON registra o comando, o commit, a seed e o ambiente. Sem medição,
   o campo fica `null` com o motivo.
2. **Nunca fabrique dados, rótulos ou citações.** Se o SMIRK, um modelo ou um dataset não
   estiver disponível, pare e reporte. Não troque por outro sem perguntar.
3. **Falhas são resultado.** Registre-as em `exp/<E>/NOTES.md` com data, comando, erro e o
   que tentou. Não apague tentativas que falharam.
4. **Nada de ajuste para "dar certo".** Thresholds e hiperparâmetros são escolhidos no split
   de validação, nunca no de teste. Se você olhou o teste antes, declare isso em NOTES.md.
5. **Autoria:** commits como `Osvaldo Janeri Filho <janeri@gmail.com>`, **sem** linha
   `Co-Authored-By` e sem citar Claude ou qualquer modelo como autor.

## 5. Proteções: peça confirmação ao usuário antes de

- gastar dinheiro (AWS, APIs pagas, compra de dataset);
- instalar drivers, CUDA de sistema, WSL ou qualquer coisa fora de um venv ou da pasta
  `work/`;
- apagar qualquer coisa fora de `exp/<E>/work/`;
- `git push --force`, `git reset --hard`, rebase ou reescrever histórico (**proibido**:
  peça e espere);
- qualquer comando no servidor além de `mkdir`, `scp` e `ls` em `/root/xai-results/`;
- baixar mais de 20 GB de uma vez, ou continuar quando o disco livre cair abaixo de 30 GB.

**Nunca:**
- commite credenciais, chaves `.pem`, tokens, `~/.aws`, datasets, pesos (`*.pt`, `*.pth`,
  `*.onnx`) ou vídeos;
- mexa no Overleaf, no manuscrito (`main.tex`) ou em `/var/www` no servidor;
- desligue o antivírus, o firewall ou o UAC;
- deixe instância AWS rodando. No E6, o script encerra a instância; confira com o comando
  de "leftover check" do README.

## 6. Como trabalhar

1. Leia o runbook do experimento inteiro antes de executar.
2. Antes de cada passo, diga em uma frase o que vai fazer. Depois, diga o que saiu, com o
   número e o caminho do arquivo.
3. Use processos longos em segundo plano com log em `exp/<E>/work/logs/`. Não fique
   esperando em silêncio: informe o progresso.
4. Ao fim de cada passo do runbook, commite os resultados pequenos com a mensagem
   `E1 step N: <o que foi medido>`.
5. Se o tempo estimado de um passo passar de 2× o previsto, pare e pergunte.

## 7. Devolver os resultados

**A. Git (preferido):**

```powershell
git add exp/E1/results exp/E1/NOTES.md exp/E1/SUMMARY.md
git commit -m "E1: <resumo>"
git push origin exp/e1-perception
```

Se o push pedir credencial que você não tem, peça ao usuário e siga com a opção B.

**B. Cópia para o servidor (sempre, além do git):**

```powershell
$E  = "E1"
$TS = Get-Date -Format yyyyMMddTHHmmss
ssh -p 2222 root@janeri2.janeri.com.br "mkdir -p /root/xai-results/$E/$TS"
scp -P 2222 -r exp/$E/results exp/$E/NOTES.md exp/$E/SUMMARY.md "root@janeri2.janeri.com.br:/root/xai-results/$E/$TS/"
ssh -p 2222 root@janeri2.janeri.com.br "du -sh /root/xai-results/$E/$TS; ls -R /root/xai-results/$E/$TS | head -50"
```

Antes do scp, confira que `exp/$E/results` tem **menos de 200 MB**. Se passar, não envie:
avise o usuário.

**`SUMMARY.md`** (um por experimento) traz:
- uma tabela com cada valor medido, sua unidade, o IC, o n e o arquivo JSON de origem;
- o que falhou ou não foi feito, e por quê;
- quais `\tbd{...}` do manuscrito cada valor preenche. Os rótulos estão nos runbooks, por
  exemplo `E1: saliency mass on box`.

## 8. Ao terminar a sessão

Diga ao usuário:
- quais passos foram concluídos;
- o commit hash enviado ao GitHub;
- o caminho em `/root/xai-results/...`;
- o que ficou pendente.

O servidor preenche o manuscrito a partir dali.
