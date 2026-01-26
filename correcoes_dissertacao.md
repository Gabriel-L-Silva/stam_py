# Correções da Dissertação

Lista de correções necessárias baseada nas notas dos revisores: Castelo, Douglas e Fabiano.
**Decisões tomadas em análise conjunta.**

---

## 1. Metodologia e Definições

### 1.1 Solver da equação de Poisson
- **Correção:** Trocar `spsolve` (solver direto) para método iterativo BiCGSTAB com precondicionador
- **Decisão:** IMPLEMENTAR no código e documentar no texto
- **Arquivo:** `code/modules/trisolver.py` linha 123
- **Texto:** Seção 4.1.4 (Projeção do Campo de Velocidades)
- [x] Implementar BiCGSTAB com precondicionador (scipy.sparse.linalg.bicgstab)
- [x] Documentar no texto a escolha do método e justificar

### 1.2 Tipo de malha: Collocated vs Staggered
- **Correção:** Texto diz "deslocada" mas trabalho usa collocated
- **Decisão:** CORRIGIR texto para "collocated"
- **Arquivo:** `thesis/tex/introducao.tex` linha 36
- [ ] Mudar "malhas triangulares deslocadas" para "malhas triangulares co-localizadas"
- [ ] Adicionar definição clara de malha collocated na seção apropriada

### 1.3 Definições faltantes
- [ ] **Definir malha irregular/não-estruturada:** malha onde os elementos não seguem padrão regular de conectividade
- [ ] **Definir geometria complexa:** domínios com fronteiras de alta curvatura ou não-convexos
- [ ] **Definir dados irregulares** (Seção 4.2.3): pontos que não coincidem com vértices da malha

### 1.4 Injetor de fumaça
- **Decisão:** Adicionar subseção na metodologia
- [ ] Criar subseção explicando o funcionamento do injetor
- [ ] Descrever que é uma fonte de velocidade e densidade aplicada em células específicas
- [ ] Mencionar que o injetor é desligado na metade da simulação (frame 500)

---

## 2. Notação e Equações

### 2.1 Tabela de símbolos
- **Decisão:** Criar tabela de símbolos no início do texto
- [ ] Criar tabela com todos os símbolos usados:
  - n: timestep (superscript), número de pontos (contexto), vetor normal (negrito)
  - x*: posição após backtracking
  - i: índice do ponto/nó
  - outros símbolos relevantes

### 2.2 Polinômios P_j (eq 3.16)
- **Decisão:** Mostrar base explícita
- [ ] Adicionar após eq 3.16:
  ```
  Para m=1 (linear): P_1=1, P_2=x, P_3=y
  Para m=2 (quadrático): P_1=1, P_2=x, P_3=y, P_4=x², P_5=xy, P_6=y²
  ```

### 2.3 Definições nas equações
- [ ] **Eq 4.1:** Explicitar que x* é a posição de origem do backtracking
- [ ] **Eq 4.7:** Definir que i é o índice do ponto na malha
- [ ] **Eq 4.11:** Definir claramente o timestep atual
- [ ] **Eq 4.24:** Adicionar discussão sobre quando omega^n_g pode ser zero (configurações geométricas específicas onde derivada normal é zero)

### 2.4 Consistência de notação
- [ ] Verificar uso consistente de negrito para vetores
- [ ] Padronizar notação de ponto (produto escalar vs multiplicação)

---

## 3. Resultados e Visualização

### 3.1 Divergente
- **Decisão:** Colormap 2D na malha (em vez de gráfico 3D)
- [ ] Modificar visualização do divergente para colorir triângulos/vértices
- [ ] Usar colormap (ex: coolwarm) sobre a malha 2D
- [ ] Manter gráfico temporal do divergente máximo

### 3.2 Malhas não-conformes
- **Decisão:** REMOVER do escopo
- [ ] Remover menções a malhas não-conformes na introdução
- [ ] Ajustar objetivos se necessário
- [ ] Focar apenas em malhas não-estruturadas (triangulares arbitrárias)

### 3.3 Visualização melhorada
- [ ] Adicionar malha sobreposta na visualização da fumaça
- [ ] Quebrar resultados em figuras separadas (instante inicial, intermediário, final)
- [ ] Melhorar captions para serem auto-contidos

### 3.4 Interpolação
- [ ] Gerar figuras comparando interpolação baricêntrica vs RBF
- [ ] Usar código existente em `interpolator.py:main()`
- [ ] Mostrar erro de cada método

---

## 4. Figuras

### 4.1 Figura do backtracking
- **Problema:** Trajetória desenhada como curva, mas método é linear
- [ ] Redesenhar com linha RETA de x_i^n para x_i^{n-1}
- **Arquivo:** `thesis/images/backtracking.pdf`

### 4.2 Figura 3.8 (rbffdfd.pdf)
- **Problema:** Mostra caso 1D, revisor pede malha triangular
- [ ] Criar nova figura com exemplo em malha triangular
- [ ] Manter figura 1D como complemento ou substituir

### 4.3 Padrão ABNT
- [ ] Verificar todas as figuras: legenda em cima, fonte embaixo
- [ ] Formato atual parece correto (\caption antes de \includegraphics, \subcaption* para fonte)

---

## 5. Seções Específicas

### 5.1 Seção 4.2 - Interseção com Fronteira
- [ ] **4.2.1:** Citar complexidade do algoritmo: O(E) onde E = número de arestas da fronteira
- [ ] **4.2.2:** Quantificar melhora do Numba (speedup aproximado)
- [ ] **4.2.3:** Definir que usa interpolação baricêntrica para backtracking

### 5.2 Seção 4.3 - Malhas arbitrárias
- [ ] **4.3.1:** Mover análise de precisão RBF para capítulo de Resultados

### 5.3 Seção 4.4 - Tratamento de Fronteiras
- [ ] Explicar delta_x_local: espaçamento médio ou mínimo entre vizinhos
- [ ] Documentar parâmetro alpha (ghost_distance): valor usado = 0.1
- [ ] Explicar cálculo de normal: média das normais das arestas adjacentes ao vértice
- [ ] Adicionar discussão sobre caso omega^n_g = 0 (instabilidade)

### 5.4 Seção 4.5 - Aspectos Computacionais
- [ ] **4.5.2:** Adicionar detalhes OpenGL (shaders, VAO/VBO)
- [x] **4.5.1 (novo):** Adicionada seção sobre resolução do sistema linear (BiCGSTAB + ILU)
- [ ] **4.5.4:** Especificar KD-Tree do scipy/libpysal para busca de vizinhança
- [x] **4.5.4:** Adicionar versões em apêndice (usar requirements.txt)

---

## 6. Aspectos Numéricos (Castelo)

### 6.1 CFL e termo convectivo
- [ ] Comentar que sem viscosidade, termo convectivo domina (primeira ordem)
- [ ] Enaltecer que CFL limita o backtracking a distâncias pequenas
- [ ] Mencionar que método da projeção é de primeira ordem

---

## 7. Texto e Estrutura

### 7.1 Conexão objetivos-resultados
- [x] Revisar objetivos após remoção de malhas não-conformes
- [x] Garantir que resultados demonstrem os objetivos definidos

### 7.2 Definição precisão vs acurácia
- [ ] Adicionar definição clara:
  - Precisão: repetibilidade/consistência dos resultados
  - Acurácia: proximidade do valor verdadeiro

### 7.3 Detalhamento do texto
- [ ] Expandir revisão bibliográfica com mais contexto e motivação
- [ ] Adicionar imagens ilustrativas onde apropriado

---

## 8. Apêndices

### 8.1 Versões das bibliotecas
- **Decisão:** Adicionar requirements.txt como apêndice
- [x] Criar apêndice com tabela de dependências
- [x] Incluir: numpy, scipy, numba, trimesh, triangle, networkx, libpysal, etc.

---

## Itens REMOVIDOS do escopo

Os seguintes itens foram decididos como fora do escopo desta revisão:

- ~~Malhas conforme vs não conforme~~ (removido - foco em malhas não-estruturadas)
- ~~Tempo de execução e profiling~~ (não incluir análise de performance detalhada)
- ~~Mudar resolução da malha~~ (não fazer análise de convergência)

---

## Resumo de ações por arquivo

### Código
| Arquivo | Ação |
|---------|------|
| `trisolver.py` | Trocar spsolve por bicgstab |

### Texto LaTeX
| Arquivo | Ações |
|---------|-------|
| `introducao.tex` | Corrigir "deslocada"→"collocated", remover malhas não-conformes |
| `fundteorica.tex` | Adicionar polinômios P_j, tabela de símbolos |
| `metodologia.tex` | Adicionar subseção injetor, definições, detalhes computacionais |
| `resultados.tex` | Mudar visualização divergente, figuras comparativas |

### Figuras
| Figura | Ação |
|--------|------|
| `backtracking.pdf` | Redesenhar com linha reta |
| `rbffdfd.pdf` | Criar versão com malha triangular |

---

## Ordem sugerida de execução

1. **Fase 1 - Texto básico:**
   - Criar tabela de símbolos
   - Corrigir collocated vs staggered
   - Remover malhas não-conformes
   - Adicionar definições faltantes

2. **Fase 2 - Equações e detalhes:**
   - Adicionar polinômios P_j
   - Definir símbolos nas equações
   - Explicar parâmetros (alpha, delta_x, normal)

3. **Fase 3 - Figuras:**
   - Corrigir backtracking.pdf
   - Criar figura RBF-FD triangular
   - Verificar padrão ABNT

4. **Fase 4 - Resultados:**
   - Mudar visualização divergente para 2D
   - Gerar comparação interpoladores
   - Melhorar captions

5. **Fase 5 - Código:** ✅ CONCLUÍDA
   - ✅ Implementar BiCGSTAB
   - ✅ Adicionar requirements.txt como apêndice

6. **Fase 6 - Revisão final:** ✅ CONCLUÍDA
   - ✅ Verificar conexão objetivos-resultados
   - ✅ Revisar consistência geral
