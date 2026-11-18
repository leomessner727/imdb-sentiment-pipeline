# Classificazione distribuita del sentiment su IMDB

Elaborato di Leo Messner (Matr. 0322600155).

Il sistema addestra un modello di **classificazione del sentiment** (recensione positiva o negativa) sul dataset **IMDB** di Hugging Face. Più configurazioni del modello vengono addestrate **in parallelo da worker separati**, ciascuno in un proprio processo; al termine viene scelta la configurazione migliore e valutata sul test set.

- **Dati:** [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb), 25.000 recensioni di training (divise in 80% training e 20% validation) e 25.000 di test.
- **Modello:** regressione logistica su feature TF-IDF (scikit-learn), addestrata da zero.
- **Metriche:** accuracy e F1-score sul test set.

## Architettura

```
                    +--> worker w1 (TF-IDF uni, C=1) --+
                    |                                  |
                    +--> worker w2 (TF-IDF uni, C=4) --+
   prepare  --------+                                  +-------->  select-best
   (passo 1)        +--> worker w3 (TF-IDF bi,  C=1) --+           (passo 3, fan-in)
                    |                                  |
                    +--> worker w4 (TF-IDF bi,  C=4) --+
                         (passo 2, fan-out, 4 processi in parallelo)
```

1. **prepare** (`src/prepare.py`): scarica IMDB, crea lo split training/validation (stratificato, seed fisso) e calcola le matrici TF-IDF per ogni variante di feature.
2. **worker** (`src/worker.py`): ogni worker è un processo separato, addestra **una** configurazione e salva modello e metriche di validation.
3. **select-best** (`src/select_best.py`): raccoglie i risultati di tutti i worker, sceglie la configurazione con l'F1 di validation più alto e la valuta **una sola volta** sul test set.

Il modulo principale `src/pipeline.py` esegue i tre passi in sequenza; nel passo 2 avvia i worker in parallelo con `ProcessPoolExecutor`. Tutti i passi condividono la cartella `data/`.

## Pattern architetturali

### Fan-out / fan-in

Le configurazioni (variante TF-IDF e parametro di regolarizzazione C) sono indipendenti tra loro, quindi il loro addestramento viene distribuito su più worker che lavorano in parallelo (**fan-out**). Il passo successivo attende tutti i worker e unisce i loro risultati per scegliere il modello migliore (**fan-in**). Il tempo totale dipende così dalla configurazione più lenta e non dalla somma di tutte. La scelta avviene sul validation set, così il test set resta indipendente dalla selezione.

### Step memoization

Il download, lo split e soprattutto il calcolo delle matrici TF-IDF sono i passi più costosi e sono uguali per più worker. `src/memo.py` salva l'output di ogni passo su disco con una chiave che è l'hash dei suoi parametri:

- se il passo viene richiesto di nuovo con gli stessi parametri, l'output viene caricato (**cache hit**) invece di essere ricalcolato;
- i worker con la stessa variante TF-IDF riutilizzano le stesse feature, calcolate una sola volta;
- se un parametro cambia (ad esempio `max_features`), cambia l'hash e il passo viene ricalcolato automaticamente;
- anche ogni worker controlla se la sua configurazione è già stata addestrata: dopo un errore, una nuova esecuzione addestra solo le configurazioni mancanti.

## Struttura del repository

```
src/
  config.py       seed, cartelle, dataset, varianti TF-IDF, configurazioni
  utils.py        seed, logging, hash, file JSON
  memo.py         step memoization
  prepare.py      passo 1: dati e feature
  worker.py       passo 2: addestramento di una configurazione
  select_best.py  passo 3: scelta del migliore e valutazione sul test set
  pipeline.py     modulo principale: intera pipeline (multiprocesso)
  predict.py      demo: classifica una recensione con il modello migliore
requirements.txt  dipendenze con versioni fissate (pip)
environment.yml   ambiente conda (Python 3.11 e le stesse dipendenze)
```

## Installazione

Con conda (Anaconda, Miniconda o Miniforge):

```bash
conda env create -f environment.yml
conda activate imdb-sentiment
```

In alternativa con Python 3.11 e pip:

```bash
pip install -r requirements.txt
```

Al primo avvio serve una connessione Internet per scaricare il dataset (circa 80 MB).

Il progetto è stato sviluppato ed eseguito con Miniforge su Windows 11, Python 3.11.

## Esecuzione

```bash
python -m src.pipeline
```

- `prepare` prepara dati e feature, poi i quattro worker partono **in parallelo** come processi separati, infine `select-best` stampa la tabella dei risultati e le metriche sul test set.
- Nella tabella la colonna `pid` mostra il processo che ha addestrato ogni configurazione: quattro numeri diversi, quindi quattro processi distinti.
- Le metriche finali sono salvate in `data/final_metrics.json`.

Una seconda esecuzione mostra la step memoization: `prepare` trova tutto in cache e i worker saltano l'addestramento. Per ripartire da zero basta cancellare la cartella `data/`.

I passi si possono avviare anche singolarmente:

```bash
python -m src.prepare
python -m src.worker --config w1
python -m src.select_best
```

## Classificare una recensione

```bash
python -m src.predict "This movie was really great!"
```

## Riproducibilità

- Seed fisso (`SEED = 42` in `src/config.py`) per lo split e per l'addestramento.
- Versioni delle librerie fissate in `requirements.txt` e `environment.yml`.
- Il TF-IDF viene stimato solo sui dati di training; il test set viene usato una sola volta, alla fine.

## Modificare le configurazioni

Le configurazioni e le varianti TF-IDF sono definite in `src/config.py` (`CONFIGS`, `TFIDF_VARIANTS`). Una nuova voce in `CONFIGS` viene addestrata automaticamente da un worker in più.
