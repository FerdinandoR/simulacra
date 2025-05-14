# Simulacra

Use real patients' data to benchmark and validate your AI model, use `Simulacra` to train them.

`Simulacra` provides a framework to generate synthetic patient tabular data to be drawn from the same distribution as that of a given input dataset. It has been designed with longevity application in mind, therefore with DNA methylation, proteomics, metabolomics etc. are its prime focus. However, it can be applied to other contexts as well, and we welcome contributions that do so.

## Why `Simulacra`?
There are several reasons why you would want to train your AI models on synthetically augmented data rather than real data:
* **Abundance**: Real data is limited and often so scarce that it bottlenecks performance. You can generate as much synthetic data as you'd like.
* **Memory**: Real data has to be stored in memory in ordered to be processed, both in the RAM at time and training and on hard drives for storage between usage sessions. On the other hand, synthetic data can be generated on-the-fly in a reproducible manner and be deleted when it's no longer of use.
* **Privacy**: Real data is obtained from real people, therefore its use and dissemination must be strictly regulated to protect patient privacy. Conversely, synthetic data is not directly linked to single individuals, and can therefore be shared freely.
* **Accessibility**: Access to real data is often 

## Funzionalità
- Generazione realistica di pazienti sintetici
- Supporto a configurazioni personalizzabili
- Esportazione in diversi formati

## Esempio d’uso
```bash
python scripts/generate_dataset.py --output data/synthetic.csv
```

## Requisiti
Vedi `requirements.txt`.

## Licenza
[Inserire tipo di licenza]
