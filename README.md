# ContAttriEval

Code for submitted paper: A Contrastive Framework for Enhanced Automatic Attribution
Evaluation Through Error Generation

In this paper we follow a two-stage methodology to enhance model performances:

1. Analysis of Attribution evaluation Error types by examining 3 SOTA models (T5-XXL-True, AlignScore, Qwen3-4B)
2. Model Enhancement:
   - Synthetic Error Generation
   - Model Training through a contrastive objective

![Methodology](media/pipeline.jpg)
# Stage 1: 
We provide extended examples of all error types identified in our Prelimaniry analysis and according to our taxonomy:

| Category | Example |
| :--- | :--- |
| **Label error (E1)** | **$s_i$**: To determine whether a drug is an agonist or antagonist of a receptor in molecular pharmacology, C) you will need to combine functional and binding experiments [1] [2]. <br><br> **$C_i$**: ["Access Denied... Your access to the NCBI website at www.ncbi.nlm.nih.gov has been temporarily blocked due to a possible misuse/abuse situation..."] |
| **Subjective (E3)** | **$s_i$**: The belief in Greek gods is called Ancient Greek religion. <br><br> **$C_i$**: ["Title: Ancient Greek religion. Most ancient Greeks recognized the twelve major Olympian gods and goddesses—Zeus, Hera, Poseidon, Demeter, Athena, Ares, Aphrodite, Apollo, Artemis, Hephaestus, Hermes, and either Hestia or Dionysus..."] <br><br> **NLI**: 1; **ALIGNSCORE**: 0.98; **Label**: Not Attributable <br> *Interpretation of 'called' as the name of the religion or the definition.* |
| **Subjective (E3) cont.** | **$s_i$**: UNESCO inaugurated 'World Teachers' Day'. <br><br> **$C_i$**: ["Title: World Teachers' Day... Established in 1994, it commemorates the signing of the 1966 UNESCO/ILO Recommendation concerning the Status of Teachers..."] <br><br> **NLI**: 0; **ALIGNSCORE**: 0.182; **Label**: Attributable <br> *Interpretation of 'inaugurated' and 'established'.* |
| **Relevance / Token Overlap** | **$s_i$**: The road that connects the tombs is called the spirit way. <br><br> **$C_i$**: ["Title: Ming tombs. A 7-kilometer road named the 'Spirit Way' (pinyin: Shéndào) leads into the complex..."] <br><br> **NLI**: 1; **ALIGNSCORE**: 0.97; **Label**: Not Attributable <br> *Interpretation: mention the name and describing the roads.* |
| **Relevance / Token Overlap cont.** | **$s_i$**: Rocky Dzidzornu plays the bongos on 'Sympathy for the Devil.' <br><br> **$C_i$**: ["Title: Rocky Dzidzornu. Critic Ned Sublette has written that the addition of his conga drumming on 'Sympathy for the Devil' transformed the song from 'a dirge, and a dull one at that . . . making it come alive'."] <br><br> **NLI**: 0; **ALIGNSCORE**: 0.017; **Label**: Attributable <br> *Conflict: bongos vs conga.* |
| **Fine-grained sensitivity (E5)** | **$s_i$**: At an altitude of 7000 **meters** above sea level, water boils at approximately 92.7°C (198.9°F). <br><br> **$C_i$**: At 7,000 **feet** water boils 92.7C (198.9F). <br><br> **NLI**: 1; **ALIGNSCORE**: 0.91; **Label**: Not Attributable |
| **Conflicting Sources (E9)** | **$s_i$**: The best selling album of 2017 was Taylor Swift’s "Reputation". [1][2] <br><br> **$C_i$**: [1] Taylor Swift’s "Reputation" was the best selling album of 2017... [2] ...In the US, the album sold 1.216 million copies... making it **the country’s best-selling album**, while globally it was the **second best-selling album** of 2017 worldwide. <br><br> **NLI**: 1; **ALIGNSCORE**: 0.98; **Label**: Not Attributable |
| **Fact Synthesis (E7)** | **$s_i$**: The first overseas branch of Bible Students was opened in London in 1900, and a German branch office of the Watch Tower Society opened in Elberfeld in 1902. <br><br> **$C_i$**: [1] A German branch office of the Watch Tower Society opened in Elberfeld in 1902. [2] Bible Student missionaries were sent to England in 1881 and the first overseas branch was opened in London in 1900. <br><br> **NLI**: 0; **ALIGNSCORE**: 0.0; **Label**: Attributable |

# Stage 2
## Synthetic Error Generation
Different prompts for generating different types of errors can be found in:
```
prompts.json
```

To augment a dataset with generated errors:
```
python error_gen/scripts/gen_negative_examples.py --data_file {file}  --prompt_model_name Qwen/Qwen3-30B-A3B-Instruct-2507

```
## Model Training

We repurpose alignement-handbook to implement the training code in model_training

# Results
We report the full results on AttributuonBench and TrueBenchmark

### TrueBench Evaluation Results (AUC-ROC)

| Type | Metric | SE | PAWS | Q2 | VitC | FRR | FRK | DF | MNBM | Q-C | Q-X | BEGIN | AVG | AVG-OOD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **QA** | QAFactEval | 80.9 | 86.1 | 75.8 | 73.6 | 86.0 | 88.5 | 81.8 | 67.3 | 83.9 | 76.1 | 81.0 | 80.1 | 79.4 |
| **Similarity** | ROUGE-2 | 79.4 | 68.6 | 61.4 | 59.9 | 55.5 | 84.5 | 67.7 | 65.0 | 78.4 | 60.2 | 82.8 | 69.4 | 72.4 |
| **Matching** | BLEU | 74.8 | 71.3 | 55.2 | 56.1 | 51.7 | 84.1 | 61.2 | 56.7 | 77.4 | 54.7 | 74.6 | 65.2 | 67.3 |
| **Matching** | SimSC | 70.2 | 69.2 | 66.2 | 63.8 | 72.7 | 72.9 | 70.6 | 64.6 | 74.9 | 56.5 | 86.1 | 69.8 | 70.3 |
| **NLI** | MNLI | 44.6 | 81.3 | 71.8 | 80.2 | 93.1 | 57.2 | 76.5 | 59.1 | 42.6 | 50.1 | 81.5 | 67.1 | 60.4 |
| **NLI** | DAE | 60.3 | 55.8 | 57.7 | 60.2 | 77.8 | 77.9 | 54.7 | 81.0 | 56.9 | 67.5 | 69.4 | 65.4 | 65.7 |
| **NLI** | SummaC-ZS | 77.6 | 89.0 | 81.8 | 97.2 | 92.8 | 86.9 | 87.1 | 58.0 | 76.0 | 75.3 | 83.2 | 82.2 | 78.2 |
| **NLI** | SummaC-Conv | 79.1 | 88.2 | 77.5 | 97.5 | 92.0 | 89.0 | 81.2 | 67.2 | 77.7 | 76.0 | 81.6 | 82.5 | 78.7 |
| **NLI** | T5-XXL-TRUE (11B) | 80.5 | 86.4 | 72.7 | 88.3 | 93.2 | 89.4 | 77.7 | 77.9 | 82.1 | 83.8 | 82.6 | 83.1| 81.5 |
| **Misc** | UniEval | 81.2 | 80.1 | 70.4 | 79.1 | 92.1 | 88.1 | 80.4 | 66.8 | 86.5 | 76.7 | 73.6 | 79.5 | 78.0 |
| **Misc** | CTC | 79.8 | 63.1 | 66.8 | 65.0 | 72.5 | 87.1 | 63.7 | 65.0 | 77.3 | 67.7 | 72.0 | 70.9 | 72.4 |
| **Misc** | BARTScore | 78.9 | 77.1 | 65.1 | 64.2 | 66.1 | 87.8 | 60.8 | 63.5 | 83.9 | 60.2 | 86.7 | 72.2 | 73.4 |
| **Misc** | ALIGNSCORE-large | 82.9 | 98.4 | 78.6 | 98.3 | 94.9 | 92.1 | 85.1 | 76.1 | 89.5 | 83.5 | 82.7 | 87.4 | 83.8 |
| **Ours** | **Qwen3-4B_sft** | 81.2 | 89.2 | 82.7 | 94.8 |  94.5|  88.9 |89.6| 75.5 |82.2| 83.3 | 83.4 |  85.6 |80.1
| **Ours** | **Qwen3-4B_cont_ALL** | 82.6| 90.1| 82.3 | 93.7 | 94.9|  89.0 | 89.8 |  74.2| 85.6|  84.1| 82.7|  86.3| 83.7
