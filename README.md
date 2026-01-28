Este repositório é um fork do [projeto original](https://github.com/Asap7772/fewshot-preference-optimization). O trabalho original acompanha o artigo
[FSPO: Few-Shot Preference Optimization of Synthetic Data Elicits LLM Personalization to Real Users](https://www.arxiv.org/abs/2502.19312).


Este repositório contém código para treinar modelos personalizados usando FSPO (Few-Shot Preference Optimization), construído sobre a base do código de [Direct Preference Optimization (DPO) de Eric Mitchell]((https://github.com/eric-mitchell/direct-preference-optimization)). Como os prompts usados em FSPO tendem a ser longos, o treinamento utiliza Flash Attention para acelerar a execução.

## Setup do ambiente
Recomenda-se o uso de Python 3.12.
```bash
conda create --name FSPO python=3.12
source activate FSPO
pip install -r requirements.txt
```
Além disso, certifique-se de definir as seguintes variáveis de ambiente: HF_TOKEN e WANDB_API_KEY.

## Treinamento
Abaixo estão exemplos de comandos adaptados especificamente para o cenário Elix.

```bash
python -u train.py model=qwen3-4b datasets=[elix] n_epochs=1 loss=sft lr=1e-7 exp_name=elix_ipo trainer=FSDPTrainer sample_during_eval=false eval_every=10000  do_first_eval=false debug=true wandb.project=personalization batch_size=4 max_prompt_length=8192 max_length=8192 eval_batch_size=4

python -u train.py model=qwen3-4b datasets=[elix] n_epochs=1 loss=ipo lr=1e-6 loss.beta=0.01 exp_name=elix_ipo trainer=FSDPTrainer sample_during_eval=false eval_every=10000  do_first_eval=false debug=false wandb.project=personalization batch_size=2 max_prompt_length=8192 max_length=8192 eval_batch_size=2 model.archive=/PATH_TO_SFT_OUTPUT/LATEST/policy.pt
```
## Data
- [Elix - original](https://huggingface.co/datasets/Asap7772/elix_generations_gpt4omini_pref)


## BibTeX
```
@misc{singh2025fspofewshotpreferenceoptimization,
      title={FSPO: Few-Shot Preference Optimization of Synthetic Preference Data in LLMs Elicits Effective Personalization to Real Users}, 
      author={Anikait Singh and Sheryl Hsu and Kyle Hsu and Eric Mitchell and Stefano Ermon and Tatsunori Hashimoto and Archit Sharma and Chelsea Finn},
      year={2025},
      eprint={2502.19312},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.19312}, 
}
```
