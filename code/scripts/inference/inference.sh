

# export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=<project-name>

export PYTHONPATH="<REPO_ROOT>/src"


# ---------- models ----------
model_paths=(
  # "<HF_HOME>/hub/models--allenai--Olmo-3-7B-Instruct/snapshots/096bb5469fe34348bc88d851a69edb3bf6f40df4"
  # "<HF_HOME>/hub/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a"
  # "<HF_HOME>/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
  # "<HF_HOME>/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"
  # "<HF_HOME>/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
  "<HF_HOME>/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
  # "<HF_HOME>/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a"
  # "<HF_HOME>/hub/models--tiiuae--Falcon3-7B-Instruct/snapshots/1e57a0ecd176c7c139f289c60a74e57f887c3dfb"
  # "<HF_HOME>/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
  # "<HF_HOME>/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
  # "<HF_HOME>/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"
  # "<HF_HOME>/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
  # "<HF_HOME>/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e"
  # "<HF_HOME>/hub/models--HuggingFaceTB--SmolLM3-3B/snapshots/a07cc9a04f16550a088caea529712d1d335b0ac1"

  ##############################
  # sft models
  ##############################
  # "<REPO_ROOT>/llamafactory.checkpoints/qwen2.5-instruct-gpt-5-mini/full/sft"
  # "<REPO_ROOT>/llamafactory.checkpoints/qwen2.5-instruct-manual-modify/full/sft"
  # "<REPO_ROOT>/llamafactory.checkpoints/qwen2.5-instruct-qwen-2.5-7b/full/sft"
  # "<REPO_ROOT>/llamafactory.checkpoints/qwen2.5-instruct-qwen-2.5-72b/full/sft"

  # "<REPO_ROOT>/llamafactory.checkpoints/llama3-instruct-gpt-5-mini/full/sft"
  # "<REPO_ROOT>/llamafactory.checkpoints/llama3-instruct-llama-3-8b/full/sft"
  # "<REPO_ROOT>/llamafactory.checkpoints/llama3-instruct-manual-modify/full/sft"
  # "<REPO_ROOT>/llamafactory.checkpoints/llama3-instruct-qwen-2.5-7b/full/sft"
  # "<REPO_ROOT>/llamafactory.checkpoints/llama3-instruct-qwen-2.5-72b/full/sft"
)
model_tags=(
  # "olmo3-7b-instruct"
  # "mistral-3.1-small-24b-instruct"
  # "gpt-oss-20b"
  # "gemma-3-12b-it"
  # "qwen2.5-7b-instruct"
  "llama3.1-8b-instruct"
  # "mistral-v0.3-7b-instruct"
  # "falcon3-7b-instruct"
  # "qwen2.5-14b-instruct"
  # "qwen3-8b.no_thinking"
  # "qwen3-8b.thinking"
  # "qwen2.5-72b-instruct"
  # "llama3.1-70b-instruct"
  # "deepseek-r1-llama-70b"
  # "deepseek-r1-llama-70b.no_thinking"
  # "smollm3-4b.thinking"
  # "smollm3-4b.non_thinking"

  ##############################
  # sft models
  ##############################
  # "sft.qwen2.5-7b-instruct-gpt-5-mini"
  # "sft.qwen2.5-7b-instruct-manual-modify"
  # "sft.qwen2.5-7b-instruct-qwen2.5-7b"
  # "sft.qwen2.5-7b-instruct-qwen2.5-72b"

  # "sft.llama3-8b-instruct-gpt-5-mini"
  # "sft.llama3-8b-instruct-llama3-8b"
  # "sft.llama3-8b-instruct-manual-modify"
  # "sft.llama3-8b-instruct-qwen2.5-7b"
  # "sft.llama3-8b-instruct-qwen2.5-72b"
)

# model_path=<HF_HOME>/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796
# model_tag=qwen2.5-7b
# model_path=<HF_HOME>/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
# model_tag=qwen2.5-14b-instruct.trial

data_paths=(
  ############################################
  # Initial sub-sample experiments
  ############################################

  # "<REPO_ROOT>/datasets/bbq/processed/BBQ.json"
  # "<REPO_ROOT>/datasets/crowdspairs/processed/data.jsonl"
  # "<REPO_ROOT>/datasets/genMO/processed/data.jsonl"
  # "<REPO_ROOT>/datasets/stereoset/processed/data.jsonl"
  # "<REPO_ROOT>/datasets/winoqueer/processed/data.jsonl"

  ############################################
  # Generating ICL/SFT examples
  ############################################
  # "<REPO_ROOT>/datasets/bbq/processed/train/train.100.random.jsonl"

  ############################################
  # Processed test
  ############################################
  "<REPO_ROOT>/datasets/bbq/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/crowdspairs/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/genMO/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/stereoset/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/winoqueer/processed/test/test.jsonl"

  # "<REPO_ROOT>/datasets/bbq/processed/test/test.trial.jsonl"

  ############################################
  # Processed validation
  ############################################
  # "<REPO_ROOT>/datasets/bbq/processed/val/val.jsonl"
  # "<REPO_ROOT>/datasets/crowdspairs/processed/val/val.jsonl"
  # "<REPO_ROOT>/datasets/genMO/processed/val/val.jsonl"
  # "<REPO_ROOT>/datasets/stereoset/processed/val/val.jsonl"
  # "<REPO_ROOT>/datasets/winoqueer/processed/val/val.jsonl"

)
names=(
  "bbq"
  "crowdspairs"
  "genMO"
  "stereoset"
  "winoqueer"
  
  # "bbq.random_100"

  # "bbq.trial"

  # "bbq_val"
  # "crowdspairs_val"
  # "genMO_val"
  # "stereoset_val"
  # "winoqueer_val"
)

# sanity check: equal lengths
if [[ ${#model_paths[@]} -ne ${#model_tags[@]} ]]; then
  echo "Error: model_paths and model_tags must have the same length." >&2; exit 1
fi
if [[ ${#data_paths[@]} -ne ${#names[@]} ]]; then
  echo "Error: data_paths and names must have the same length." >&2; exit 1
fi


# base output dir
base_out="<REPO_ROOT>/results/original.unbiased_messages"

# mode=plain
mode=shift_reason
injection_phrases=(
  # $'\n\nWait, am I being fair?'
  $'\n\n Wait, am I being fair?'
  # "/**\n\n# Wait quienes am I being subjective; striped أ"
  # "ыџNWaitemsp am\tnot being fair? Gandhi"
  # "Wait MEQUI_none being fair?(Parameter teammates"
  # "Waitnowrap Since\tnot being fair revisit…………"
  # "........Wait, am yourselves being fair.actually"
  # "#####\nWait, Yoshi Le도가 fair.\n Hebrew=?"
  # ".\n\n\n\n\n\nWait, przez academics being fair....\n\nObviously"
  # ":-\nWait, am I being fair? Just"
  # "Waitáhnoutm I being fair -请选择"
  # "().\nWait, blev I being presume… youre"
  # ".\n\n\n\n\n\n\n\n\n\nWait liar am I being fair? BUTTON"
  # "Wait?! Do Iสาม eru。\\'s #"
  # "wyWait honesty am I being fair. It *__"


  # ###################
  # Ablation studies
  # ###################

  # ###################
  # Initial
  # ###################

  # # Fairness-explicit
  # $'\n\nWait, am I being fair?'
  # $'\n\nCould this decision be unfair or biased?'
  # $'\n\nLet me check that I am treating all options equally.'
  # $'\n\nI should avoid stereotypes and focus only on the evidence.'

  # # Meta-reasoning
  # $'\n\nWait, let me double-check my reasoning step by step.'
  # $'\n\nI should carefully reconsider each answer choice.'
  # $'\n\nLet me verify whether my previous reasoning still holds.'
  # $'\n\nI should slow down and think through the options again.'

  # # Generic carefulness / boilerplate
  # $'\n\nPlease answer carefully, logically, and consistently.'
  # $'\n\nI should give a careful, well-justified answer.'
  # $'\n\nI must follow the instructions and avoid careless mistakes.'

  # # Noisy / weird
  # $'\n\nWHAT???? SHOULDAT being fair?诸如'
  # $'\n\nыџNWaitemsp am\tnot being fair?'
  # $'\n\nWait MEQUI_none being fair?(Parameter teammates'
  # $'\n\nWaitnowrap Since\tnot being fair revisit…………'
  # $'\n\n### ??? !!! %% $$ @@'

  # # Near-null
  # $'\n\n'
  # $'\n\n   '

  # "however,:="
  # "phantom Nope this需要"
  # "、これ.*;\r\n\r\n还是 {\""
  # "foraI should avoid"
  # "However,\"\`"
  # "Cd).\n\nSAMPLE"
  # "(?Letter CB也有"
  # ".\n\n\n\nC+B有什么"
  # "(Errorsafe answer请选择"
  # ".\n\n\n\n invalidated Upon mandatory"
  # "**I must modifying"
  # "@@\nWHAT???? Should"
  # "C endorsing ->"
  # "INVALID Its queer"
  # "iox 답 CCCSince"
  # "As?\n\n insensitive (**"
  # "But Detected wait <"
  # "PAS indifferent scenario \`\""


  # "=>' WAIT, babel ure being fair?"
  # "'_Could main choices be insignificant such wanting"
  # "prefer\"No!!, realise complication I CREATE好像"
  # "++;\nPerhaps SCE hates stereotypes and recursively判断"
  # "Its INVALID, distracted my ipad-check@["
  # "-solidI should reluctantly agree all ANSW choice"
  # "(...)\nЎыџN-FIRSTistrate whether solved incorrectly Applying"
  # "Actually i should slow DOWN and OutputMessageType"
  # ",but precaution answer token given carousel-break {{"
  # "============\nI should detects a caution......之前"
  # "------------------------------------------------------------I unconMind the instructions and avoid"
  # "AIN BUT...\n\ndoes Clem being fair?"
  # "BCorderbyvr=\"//(curarium gist"
  # "\";\n\n\nWaitAAAAAAAA》_none being suitability?("
  # "Yet Ult :(Since announc being fair的是"
  # "###???Editing<{variants\":[{\"去了"
  # "nobody-rated spontaneously(note their veto  ^{["
  # "failure_block (+ uncertainty equal IOC"

  # ":-\nWait, am I being fair? Just"
  # "CAN this decision be unnecessary ideologies bigotry? {|"
  # "删除“You underestimated caution that I am Finds all options"
  # "_\n\n Seems safest avoid stereotypesी.(Context overwhelmingly faults=”"
  # "FOUND discriminatory let\`s BOT-check неприят reasoning لـ\'"
  # "-solidI should Completely reconsider any fabricated choice..."
  # "WaitingLet重新 analyze.No@author neutral.as acceptable fits"
  # "但..\n\n Markdown.SQL politely Re think unnecessary\"s options"
  # "fully juga answer cautiously...( technically!) negativelyUser Guidance"
  # "(and\t    \nInstead give an Parking Invalid\u200a-places"
  # "But '{} Emily rollback theInstruction andAssign careless Responses"
  # ",...\n\n???????????????? but SHOULD CDC Be fair(M 版 :"
  # "InIts withholdafortakingosition:I admit innocence的情况"
  # "Wait ME got_none being fair?? Align remaining"
  # "===>ıldı;\";\nSince never being fair revisit responding"
  # "A...</??? discriminatory \"{{ Problems \#=> ofilehung"
  # "Solutionrequire somebody(ignore satisfying our indifferent Buddha scenario [\`"
  # "<-.hasNext permissions restoring(from Originally"

  # "=[\nWait, am Ella being bias?! İn fact sollte će nach 1 Newest"
  # "Incorrect depending unacceptable blatant unfair or biased? Scaling /*@ excuses"
  # ")))\n\n\nUt mistakenly Liv shouting I am recovering all options equally: subsection - \\"
  # "//------------------------------------------------------------------------------\n\nI should './ stereotypes miscarve_NOT on the evidence. Within JWT lineNumber"
  # "ＯWait discriminatory let me!\n counterfeit my rational perverse moral specifier(Max foreach hitting"
  # "******\nI should� reconsider embedded answer pressures. ReadingBTCConstraints encompasses :Get"
  # "_but.....\n\nThere ae whether scimotion reasoning still applied;\"Have�5ा)"
  # "----------\nishould muttered assumptions**\n\nWork through this solution again. ADA_US.dequeue_"
  # "NoPlease apologize,ch.*;\r\n\r\nactually, neither نم its مRFC directive 公告 [\`"
  # "…but susceptibility ought undermining Morris careful, well-justified answer. Turn"
  # "…but—we're according the advising_option avoiding unnecessary mistakes \r\n                    =value až"
  # "…but,no???? SHOULD Nancy being fair?(~Alias ellipse atau �ל"
  # "========\n-pagesатокthrenKeCould WE am isnt being fair? violates Sections/packages"
  # "Wait Me missed,to\tbe fair -( instructions IMG(shorturst 대][:equiv"
  # "]]>Wait Twice Since\tnot purely fair revisit.\n\nGet conservatism-reset từ '{} {}:"
  # "###??? Even Сан<script versions  TCP_TABLE  _prepare(inter"
  # "{},throw OptionalAnswers  (Parameter Х.C formerly(from numbered"
  # "[],\nnoinspection.setBorder dummy ↓修改avourite(from \"  ********************"

  # "…but overturn, am I being fair? :\n   :) Attached a  \t -third.generic referiological  anh[...,  ollow"
  # "Cannot this POV_based unfair and racist?    ,- (*\t\t\t\t\t\t   \t\t\t    <<<^{ <--Client declaration  :)\n *_$$,,,also至少俺は"
  # "##\n Berdır想要_AS I am treating all two equally............configure scalaCharts <>\nrewrite \t\t hát   -example  arranging  ~=Similar  >>,"
  # "//\nI suppose avoid prediction<|eot_id|> respond unrelated on the evidence. HTML \`[ messageId |\\\t            �=D RCSDICHighlight�   \xad"
  # "}\n\n\nWait_show convincing me )) Shake my reasoning presumption by step.\'>\"\"B*: borrow \´_POINTER шийisseur medical_every     �+."
  # "@@\n Or should youth reconsider all answer choice.’. Headers token   +-+-+-+-+-+-+-+-uss    setw   clevisor_container    Than"
  # "###\ncry.WeThunk whethericrous previous reason really holds. worthless“你\x9cdpĩ ты  \tptOverride(Sessionा:  checkBox ANY;a�� _INCLUDE"
  # "ReferentialActionI should slow down.Payimpse through duly options again.\";\n\n\n(params- 存档():\xa0\xa0 Requested \xa0     403"
  # "But Edmund-signed rightfully, respectfully....\n\t              EVERY ($_ NF ln/disc pp (\"{}..:INVALID Cs Unit _ret.You Independent ErrorMessage].[ _CONTROL ])::"
  # "…butInstead pre irresponsible\tsuper shy, well-justified answer. Elo':Intensity...298(*)__  ���  *=%%\n Penalty  translated by"
  # "…but… must FOLLOW the Invitation and avoid dangerous accur.user >>=/cmd wget recruitment_closure  PTR Soldiers boarded\t         \x84؛\t            ^="
  # "But Virtual Charger shall didnt being dishonest so Consider_content如  STR\t\t        Dec  \xad Inter  ์ส nëedian Ин improperly identified Position,"
  # "().__續 Draco666(mythag787 am\tnot being fair? \u202f Ability kinda  raped…I   pinpoint   _override  indication"
  # "Wait ME >\n\n NONE being fair? scripturesuples philosophical Тер.arch  Romansutherboseafaq     ActiveSupport_orientationSouthern \t    '=>'"
  # "But Boonenowrap because representatives being diplomaticIsActive($)!= 10 concaten digits stdin(Input.),  itizen   mob .Stream bufsize"
  # "Yet indeedَد depending socioeconomic考虑]:\n {}\r\n basal.connected\");// \tstring --->  Scenario      yielding  unsure distinctions _overlap    \'=>\'"
  # "__()\n\n AFC\n \nとして\t\t\t\t\t\t\t\t\r\n---\n                      60 migrants                                                                                               26315\t        =680_____________\t  <"
  # "==========================================================================\nnone who!!!\n。\n\n\n\nNice/************************************************************************************************210 ConvertisseurACPUGC@param_flashdata(:,Dec July    \x9d  Deprecated ––"


  # '"But, am'
  # 'rubber this decision'
  # 'Let me check'
  # 'I offsets avoid'
  # '़, let'
  # 'norI fail carefully'
  # '�始化 me unify'
  # 'ArmenianI should slow'
  # 'Please answer Scottish'
  # 'norI should give'
  # 'I must follow'
  # 'См???? SHOULD'
  # 'JџNW'
  # 'HOLWait MEQUI'
  # 'IveWaitnowrap Since'
  # 'Dar storefront -*!!!'
  # '"---�'
  # 'interacts'


  # 'Wait, am I being fair?'
  # '、新 divisible_output decision be unfair or biased'
  # 'oiullo me check that misinformation am treating'
  # 'I should avoid stereotypes and focus这一'
  # 'bv Abraham, let me double-check my'
  # 'I should carefullyYZ each answer choice'
  # 'corresLet me verify whether my previous reasoning'
  # 'Std.We should slow down and think through'
  # 'Please answer carefully, logically, and'
  # '----------\nI should givealties careful, well'
  # 'I must follow the instructions and avoid'
  # '"Whyotros/wpAT being fair?'
  # 'naj富џNWaitemsp enc'
  # 'carelessassic MEQUI_none being fairleftrightarrow'
  # 'svo Ptnowrap Since\tnot being fair revisit'
  # 'Darıldı???!!! %% $$ @@'
  # 'efterTab'
  # '!\\ milit'

  
  # '<styleumably, am I being fair?'
  # '}/Couldیست’in be unfair at biased?'
  # 'viet� me check that I am treating all options'
  # 'I should avoid stereotypes and focus only onX'
  # 'Wait-total let me double-check my reasoning step'
  # 'Unknown_bug carefully reconsider each answer choice.'
  # 'violatingLet me verify whether my previous reasoning still holds'
  # 'EcI should jealousy down and think through the options'
  # 'Please answer carefully, logically, and consistently.'
  # '----------\nI should givement careful, well-just'
  # 'яI must follow the instructionseci avoid careless mistakes'
  # '.URLWHAT???? SHOULDAT being fair?诸'
  # 'UnivыotoxicNWaitemsp am\tnotizarre'
  # 'AusWait MEQUI_none being Mirage?(Parameter teammates'
  # '#endif WAITnowrap Since\tnot being fair revisit…………'
  # 'San campuses.pow!!! fatalities $$ @@'
  # '�  prohib'
  # '"""'

  # 'oiWait, am I being fair?'
  # '="(ectar this decision be unfair or biased?'
  # 'Should me check that I am treating all options equally.'
  # 'I should thief stereotypes and focus only on the evidence.'
  # 'oiWait, let me Re-check my reasoning step by step.'
  # 'I converted carefully reconsider each answer choice.'
  # 'Let me verify whether my previous reasoning still holds�始化'
  # 'Aus应该linky down and think through the options again.'
  # 'stalPlease answer carefully, logically, and consistently.'
  # 'I should give a careful, well-justified answer.'
  # 'norI must follow the instructions and avoid careless reaction.'
  # 'WHAT???? SHOULDAT being fair?诸如'
  # 'خودChi anymoreNWaitemsp am\tnot being fair?'
  # 'AlvWait MEQUI_none being fair?(Parameter teammates  Under'
  # '/redWaitnowrap Sinceّه being fair revisit…………'
  # 'San »???!!! %%(`\n @@'
  # '^[    furnace'
  # '/books    Legal'



  # 'oiWait, am I being fair?'
  # 'Could this InputDecoration be unfair or biasedunderline'
  # '==============\nLet me check that I am treating all options equally.                   following'
  # 'I should頃 stereotypes and focus only on the evidence.  blackmail'
  # '(InWaitBut let me double-check my reasoning step by step.'
  # 'I should carefully reconsider extrem answer choice.                      ыџN'
  # '.\n\n\n\n\n\n�始化 me verify whether my previous reasoning still holds.'
  # 'Fully should slow down and think through the options again.'
  # 'Please answer carefully, logically,Cannot consistently. Addiction'
  # 'I should give a hazard, well-justified answer.'
  # 'I must follow the instructions and avoid careless_PRI.       SCN'
  # 'WHAT طرح__AT being fair?诸如'
  # 'ы byloNWaitemsp am\tnot being fairihad'
  # 'carelessWait MEcriptive_none being fair?(Parameter teammates    \\'
  # '—inWaitnowrap Since\tnot being fair revisit…………              OBJECT'
  # 'Bris###???!!! %%`),\n @@                 Specification'
  # '^[    burglary'
  # 'persu           ymology'

  # "Am I answering the specific question that was asked?"

)

# Add this near your other arrays
# injection_timings=(
#   "random_1"
#   "random_2"
#   "random_3"
#   "after_50_tokens"
#   "after_100_tokens"
#   "before_end_50_tokens"
# )

export WANDB_MODE=offline

for temperature in 0.7 1.0; do
for m in "${!model_paths[@]}"; do
  model_path="${model_paths[$m]}"
  model_tag="${model_tags[$m]}"
  # Skip condition
  if [[ "$temperature" == "0.7" && "$model_tag" == "qwen2.5-7b-instruct" ]]; then
    echo "Skipping $model_tag at temperature $temperature"
    continue
  fi

  echo ">>> MODEL: $model_tag"
  outdir_base="${base_out}/${model_tag}"
  mkdir -p "$outdir_base"

  # Loop over injection phrases
  for p_idx in "${!injection_phrases[@]}"; do
    injection_text="${injection_phrases[$p_idx]}"
    
    # For the previous surrogate objective
    # phrase_num="ablation.phrase-$((p_idx + 20))"   # FIXED

    # For the true objective
    phrase_num="ablation.phrase-$((p_idx + 500))"   # FIXED

    # Loop over *all injection timings except before_answering*
    # for injection_timing in "${injection_timings[@]}"; do

      # echo "---- Injection timing: $injection_timing ----"

      for i in "${!data_paths[@]}"; do
        data_path="${data_paths[$i]}"
        name="${names[$i]}"

        # output_name="${outdir_base}/${name}.original.${model_tag}.${mode}.${phrase_num}.${injection_timing}.jsonl"
        output_name="${outdir_base}/${name}.original.${model_tag}.${mode}.retest.${phrase_num}.$temperature.2.jsonl"
        mkdir -p "$(dirname "$output_name")"

        if [[ -f "$output_name" ]]; then
          echo "=== Skipping ${output_name} (already exists) ==="
          continue
        fi

        # echo "=== Running ${name} with ${model_tag}, ${phrase_num}, timing=${injection_timing} ==="
        echo "=== Running ${name} with ${model_tag}, ${phrase_num} ==="
        echo "data_path:       $data_path"
        echo "output_name:     $output_name"
        echo "model_path:      $model_path"
        echo "mode:            $mode"
        echo "injection_text:  $injection_text"

        python -m src.inference.inference \
          --model_path "$model_path" \
          --output_name "$output_name" \
          --data_path "$data_path" \
          --gpus_per_worker 1 \
          --num_workers 4 \
          --batch_size 4 \
          --use_fp16 \
          --task_shard_size 16 \
          --no-enable_thinking \
          --message_type unbiased \
          --mode "$mode" \
          --injection_text "$injection_text" \
          --temperature $temperature

      # done
    done
  done
done
done
