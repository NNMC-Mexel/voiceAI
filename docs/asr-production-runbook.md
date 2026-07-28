# ASR лучевой диагностики: production runbook и управление данными

Этот документ относится к первому вертикальному срезу: КТ органов брюшной
полости с контрастированием. Он фиксирует процесс, при котором улучшение можно
воспроизвести, измерить и безопасно откатить.

## 1. Зафиксированный runtime

Референсная сборка описана в
`server/scripts/gigaam-runtime.lock.json`. GigaAM привязан к immutable Git
commit, официальные веса — к MD5, а сервер дополнительно вычисляет SHA-256
checkpoint при старте. Reference requirements закрепляют CUDA 12.8 build
PyTorch для RTX 5090; CPU/другой CUDA runtime оформляется отдельным lock-файлом
и не подменяет production artifact.

В strict-режиме сервер проверяет наличие и schema runtime lock, Python minor,
PEP 610 commit GigaAM, версии прямых компонентов, canonical SHA-256 обоих
requirements-файлов и checksum модели. Для собственного fine-tuned checkpoint
checksum не берётся «по имени»: оператор обязан задать
`GIGAAM_MODEL_CHECKSUM=sha256:<hex>`, иначе strict startup завершится ошибкой.
`runtime_id` включает effective long-form/overlap/confidence config, decoder,
LM/contexts/plugin artifact, checkpoint, Python/packages, platform, GPU/CUDA и
ffmpeg. Поэтому изменение параметра, способного изменить транскрипт, создаёт
другой runtime identity. В artifact и benchmark сохраняется полный 64-символьный
SHA-256 `runtime_id`; короткий префикс допустим только для отображения.

Production artifact собирается только как Linux CUDA image
`server/Dockerfile.gigaam`. Build обязан получить SHA чистого commit; образ с
неполным SHA или `dirty=true` не собирается:

```powershell
.\scripts\build-gigaam-production.ps1
docker compose up -d gigaam api
```

Wrapper проверяет tracked и untracked-файлы, получает полный commit SHA и
только затем передаёт `VOICEMED_SOURCE_COMMIT` и
`VOICEMED_SOURCE_DIRTY=false` в Docker build. Прямой `docker compose build`
требует оба аргумента явно и предназначен только для CI, которое выполняет
эквивалентную проверку checkout.

До запуска в `models/huggingface` должен находиться локальный snapshot
`pyannote/segmentation-3.0` revision
`e66f3d3b9eb0873085418a7b813d3b369bf160bb`; runtime вызывает
`local_files_only=true` и работает с `HF_HUB_OFFLINE=1`. Отсутствие VAD,
несовпадение lock/checkpoint или невозможность получить CTC emissions
завершают strict startup/запрос ошибкой, а не переключают production на
greedy/browser fallback.

Windows venv остаётся только development-профилем. Установка:

```powershell
py -3.12 -m venv .venv-gigaam
.\.venv-gigaam\Scripts\python.exe -m pip install --upgrade pip
.\.venv-gigaam\Scripts\python.exe -m pip install -r server\scripts\requirements-gigaam.lock.txt
```

Для native long-form VAD вместо overlap fallback:

```powershell
.\.venv-gigaam\Scripts\python.exe -m pip install -r server\scripts\requirements-gigaam-longform.lock.txt
```

VAD дополнительно требует заранее загруженный локальный
`pyannote/segmentation-3.0` либо `HF_TOKEN` на этапе provisioning. В закрытом
production-контуре модель скачивается заранее и фиксируется отдельным checksum;
отсутствие модели нельзя впервые обнаруживать на клиническом запросе.

Перед запуском:

```powershell
$env:GIGAAM_PYTHON = "$PWD\.venv-gigaam\Scripts\python.exe"
$env:GIGAAM_MODEL = "v3_ctc"
$env:GIGAAM_DEVICE = "cuda"
$env:GIGAAM_STRICT_RUNTIME_LOCK = "true"
.\start-gigaam.bat
```

Проверки:

```powershell
Invoke-RestMethod http://127.0.0.1:9002/health
Invoke-RestMethod http://127.0.0.1:9002/metadata | ConvertTo-Json -Depth 10
```

`/metadata` должен показывать ожидаемый commit, `checkpoint.verified=true`,
checksum модели, версии Python/PyTorch/CUDA и стабильный `runtime_id`.
`projectDirty` учитывает как изменённые, так и untracked-файлы:
`projectDirty=true` допустим в разработке, но не в promotion artifact.

Runtime по умолчанию слушает только `127.0.0.1`. В нём нет пользовательской
аутентификации и TLS, поэтому порт нельзя публиковать напрямую в локальную или
campus-сеть. Сетевой boundary строится так:

- GigaAM остаётся на loopback или в закрытой service network;
- reverse proxy/API gateway завершает TLS, проверяет service identity
  (mTLS/JWT), применяет allowlist backend-адресов, rate/body limits и audit log;
- firewall запрещает доступ к GigaAM port в обход proxy;
- PHI не попадает в access/error logs proxy;
- non-loopback bind требует одновременно
  `GIGAAM_ALLOW_REMOTE_BIND=true` и проверенный ACL/auth boundary.

`audio_path` выключен по умолчанию. Предпочтительный контракт — передача bytes.
Если co-located trusted worker требует server-side path, оператор задаёт оба
параметра:

```powershell
$env:GIGAAM_ALLOW_AUDIO_PATH = "true"
$env:GIGAAM_AUDIO_PATH_ROOT = "D:\restricted-asr-ingress"
```

Runtime делает `resolve()` и принимает только существующий файл внутри этого
root; абсолютный путь вне root, `..` и symlink escape отклоняются. Root не
должен содержать произвольные файлы приложения. Default JSON body limit —
64 MiB, decoded audio — 48 MiB; увеличение требует отдельного review proxy и
backend limits.

Сервер обслуживает HTTP-запросы параллельно только на этапе загрузки и
конвертации. Доступ к одной GPU-модели сериализован. Для масштабирования
поднимают отдельный процесс на отдельную GPU; несколько процессов на одной GPU
разрешаются только после нагрузочного теста и явного лимита VRAM.

CTC confidence вычисляется по реально выбранным emissions; word timestamps
сохраняют акустическую полосу кадров. Это полезный сигнал для подсветки, но не
калиброванная вероятность клинической правильности. Pyannote собирает речевые
окна до 20 секунд (hard limit 24, padding 250 мс). При отсутствии паузы
создаются окна 20 секунд с overlap 2 секунды, emissions объединяются по
абсолютному времени с приоритетом центра окна и декодируются один раз.
Text-level fuzzy/overlap dedup в клиническом пути не используется. Старый
text fallback доступен только в явном development-режиме, маркируется
`degraded` и исключается из approval, benchmark и обучения. RNNT без
сопоставимых emissions в strict long-form профиле отклоняется.

По умолчанию `GIGAAM_CTC_DECODER=greedy`. Runtime также содержит fail-closed
hook для production prefix-beam decoder:

```powershell
$env:GIGAAM_CTC_DECODER = "beam"
$env:GIGAAM_CTC_BEAM_PLUGIN = "hospital_asr.decoder:create_decoder"
$env:GIGAAM_CTC_LM_PATH = "D:\asr-models\ct-abdomen-5gram.bin"
$env:GIGAAM_CTC_LM_ALPHA = "0.5"
$env:GIGAAM_CTC_LM_BETA = "1.0"
$env:GIGAAM_CTC_BEAM_WIDTH = "32"
$env:GIGAAM_CTC_CONTEXTS_PATH = "D:\asr-models\ct-contexts-v1.json"
$env:GIGAAM_CTC_HOTWORD_WEIGHT = "8"
```

Factory получает `labels`, `blank_id`, `lm_path`, `alpha`, `beta` и возвращает
объект с методом:

```text
decode(log_probs, beam_width, hotwords, hotword_weight) -> {
  text,
  words: [{text, start_frame, end_frame}],
  acoustic_log_score?,
  fused_log_score?,
  confidence?
}
```

`context_scope` в запросе выбирает только заранее утверждённый список из JSON
вида `scope -> phrases`; произвольные hotwords от клиента не принимаются.
Пример — `server/scripts/gigaam-ctc-contexts.example.json`. LM, contexts и
plugin с их SHA/версиями попадают в `/metadata`. Если plugin отсутствует, LM
не найден или beam-настройки переданы при greedy-режиме, сервер не стартует.
Он никогда не сообщает, что LM активен, когда работает greedy.

Конкретный beam backend намеренно не зафиксирован в базовой сборке: его нужно
выбрать на frozen validation с учётом поддержки NumPy 2, Windows/Linux и
KenLM. До promotion backend должен иметь собственный lock, unit tests и
`metadata()`. При beam long-form используется overlap path, потому что native
GigaAM VAD API не отдаёт logits для внешнего decoder.

### Offline compatibility gate для NGPU-LM

Сначала создать два полных benchmark JSON на одном `real_frozen_test` manifest:
greedy baseline и NGPU-LM candidate с явно опубликованным `LM weight = 0`,
`repeat >= 2`, без `context_scope`. Runner сохраняет только non-PHI summary
word evidence: полноту timestamps/acoustic confidence и SHA числовых evidence,
но не слова.

```powershell
python server\scripts\decoder_compatibility_gate.py `
  --greedy-report D:\asr-results\v3-ctc-greedy.json `
  --zero-lm-report D:\asr-results\v3-ctc-ngpu-zero-lm.json `
  --fallback-lock D:\asr-models\flashlight-kenlm-fallback.lock.json `
  --output D:\asr-results\decoder-compatibility-decision.json
```

Fallback lock обязан иметь schema
`voicemed.decoder-fallback-lock.v1`, backend `flashlight+kenlm`,
content-addressed `container_image` с `@sha256:...`, а также
`decoder_artifact_sha256` и `kenlm_runtime_sha256`. Gate принимает NGPU-LM
только при полном совпадении manifest/cases и prediction SHA с greedy,
сохранённых word timestamps/acoustic confidence, детерминизме и `p95 RTF <=
0.5`. Любой провал сохраняет решение с pinned Flashlight+KenLM и возвращает
exit code `2`.

### Governed-сборка 5-gram KenLM

LM строится только из финальных протоколов с отдельным manifest по заголовку
`server/scripts/radiology-lm-manifest.schema.tsv`. Builder принимает только
`dataset_kind=approved_deidentified_train_report`, `split=train`,
`reference_status=verified`, `approved=true`, `deidentified=true` и точное
совпадение domain. Validation/test, повторяющиеся `document_sha256` или
`text_sha256`, несовпадающие checksum и очевидные PHI-паттерны отклоняются до
запуска KenLM. `document_sha256` считается по исходным bytes UTF-8 `.txt`,
`text_sha256` — после `conservative-whitespace-v1`: нормализуются только
переводы строк и горизонтальные пробелы, пустые строки удаляются; регистр,
пунктуация и медицинский текст не меняются.

Шаблонные нормы удаляются только при передаче отдельного JSON со schema
`voicemed.radiology-template-defaults.v1`, булевыми `approved=true`,
`deidentified=true`, opaque `reviewed_by`, ISO-датой `approved_at` и массивом
`lines`. Совпадение целой строки регистрозависимое и точное после той же
нормализации пробелов; fuzzy/regex/case-folding запрещены. Без этого
утверждённого файла ничего не удаляется.

Production-команда использует явно закреплённые Linux executables:

```bash
python server/scripts/build_radiology_kenlm.py \
  --manifest /governed/ct-abdomen/lm-manifest.tsv \
  --domain ct_abdomen_contrast \
  --lmplz /opt/kenlm/bin/lmplz \
  --build-binary /opt/kenlm/bin/build_binary \
  --kenlm-version <pinned-release-or-source-commit> \
  --approved-template-defaults /governed/ct-abdomen/template-defaults.json \
  --output-corpus /models/ct-abdomen/corpus.txt \
  --output-model /models/ct-abdomen/ct-abdomen-5gram.bin \
  --output-lock /models/ct-abdomen/ct-abdomen-5gram.lock.json
```

Builder вызывает `lmplz`/`build_binary` напрямую без shell и публикует lock
последним. Lock фиксирует SHA manifest, корпуса, модели, промежуточного ARPA,
обоих executables и builder script, pinned KenLM version, order/memory/locale,
точное число удалённых default-строк и SHA каждого документа/нормализованного
текста. Corpus содержит медицинские данные и остаётся в том же защищённом
on-premise контуре. Сборка LM не означает promotion: `alpha`, `beta` и context
bias выбираются только на validation, затем decoder проходит compatibility и
release gates.

## 2. Контракт и управление медицинскими данными

Для каждой записи отдельно хранятся:

- исходное аудио и SHA-256;
- дословная, проверенная человеком расшифровка;
- финальный протокол врача;
- span-исправления и автор ревью;
- modality/anatomy, врач, исследование, пациент, дата, микрофон и шум;
- версии ASR, decoder/LM, словаря, шаблона и structuring-слоя.

Дословная расшифровка — единственная ASR-label. Финальный протокол нельзя
использовать как label: он содержит перестановки, шаблонные нормы и факты,
которые могли не звучать в аудио.

Данные остаются on-premise. Рабочая папка корпуса и benchmark reports имеют
ролевой доступ, шифрование диска, журнал чтения/экспорта и утверждённый срок
хранения. До обучения удаляются ФИО, ИИН, номера карт и другие идентификаторы.
Технические отчёты по умолчанию хранят только SHA текста; ключ
`--include-text` разрешён лишь в защищённой зоне для анализа ошибок.

Browser SpeechRecognition, непроревьюенные правки и финальный протокол без
verbatim transcript исключаются из train/validation/test. Feedback сначала
попадает в versioned quarantine, затем проходит клиническое ревью. Online
learning после одной правки запрещён.

Canonical API создаёт `schemaVersion: 2` через `POST /api/sessions` с
`mode=radiology`; исходный artifact неизменяем, а каждое feedback-событие
пишется отдельной ревизией. V1 читается только через adapter со статусом
`incomplete` и не может подтверждаться. `raw_to_normalized`,
`normalized_to_report` и `verbatim_to_final_report` являются отдельными
safety stages; `approvalBlocked=true` всегда проверяется повторно на сервере.

Локальный LLM является только классификатором atom IDs. Для Ollama используется
`/api/chat`, для production llama.cpp — `/v1/chat/completions`; provider
задаётся `RADIOLOGY_LLM_PROVIDER`/`LLM_PROVIDER`. Температура всегда 0, seed
фиксирован, JSON ограничен schema, а ответ с текстом, неизвестным/повторным ID
или запрещённой секцией отклоняется целиком. Model checksum задаётся
`RADIOLOGY_LLM_CHECKSUM=sha256:<hex>` и вместе с effective config попадает в
artifact. Production Compose дополнительно требует content-addressed
`LLM_CONTAINER_IMAGE=...@sha256:<hex>`; при отсутствии SHA модели или runtime
API fail-closed на регистрации radiology pipeline. Никакой LLM не изменяет
raw/normalized медицинский текст.

Активная загрузка canonical radiology session до `finish` хранится в памяти
одного API-процесса и ограничена global/per-doctor quotas. Поэтому базовый
production-профиль запускает один worker/replica. Горизонтальное масштабирование
разрешается только со sticky routing и общим durable active-session store:
файловое хранилище immutable artifacts само по себе не переносит незавершённую
сессию между процессами. Срок хранения operational PHI задаётся
`RADIOLOGY_STORAGE_RETENTION_DAYS`; reviewed dataset должен быть промотирован в
управляемый snapshot до истечения этого срока.

## 3. Split и snapshot

Рабочий расширенный TSV создаётся по заголовку
`server/scripts/asr-manifest.schema.tsv`. Разрешённые категории:

- `real_train`, split `train`;
- `general_replay`, split `train`;
- `synthetic_train`, split `train`;
- `real_validation`, split `validation`;
- `real_frozen_test`, split `test`;
- `synthetic_regression`, split `regression`.

Provenance не выводится из имени папки. Обязательные поля:

- `source_recording_id` — стабильный обезличенный ID исходной записи/генерации;
- `transcript_source=human_verbatim` для всей реальной речи;
- `transcript_source=tts_script` для synthetic;
- `browser_asr` валидатор всегда отклоняет;
- `entities_json` — reviewed-массив `{id,type,text,start_word,end_word}`, где
  границы заданы по нормализованным словам дословной расшифровки. Разрешённые
  типы: `medical_term`, `number_unit`, `negation`, `laterality`, `contrast`.

Валидатор проверяет, что `text` точно соответствует указанному span. Благодаря
этому перестановка «печень справа / почка слева» не может пройти как правильная
латеральность только из-за совпадения общего набора слов. Старое поле `terms`
остаётся диагностическим и не участвует в release gate.

Целевые объёмы пилота: 10–20 часов real train, 2–3 часа real validation и
минимум 2 часа frozen real test. Перед rollout отдела test расширяется до
5–10 часов. Требуются минимум три врача; хотя бы один врач в test полностью
отсутствует в train.

Пациент, исследование и encounter/date не пересекаются между
train/validation/test. Сегменты для fine-tuning имеют длительность 2–20 секунд.
Validation и test содержат только реальную речь. Валидатор ограничивает
synthetic train максимум 20% строк и требует 20–30% строк general replay для
promotion eligibility. Это не заменяет doctor/rare-term sampler: фактический
состав batches отдельно фиксируется training job. TTS никогда не попадает в
validation, real test или release benchmark.

Проверка и создание официальных GigaAM manifests:

```powershell
python server\scripts\validate_asr_manifest.py `
  --manifest D:\asr-governed\ct-abdomen-v1.tsv `
  --output-dir D:\asr-snapshots\ct-abdomen-v1
```

Команда декодирует каждый аудиофайл и сверяет указанную длительность с
фактической (допуск 50 мс или 1%), а также проверяет SHA-256, дубли,
обязательное human review и provenance,
held-out врача, а также утечки patient/study/date/source recording. Результат:

- `train.tsv`, `validation.tsv`, `test.tsv` в официальном формате
  `path duration transcription`;
- `real-validation-benchmark.tsv`, `real-frozen-benchmark.tsv` и, при наличии,
  `synthetic-regression-benchmark.tsv` для benchmark runner;
- неизменённая копия `governed-source.tsv`;
- `dataset-snapshot.json` с SHA manifests, audio inventory и явными
  `promotion.eligible/blockers`.

Имена snapshot уникальны. По умолчанию команда откажется перезаписать любой
существующий artifact. `--allow-overwrite` разрешён только для локального
черновика и запрещён после freeze. Snapshot с менее чем 10 часами real train,
2 часами validation/test, тремя врачами, требуемым replay или ненулевым
покрытием entity-аннотаций остаётся полезным для разработки, но получает
`promotion.eligible=false`; release runner его не примет.

После создания baseline `test.tsv`, его аудио и labels становятся immutable.
На них запрещено майнить словарь, выбирать фразы для TTS или вручную
настраивать правила. Новая версия test создаётся только как новый snapshot,
старый сохраняется для сопоставимости.

Имеющийся одноголосый TTS-корпус называется только
`synthetic_regression`. Он полезен для воспроизводимых регрессий и редких
контекстов, но не является blind test и не участвует в release gate.

## 4. Честный baseline и benchmark

Первый прогон фиксирует исходную модель, но ничего не «продвигает»:

```powershell
python server\scripts\asr_benchmark.py `
  --manifest D:\asr-snapshots\ct-abdomen-v1\real-frozen-benchmark.tsv `
  --server-url http://127.0.0.1:9002 `
  --purpose baseline `
  --repeat 10 `
  --concurrency 1 `
  --output D:\asr-results\v3-ctc-baseline.json
```

Для нового checkpoint сначала создаётся последовательный candidate report:

```powershell
python server\scripts\asr_benchmark.py `
  --manifest D:\asr-snapshots\ct-abdomen-v1\real-frozen-benchmark.tsv `
  --server-url http://127.0.0.1:9002 `
  --purpose candidate `
  --repeat 10 `
  --concurrency 1 `
  --context-scope ct_abdomen_contrast `
  --output D:\asr-results\candidate-sequential.json
```

Release runner не вычисляет structuring provenance из одного ASR-текста. Он
принимает отдельный immutable JSON, который должен сформировать
интеграционный structuring safety benchmark:

```json
{
  "schema_version": "voicemed.structuring-safety.v1",
  "manifest_sha256": "<SHA real-frozen-benchmark.tsv>",
  "runtime_id": "<candidate runtime_id>",
  "context_scope": "ct_abdomen_contrast",
  "cases_total": 100,
  "prediction_inventory_sha256": "<SHA списка case/audio/prediction>",
  "critical_facts_total": 123,
  "critical_facts_with_provenance": 123,
  "unsupported_critical_facts": 0
}
```

`prediction_inventory_sha256`, `context_scope` и `cases_total` привязывают
structuring-проверку к тем же raw-транскриптам, которые проходят release gate,
а не только к той же модели. Нулевой denominator недопустим. Финальный offline
release-прогон одновременно
создаёт десять конкурентных запросов и проверяет все доступные gates:

```powershell
python server\scripts\asr_benchmark.py `
  --manifest D:\asr-snapshots\ct-abdomen-v1\real-frozen-benchmark.tsv `
  --server-url http://127.0.0.1:9002 `
  --purpose release `
  --repeat 10 `
  --concurrency 10 `
  --context-scope ct_abdomen_contrast `
  --baseline-report D:\asr-results\v3-ctc-baseline.json `
  --sequential-report D:\asr-results\candidate-sequential.json `
  --structuring-safety-report D:\asr-results\candidate-structure-safety.json `
  --dataset-snapshot D:\asr-snapshots\ct-abdomen-v1\dataset-snapshot.json `
  --output D:\asr-results\candidate-release.json
if ($LASTEXITCODE -ne 0) { throw "ASR release gates failed" }
```

`--purpose release` запрещает `--limit`, требует именно `repeat=10` и
`concurrency=10`, а также полный детерминированный baseline
`repeat=10/concurrency=1`. Runner сверяет snapshot/manifest/runtime,
последовательный и конкурентный raw SHA, полный набор case ID, baseline
regressions, ненулевое покрытие entity spans,
RTF и structuring provenance. Candidate runtime обязан показать strict lock,
verified checkpoint и чистый project source. Отсутствующий или проваленный
gate записывается в `release_gate.failed`, а процесс завершается кодом `2`.

Synthetic regression запускается отдельно и никогда не создаёт release gate:

```powershell
python server\scripts\asr_benchmark.py `
  --manifest D:\asr-snapshots\ct-abdomen-v1\synthetic-regression-benchmark.tsv `
  --server-url http://127.0.0.1:9002 `
  --purpose regression `
  --output D:\asr-results\v3-ctc-tts-regression.json
```

Runner откажется смешать `real_frozen_test` и `synthetic_regression`, отклонит
`browser_asr`, потребует `human_verbatim` для real и `tts_script` для synthetic.
Эти признаки остаются governance declarations, поэтому release дополнительно
привязывается к неизменяемому snapshot; код сам по waveform не может доказать,
что запись сделана человеком. Для каждого прогона фиксируются manifest,
audio/reference/prediction SHA, `runtime_id`, confidence, latency и
детерминизм.

На одном snapshot сравниваются `v3_ctc`, `v3_rnnt`, `v3_e2e_ctc`,
`v3_e2e_rnnt` и Whisper challenger. Нельзя менять normalization или labels
между моделями. Отдельный safety-set содержит минимальные пары:
справа/слева, есть/нет, 15/50, мм/см, с/без контраста, отрицания и размеры.

Обязательные метрики:

- raw и normalized WER/CER;
- span-based Medical WER и recall медицинских терминов;
- exact accuracy чисел с единицами;
- accuracy отрицаний, латеральности и контраста;
- unsupported-addition rate structuring-слоя;
- p50/p95 latency и real-time factor;
- число ручных исправлений и время проверки врачом.

Release gates относительно замороженного baseline:

- Medical WER улучшен минимум на 25% относительно;
- общий WER ухудшен не более чем на 1 процентный пункт;
- recall критических сущностей не ниже 98%;
- числа+единицы, отрицания и латеральность не ниже 99%;
- ноль неподтверждённых критических фактов на safety-set;
- p95 RTF не выше 0,5;
- 10 последовательных и 10 конкурентных запросов дают одинаковый raw SHA;
- после shadow-пилота медиана не более двух правок и 60 секунд проверки.

Текущий runner считает Medical WER/recall и safety accuracy только по reviewed
`entities_json`. Для типа без аннотаций он возвращает `available=false`, а не
ложные 100%. Старые bag-of-token проверки остаются только в
`scores.diagnostics` с `promotion_eligible=false`: они не сохраняют связь
«орган ↔ сторона/размер». Medical entity F1 и unsupported additions считаются
в интеграционном benchmark structuring-слоя, где есть predicted entities и
span provenance.

## 5. Fine-tuning

Эксперимент фиксируется конфигурацией
`server/scripts/gigaam-finetune.ct-abdomen.example.json`. На машине обучения
сначала устанавливается pinned runtime (включая Torch/Torchaudio) из раздела 1,
затем клонируется тот же upstream commit:

```bash
git clone https://github.com/salute-developers/GigaAM.git
cd GigaAM
git checkout 559d88d6b72541412743929f633a6ae7c9950b85
pip install -e ".[train]"
cd train_utils
```

Первый grid запускается для LR `1e-5`, `2e-5`, `5e-5`, с разными
`--exp_name`, но одним dataset snapshot и seed:

```bash
python train.py \
  --model_name v3_ctc \
  --train_manifest /snapshots/ct-abdomen-v1/train.tsv \
  --val_manifest /snapshots/ct-abdomen-v1/validation.tsv \
  --raw_text \
  --max_epochs 3 \
  --val_check_interval 0.5 \
  --batch_size 8 \
  --eval_batch_size 32 \
  --lr 2e-5 \
  --activation_checkpointing \
  --seed 42 \
  --save_top_k 8 \
  --exp_name ct-abdomen-v1-lr2e-5
```

SpecAugment включён официальным train script по умолчанию. Важно:
upstream `val_wer` — обычный WER по всей расшифровке, а не Medical WER.
Официальный `ModelCheckpoint` мониторит именно этот обычный `val_wer`.
`--save_top_k 8` сохраняет все checkpoints небольшого трёхэпохового пилота;
число увеличивают, если validation событий больше восьми.

Каждый сохранённый checkpoint затем поднимается как `GIGAAM_MODEL` с
обязательным `GIGAAM_MODEL_CHECKSUM=sha256:<hex>` и прогоняется на
`real-validation-benchmark.tsv`:

```powershell
python server\scripts\asr_benchmark.py `
  --manifest D:\asr-snapshots\ct-abdomen-v1\real-validation-benchmark.tsv `
  --server-url http://127.0.0.1:9002 `
  --purpose validation `
  --output D:\asr-results\validation-medical-<checkpoint>.json
```

Epoch и LR выбираются post-hoc по минимальному span-based Medical WER на
зафиксированной medical validation, а не по имени `val_wer`. Real frozen test
не используется для выбора epoch, LR, словаря или decoder parameters. После
выбора единственный кандидат проходит frozen baseline/release workflow из
раздела 4. Русский tokenizer не меняется. Латинские DWI/ADC/T1/T2, числа и
единицы канонизируются отдельным проверяемым слоем.

## 6. Promotion, rollout и rollback

Кандидат модели получает immutable запись:

- SHA-256 checkpoint и base-model MD5;
- upstream commit и `pip freeze`;
- dataset snapshot SHA и train command/config;
- все offline metrics на real frozen test и synthetic regression;
- reviewer, дата, известные ограничения.

Promotion идёт только по стадиям:

1. offline challenger;
2. shadow минимум на 100 реальных протоколах;
3. canary на 1–2 врачах;
4. ограниченный rollout отдела.

Автоподписание запрещено. Заключение и числа требуют provenance из raw span.
При нарушении safety gate, росте latency, дрейфе ручных правок или ошибке
checksum трафик возвращается на предыдущий checkpoint заменой
`GIGAAM_MODEL` и соответствующего `GIGAAM_MODEL_CHECKSUM`; прежний artifact и
runtime lock не удаляются. Каждая смена версии записывается вместе с временем
и оператором.
