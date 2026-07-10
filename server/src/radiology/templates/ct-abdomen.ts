// КТ органов брюшной полости — эталонный шаблон v1 (ТЗ разделы 5–22, 34).
// Контент (формулировки, формулы, пороги) — из docs/radiology-ct-spec-v0.1.md.

import type {
  Conflict, FindingView, ProtocolState, RadiologyTemplate, TechniqueState,
} from '../schema.js';

// ─── Форматирование чисел (ru) ───────────────────────────────────────────────
const ru = (n: number): string => String(n).replace('.', ',');
const dim = (n: number): string => (Number.isInteger(n) ? `${n},0` : ru(n)); // 14 → «14,0»
const has = (v: FindingView, f: string): boolean => v.flags[f] === true;
// Склонения стороны по роду органа (надпочечник — м.р., почка — ж.р.).
const sideAdrenalPrep = (v: FindingView): string =>
  v.side === 'right' ? 'В правом надпочечнике' : v.side === 'left' ? 'В левом надпочечнике' : 'В надпочечнике';
const sideAdrenalGen = (v: FindingView): string =>
  v.side === 'right' ? 'правого надпочечника' : v.side === 'left' ? 'левого надпочечника' : 'надпочечника';
const sideKidneyPrep = (v: FindingView): string =>
  v.side === 'right' ? 'В правой почке' : v.side === 'left' ? 'В левой почке' : 'В почке';
const sideKidneyGen = (v: FindingView): string =>
  v.side === 'right' ? 'правой почки' : v.side === 'left' ? 'левой почки' : 'почки';

// ─── Техника ─────────────────────────────────────────────────────────────────
function techniqueText(t: TechniqueState): string {
  const phaseNames: Record<string, string> = {
    arterial: 'артериальной', portal: 'портально-венозной', delayed: 'отсроченной',
    pancreatic: 'панкреатической', excretory: 'экскреторной',
  };
  const contrastPhases = t.phases.filter((p) => p !== 'native');
  let s: string;
  if (!t.studyType || t.studyType === 'native') {
    s = 'Выполнена КТ органов брюшной полости без внутривенного контрастирования.';
  } else if (contrastPhases.length >= 2) {
    const list = contrastPhases.map((p) => phaseNames[p]).filter(Boolean).join(', ');
    s = `Выполнена КТ органов брюшной полости с внутривенным болюсным контрастированием, с получением ${list} фаз.`;
  } else {
    s = 'Выполнена КТ органов брюшной полости с внутривенным контрастированием.';
  }
  const q: Record<string, string> = {
    diagnostic: 'Качество исследования диагностическое.',
    limited_artifacts: 'Качество исследования ограничено артефактами.',
    limited_breathing: 'Качество исследования ограничено дыхательными артефактами.',
    limited_no_contrast: 'Качество исследования ограничено отсутствием контрастирования.',
    limited_prep: 'Качество исследования ограничено подготовкой пациента.',
  };
  if (t.quality) s += ' ' + q[t.quality];
  return s;
}

export const ctAbdomen: RadiologyTemplate = {
  id: 'CT_ABDOMEN',
  name: 'КТ органов брюшной полости',
  modality: 'CT',
  aliases: ['обп', 'кт обп', 'брюшная полость', 'кт брюшной полости', 'живот'],
  emptyConclusion: 'Патологических изменений органов брюшной полости не выявлено.',

  technique: [
    { triggers: ['натив', 'нативн'], apply: (t) => { t.studyType = 'native'; t.phases = ['native']; } },
    {
      triggers: ['три фаз', '3 фаз', 'трёхфазн', 'трехфазн'],
      apply: (t) => { t.studyType = 'multiphase'; t.phases = ['native', 'arterial', 'portal', 'delayed']; },
    },
    {
      triggers: ['панкреатическ'],
      apply: (t) => { t.studyType = 'multiphase'; if (!t.phases.includes('pancreatic')) t.phases.push('pancreatic'); if (!t.phases.includes('arterial')) t.phases.push('arterial'); if (!t.phases.includes('portal')) t.phases.push('portal'); },
    },
    { triggers: ['артериальн'], apply: (t) => { if (!t.phases.includes('arterial')) t.phases.push('arterial'); if (!t.studyType) t.studyType = 'contrast'; } },
    { triggers: ['венозн', 'портально'], apply: (t) => { if (!t.phases.includes('portal')) t.phases.push('portal'); if (!t.studyType) t.studyType = 'contrast'; } },
    { triggers: ['отсроч'], apply: (t) => { if (!t.phases.includes('delayed')) t.phases.push('delayed'); if (!t.studyType) t.studyType = 'contrast'; } },
    {
      triggers: ['контраст'],
      apply: (t) => { t.studyType = 'contrast'; if (t.phases.length === 0) t.phases = ['native', 'portal']; },
    },
    { triggers: ['качество хорошее', 'качество диагностическое'], apply: (t) => { t.quality = 'diagnostic'; } },
    { triggers: ['ограничено дыхан', 'дыхательными артефактами'], apply: (t) => { t.quality = 'limited_breathing'; } },
    { triggers: ['ограничено артефактами'], apply: (t) => { t.quality = 'limited_artifacts'; } },
  ],
  techniqueText,

  sections: [
    // ─── Печень ──────────────────────────────────────────────────────────────
    {
      id: 'liver',
      organ: 'Печень',
      anchors: ['печень'],
      normal: {
        measurements: [{ name: 'density', role: 'keyword', keywords: ['плотность'], unit: 'HU', physRange: [20, 90] }],
        render: (v) => {
          const d = v.slots.density;
          const huText = d !== undefined ? `${ru(d)} HU` : '___ HU';
          const warnings: string[] = [];
          let conclusion: string | undefined;
          if (d === undefined) warnings.push('Печень: не указана средняя плотность паренхимы (обязательное поле).');
          else if (d < 40) conclusion = 'КТ-признаки жировой инфильтрации печени.';
          return {
            description: `Печень обычных размеров, контуры ровные, структура паренхимы однородная. Средняя плотность паренхимы печени на нативных изображениях — ${huText}. Очаговых образований не выявлено. Внутрипечёночные желчные протоки не расширены.`,
            warnings, conclusion,
          };
        },
      },
      findings: [
        {
          id: 'steatosis',
          triggers: ['стеатоз', 'жировая инфильтрация', 'жировой гепатоз'],
          slots: [{ name: 'density', role: 'keyword', keywords: ['плотность'], unit: 'HU' }],
          render: (v) => {
            const d = v.slots.density;
            const huText = d !== undefined ? `${ru(d)} HU` : '___ HU';
            return {
              description: `Печень обычных размеров, контуры ровные. Структура паренхимы диффузно снижена по плотности, средняя плотность паренхимы на нативных изображениях — ${huText}. Очаговых образований не выявлено.`,
              conclusion: 'КТ-признаки жировой инфильтрации (стеатоза) печени.',
              warnings: d !== undefined && d >= 40 ? ['Печень: плотность ≥40 HU не подтверждает стеатоз — проверьте.'] : undefined,
            };
          },
        },
        {
          id: 'cirrhosis',
          triggers: ['цирроз'],
          flags: [{ name: 'noHcc', phrases: ['без гцр', 'без гцк'] }],
          render: (v) => {
            const warnings: string[] = [];
            let lr = '';
            if (has(v, 'noHcc')) {
              if (v.ctx.isMultiphase()) {
                lr = ' В артериальную, портально-венозную и отсроченную фазы наблюдений с типичными признаками ГЦР не выявлено; наблюдений LR-4/LR-5 не определяется.';
              } else {
                lr = ' Оценка по LI-RADS ограничена отсутствием многофазного контрастирования.';
                warnings.push('Цирроз без ГЦР: формулировка LR-4/LR-5 недоступна без многофазного контраста.');
              }
            }
            return {
              description: `Печень изменена по цирротическому типу: контуры неровные, бугристые, отмечается диспропорция долей. Структура паренхимы неоднородная.${lr}`,
              conclusion: has(v, 'noHcc') && v.ctx.isMultiphase()
                ? 'КТ-признаки цирроза печени. Данных за ГЦР не получено; наблюдений LR-4/LR-5 не выявлено.'
                : 'КТ-признаки цирроза печени.',
              warnings: warnings.length ? warnings : undefined,
            };
          },
        },
      ],
    },

    // ─── Очаговые образования печени (repeatable) ─────────────────────────────
    {
      id: 'liver_lesions',
      organ: 'Очаговые образования печени',
      anchors: ['печень киста', 'печень гемангиома', 'печень фнг', 'печень гцр', 'печень метастаз',
        'киста печени', 'гемангиома печени', 'метастазы печени', 'очаг печени'],
      repeatable: true,
      normal: { render: () => ({ description: '' }) },
      findings: [
        {
          id: 'cyst',
          triggers: ['киста'],
          slots: [{ name: 'size', role: 'bareSize', unit: 'мм' }],
          render: (v) => {
            const seg = v.segments[0] ?? 'S?';
            const sz = v.slots.size !== undefined ? `${ru(v.slots.size)} мм` : '___ мм';
            return {
              description: `В ${seg} печени определяется простая киста диаметром ${sz}: жидкостной плотности, с чёткими ровными контурами, без перегородок, солидного компонента и патологического контрастного усиления.`,
              conclusion: `Простая киста ${seg} печени.`,
            };
          },
        },
        {
          id: 'hemangioma',
          triggers: ['гемангиома'],
          requiresPhases: ['arterial'],
          slots: [{ name: 'size', role: 'bareSize', unit: 'мм' }],
          render: (v) => {
            const seg = v.segments[0] ?? 'S?';
            const sz = v.slots.size !== undefined ? `до ${ru(v.slots.size)} мм` : 'до ___ мм';
            const typical = v.ctx.isMultiphase();
            if (typical) {
              return {
                description: `В ${seg} печени определяется образование ${sz} с типичным для гемангиомы паттерном контрастирования: периферическое дискретное узловое накопление контрастного препарата в артериальную фазу с прогрессирующим центрипетальным заполнением в последующие фазы.`,
                conclusion: `КТ-картина типичной гемангиомы ${seg} печени ${sz}.`,
              };
            }
            return {
              description: `В ${seg} печени определяется образование ${sz} с признаками, вероятно соответствующими гемангиоме; оценка ограничена фазностью исследования.`,
              conclusion: `Очаг ${seg} печени ${sz}, вероятно гемангиома (оценка ограничена фазностью).`,
              warnings: ['Гемангиома: без артериальной и венозной/отсроченной фаз формулировка «типичная» недоступна.'],
            };
          },
        },
        {
          id: 'fnh',
          triggers: ['фнг', 'фокальная нодулярная', 'нодулярная гиперплазия'],
          flags: [{ name: 'scar', phrases: ['рубец', 'центральный рубец'] }, { name: 'washout', phrases: ['washout', 'вымывание'] }],
          slots: [{ name: 'size', role: 'bareSize', unit: 'мм' }],
          render: (v) => {
            const seg = v.segments[0] ?? 'S?';
            const sz = v.slots.size !== undefined ? `до ${ru(v.slots.size)} мм` : 'до ___ мм';
            const scar = has(v, 'scar')
              ? 'наличие центрального рубца с отсроченным накоплением контрастного препарата'
              : 'без убедительно визуализируемого центрального рубца';
            return {
              description: `В ${seg} печени определяется образование ${sz} с признаками, наиболее соответствующими фокальной нодулярной гиперплазии: интенсивное однородное артериальное контрастное усиление, отсутствие патологического washout, ${scar}.`,
              conclusion: `КТ-картина фокальной нодулярной гиперплазии ${seg} печени ${sz}.`,
              warnings: has(v, 'washout') ? ['ФНГ обычно не имеет washout — уточните тип очага.'] : undefined,
            };
          },
        },
        {
          id: 'hcc',
          triggers: ['гцр', 'гцк', 'гепатоцеллюлярн'],
          flags: [
            { name: 'aphe', phrases: ['aphe', 'артериальное усиление', 'гиперусиление'] },
            { name: 'washout', phrases: ['washout', 'вымывание'] },
            { name: 'capsule', phrases: ['капсула'] },
          ],
          slots: [
            { name: 'size', role: 'bareSize', unit: 'мм' },
            { name: 'lirads', role: 'category', keywords: ['li-rads', 'lirads', 'lr', 'ли-радс'] },
          ],
          render: (v) => {
            const seg = v.segments[0] ?? 'S?';
            const sz = v.slots.size !== undefined ? `до ${ru(v.slots.size)} мм` : 'до ___ мм';
            const warnings: string[] = [];
            const multiphase = v.ctx.isMultiphase();
            let lrText = '';
            if (v.slots.lirads !== undefined) {
              if (multiphase && v.ctx.hccRisk) lrText = ` КТ-картина соответствует наблюдению LI-RADS ${v.slots.lirads}.`;
              else {
                warnings.push('LI-RADS требует многофазного КТ и пациента группы риска ГЦР — категория не выставлена.');
                lrText = ' Оценка по LI-RADS ограничена (нет многофазного контраста или группы риска).';
              }
            }
            const feats = [has(v, 'aphe') && 'некраевое артериальное гиперусиление',
              has(v, 'washout') && 'некраевой washout', has(v, 'capsule') && 'усиливающаяся капсула']
              .filter(Boolean).join(', ');
            return {
              description: `В ${seg} печени определяется образование ${sz}${feats ? `: ${feats}` : ''}.${lrText}`,
              conclusion: `Очаг ${seg} печени ${sz} с КТ-признаками ГЦР${v.slots.lirads !== undefined && multiphase && v.ctx.hccRisk ? `, LI-RADS ${v.slots.lirads}` : ''}.`,
              critical: true,
              warnings: warnings.length ? warnings : undefined,
            };
          },
        },
        {
          id: 'metastases',
          triggers: ['метастаз'],
          slots: [
            { name: 'min', role: 'keyword', keywords: ['от'], unit: 'мм' },
            { name: 'max', role: 'keyword', keywords: ['до'], unit: 'мм' },
            { name: 'size', role: 'bareSize', unit: 'мм' },
          ],
          render: (v) => {
            const segs = v.segments.length ? v.segments.join(', ') : '';
            const sizeText = v.slots.min !== undefined && v.slots.max !== undefined
              ? `размерами от ${ru(v.slots.min)} до ${ru(v.slots.max)} мм`
              : v.slots.size !== undefined ? `до ${ru(v.slots.size)} мм` : '';
            const segText = segs ? `, расположенные в ${segs}` : '';
            return {
              description: `В печени определяются множественные гиповаскулярные очаги, преимущественно гиподенсные в портально-венозную фазу${sizeText ? ', ' + sizeText : ''}${segText}; КТ-картина наиболее соответствует метастатическому поражению.`,
              conclusion: 'КТ-признаки множественного гиповаскулярного метастатического поражения печени.',
              critical: true,
            };
          },
        },
      ],
    },

    // ─── Желчные протоки ──────────────────────────────────────────────────────
    {
      id: 'bile_ducts',
      organ: 'Желчные протоки',
      anchors: ['холедох', 'желчные протоки', 'холедохолитиаз', 'внутрипечёночные протоки'],
      normal: {
        render: () => ({ description: 'Внутри- и внепечёночные желчные протоки не расширены. Холедох не расширен, просвет его свободен.' }),
      },
      findings: [
        {
          id: 'choledocholithiasis',
          triggers: ['холедохолитиаз', 'камень холедоха', 'конкремент холедоха'],
          slots: [
            { name: 'choledoch', role: 'keyword', keywords: ['холедох'], unit: 'мм' },
            { name: 'stone', role: 'keyword', keywords: ['камень', 'конкремент'], unit: 'мм' },
          ],
          render: (v) => {
            const ch = v.slots.choledoch !== undefined ? `${ru(v.slots.choledoch)} мм` : '___ мм';
            const st = v.slots.stone !== undefined ? `${ru(v.slots.stone)} мм` : '___ мм';
            return {
              description: `Холедох расширен до ${ch}. В просвете дистального отдела холедоха определяется конкремент до ${st}. Внутрипечёночные желчные протоки умеренно расширены.`,
              conclusion: 'КТ-признаки холедохолитиаза с билиарной гипертензией.',
            };
          },
        },
      ],
    },

    // ─── Желчный пузырь ───────────────────────────────────────────────────────
    {
      id: 'gallbladder',
      organ: 'Желчный пузырь',
      anchors: ['желчный пузырь', 'желчный', 'холецистит', 'желчного пузыря'],
      normal: {
        render: () => ({ description: 'Желчный пузырь обычных размеров, стенка не утолщена. Рентгенконтрастных конкрементов в просвете не определяется. Перивезикальная клетчатка не изменена.' }),
      },
      removable: {
        triggers: ['удалён', 'удален', 'удалена', 'холецистэктоми'],
        render: () => ({
          description: 'Желчный пузырь удалён. В ложе желчного пузыря патологических жидкостных скоплений не определяется. Внутри- и внепечёночные желчные протоки без значимой дилатации.',
        }),
      },
      findings: [
        {
          id: 'cholelithiasis',
          triggers: ['камни без воспаления', 'воспаления нет', 'без воспаления'],
          render: () => ({
            description: 'В просвете желчного пузыря определяются конкременты. Стенка желчного пузыря не утолщена, перивезикальная клетчатка не инфильтрирована, перивезикальной жидкости не определяется.',
            conclusion: 'КТ-признаки желчнокаменной болезни без признаков острого холецистита.',
          }),
        },
        {
          id: 'acute_cholecystitis',
          triggers: ['холецистит'],
          slots: [{ name: 'wall', role: 'keyword', keywords: ['стенка'], unit: 'мм' }],
          flags: [
            { name: 'neck', phrases: ['шейка', 'шейке', 'пузырном протоке'] },
            { name: 'fluid', phrases: ['перивезикальная жидкость', 'перивезикальной жидкости', 'жидкость'] },
          ],
          render: (v) => {
            const wall = v.slots.wall !== undefined ? `до ${ru(v.slots.wall)} мм` : 'до ___ мм';
            const neck = has(v, 'neck') ? ' В области шейки желчного пузыря определяется конкремент.' : '';
            const fluid = has(v, 'fluid')
              ? ' Перивезикальная клетчатка инфильтрирована, определяется небольшое количество перивезикальной жидкости.'
              : '';
            return {
              description: `Желчный пузырь увеличен. Стенка утолщена ${wall}, отмечается её контрастное усиление.${neck}${fluid} КТ-картина соответствует острому калькулёзному холециститу.`,
              conclusion: 'КТ-признаки острого калькулёзного холецистита.',
              critical: true,
            };
          },
        },
      ],
    },

    // ─── Поджелудочная железа ─────────────────────────────────────────────────
    {
      id: 'pancreas',
      organ: 'Поджелудочная железа',
      anchors: ['поджелудочная', 'поджелудочной', 'панкреатит'],
      normal: {
        render: () => ({ description: 'Поджелудочная железа обычных размеров и формы, контуры чёткие, структура однородная. Панкреатический проток не расширен. Парапанкреатическая клетчатка не инфильтрирована.' }),
      },
      findings: [
        {
          id: 'lipomatosis',
          triggers: ['липоматоз', 'жировое замещение'],
          render: () => ({
            description: 'Поджелудочная железа обычных размеров, с диффузным жировым замещением паренхимы. Панкреатический проток не расширен. Очагового образования не выявлено. Парапанкреатическая клетчатка не инфильтрирована.',
            conclusion: 'КТ-признаки липоматоза поджелудочной железы.',
          }),
        },
        {
          id: 'chronic_pancreatitis',
          triggers: ['хронический панкреатит', 'панкреатит'],
          slots: [{ name: 'duct', role: 'keyword', keywords: ['проток'], unit: 'мм' }],
          flags: [{ name: 'calc', phrases: ['кальцинат'] }, { name: 'atrophy', phrases: ['атрофия', 'уменьшена'] }],
          render: (v) => {
            const duct = v.slots.duct !== undefined ? `до ${ru(v.slots.duct)} мм` : 'до ___ мм';
            const calc = has(v, 'calc') ? ' В паренхиме определяются множественные кальцинаты.' : '';
            const atr = has(v, 'atrophy') ? 'Поджелудочная железа уменьшена в объёме, ' : 'Поджелудочная железа ';
            return {
              description: `${atr}структура неоднородная.${calc} Главный панкреатический проток расширен ${duct}. Парапанкреатическая клетчатка без признаков острого воспаления.`,
              conclusion: 'КТ-признаки хронического кальцифицирующего панкреатита с дилатацией главного панкреатического протока.',
            };
          },
        },
      ],
    },

    // ─── Селезёнка ────────────────────────────────────────────────────────────
    {
      id: 'spleen',
      organ: 'Селезёнка',
      anchors: ['селезёнка', 'селезенка', 'селезёнки', 'селезенки'],
      normal: {
        measurements: [{ name: 'dims', role: 'dimensions', keywords: ['размеры', 'размер'], count: 3, unit: 'см' }],
        render: (v) => {
          const d = v.dims.dims;
          if (d && d.length === 3) {
            const [l, w, t] = d;
            const index = l * w * t;
            const conclusion = index > 480 ? 'КТ-признаки спленомегалии.' : undefined;
            return {
              description: `Селезёнка размерами ${dim(l)} × ${dim(w)} × ${dim(t)} см, селезёночный индекс — ${ru(Math.round(index))}. Структура паренхимы однородная, очаговых изменений не выявлено.`,
              conclusion,
            };
          }
          return { description: 'Селезёнка не увеличена, структура однородная, очаговых изменений не выявлено.' };
        },
      },
    },

    // ─── Портальная система ───────────────────────────────────────────────────
    {
      id: 'portal',
      organ: 'Портальная система',
      anchors: ['портальная система', 'портальная', 'портальной', 'воротная вена'],
      normal: {
        render: () => ({ description: 'Воротная вена, верхняя брыжеечная и селезёночная вены проходимы, не расширены. Признаков тромбоза, кавернозной трансформации воротной вены и портосистемных коллатералей не выявлено. Варикозно расширенных вен пищевода и желудка не определяется.' }),
      },
      findings: [
        {
          id: 'portal_hypertension',
          triggers: ['гипертензия', 'портальная гипертензия'],
          slots: [
            { name: 'portal', role: 'keyword', keywords: ['воротная'], unit: 'мм' },
            { name: 'smv', role: 'keyword', keywords: ['вбв', 'брыжеечная'], unit: 'мм' },
            { name: 'splenic', role: 'keyword', keywords: ['селезеночная'], unit: 'мм' },
          ],
          flags: [
            { name: 'varixEso', phrases: ['вариксы пищевода', 'вены пищевода', 'пищевода'] },
            { name: 'varixGastric', phrases: ['вариксы желудка', 'вены желудка', 'желудка'] },
            { name: 'ascites', phrases: ['асцит'] },
          ],
          render: (v) => {
            const p = v.slots.portal, s = v.slots.smv, sp = v.slots.splenic;
            const parts: string[] = [];
            if (p !== undefined) parts.push(`Воротная вена расширена до ${ru(p)} мм`);
            if (s !== undefined) parts.push(`верхняя брыжеечная вена — до ${ru(s)} мм`);
            if (sp !== undefined) parts.push(`селезёночная вена — до ${ru(sp)} мм`);
            const varix = (has(v, 'varixEso') || has(v, 'varixGastric'))
              ? ' Определяются портосистемные коллатерали, варикозно расширенные вены пищевода и желудка.' : '';
            const asc = has(v, 'ascites') ? ' В брюшной полости определяется свободная жидкость.' : '';
            return {
              description: `${parts.join(', ')}.${varix}${asc}`,
              conclusion: 'КТ-признаки портальной гипертензии.',
              critical: true,
            };
          },
        },
      ],
    },

    // ─── Желудок и парагастральные лимфоузлы ──────────────────────────────────
    {
      id: 'stomach',
      organ: 'Желудок',
      anchors: ['желудок', 'гпод', 'парагастральные', 'грыжа пищеводного'],
      normal: {
        render: () => ({ description: 'Желудок умеренно наполнен, стенки его без убедительного патологического утолщения. Парагастральная клетчатка не инфильтрирована. Парагастральные лимфатические узлы не увеличены.' }),
      },
      findings: [
        {
          id: 'gpod',
          triggers: ['гпод', 'грыжа пищеводного'],
          slots: [
            { name: 'height', role: 'keyword', keywords: ['высота'], unit: 'мм' },
            { name: 'width', role: 'keyword', keywords: ['ширина'], unit: 'мм' },
            { name: 'size', role: 'bareSize', unit: 'мм' },
          ],
          render: (v) => {
            const h = v.slots.height ?? v.slots.size;
            const hText = h !== undefined ? `около ${ru(h)} мм` : 'около ___ мм';
            const w = v.slots.width !== undefined ? ` поперечный размер до ${ru(v.slots.width)} мм.` : '';
            return {
              description: `Определяется грыжа пищеводного отверстия диафрагмы: кардиальный отдел желудка смещён выше уровня диафрагмы, высота грыжевого компонента ${hText}.${w}`,
              conclusion: 'КТ-признаки грыжи пищеводного отверстия диафрагмы.',
            };
          },
        },
      ],
    },

    // ─── Надпочечники ─────────────────────────────────────────────────────────
    {
      id: 'adrenals',
      organ: 'Надпочечники',
      anchors: ['надпочечник', 'надпочечники', 'надпочечнике'],
      normal: {
        render: () => ({ description: 'Надпочечники обычной формы и размеров, дополнительных образований не выявлено.' }),
      },
      findings: [
        {
          id: 'adenoma',
          triggers: ['аденома', 'образование'],
          slots: [
            { name: 'size', role: 'bareSize', unit: 'мм' },
            { name: 'huNative', role: 'keyword', keywords: ['натив', 'нативная'], unit: 'HU' },
            { name: 'huVenous', role: 'keyword', keywords: ['венозная'], unit: 'HU' },
            { name: 'huDelayed', role: 'keyword', keywords: ['отсрочка', 'отсроченная'], unit: 'HU' },
          ],
          render: (v) => {
            const loc = sideAdrenalPrep(v);
            const gen = sideAdrenalGen(v);
            const sz = v.slots.size !== undefined ? `до ${ru(v.slots.size)} мм` : 'до ___ мм';
            const nat = v.slots.huNative, ven = v.slots.huVenous, del = v.slots.huDelayed;

            // липид-содержащая аденома по нативной плотности ≤10 HU
            if (nat !== undefined && nat <= 10 && ven === undefined) {
              return {
                description: `${loc} определяется образование ${sz} с нативной плотностью ${ru(nat)} HU, что соответствует липид-содержащей аденоме.`,
                conclusion: `Липид-содержащая аденома ${gen}.`,
              };
            }
            // расчёт washout
            if (ven !== undefined && del !== undefined) {
              const rel = Math.round((ven - del) / ven * 100);
              const warnings: string[] = [];
              let abs: number | undefined;
              if (nat !== undefined) abs = Math.round((ven - del) / (ven - nat) * 100);
              else warnings.push('Надпочечник: нет нативной плотности — абсолютный washout не рассчитан.');
              const washText = abs !== undefined
                ? `Абсолютный washout составляет ${abs}%, относительный washout — ${rel}%`
                : `Относительный washout составляет ${rel}%`;
              return {
                description: `${loc} определяется образование ${sz}. Плотность на нативных изображениях — ${nat !== undefined ? ru(nat) : '___'} HU, в портально-венозную фазу — ${ru(ven)} HU, в отсроченную фазу — ${ru(del)} HU. ${washText}, что соответствует КТ-признакам аденомы надпочечника.`,
                conclusion: `КТ-признаки аденомы ${gen}.`,
                warnings: warnings.length ? warnings : undefined,
              };
            }
            // недостаточно данных
            return {
              description: `${loc} определяется образование ${sz}.`,
              conclusion: `Образование ${gen}, требует дообследования.`,
              warnings: ['Надпочечник: для дифференцировки аденомы нужны нативная плотность или венозная+отсроченная фазы.'],
            };
          },
        },
      ],
    },

    // ─── Почки ────────────────────────────────────────────────────────────────
    {
      id: 'kidneys',
      organ: 'Почки',
      anchors: ['почка', 'почки', 'почек', 'почке'],
      normal: {
        render: () => ({ description: 'Почки расположены типично, обычных размеров и формы. Паренхима сохранена. Чашечно-лоханочные системы не расширены. Конкрементов и объёмных образований не выявлено.' }),
      },
      findings: [
        {
          id: 'bosniak_cyst',
          triggers: ['босняк', 'bosniak', 'киста'],
          slots: [
            { name: 'bosniak', role: 'category', keywords: ['босняк', 'bosniak'] },
            { name: 'size', role: 'bareSize', unit: 'мм' },
          ],
          flags: [
            { name: 'septa', phrases: ['перегородк'] },
            { name: 'solid', phrases: ['солидн'] },
            { name: 'enhancement', phrases: ['усилен', 'накопление контраста'] },
            { name: 'calc', phrases: ['кальцинат'] },
          ],
          render: (v) => {
            const loc = sideKidneyPrep(v);
            const gen = sideKidneyGen(v);
            const cat = v.slots.bosniak;
            const roman = cat ? ['', 'I', 'II', 'III', 'IV'][cat] ?? String(cat) : '';
            const sz = v.slots.size !== undefined ? `диаметром ${ru(v.slots.size)} мм` : 'диаметром ___ мм';
            const warnings: string[] = [];
            if (cat === 1 && (has(v, 'septa') || has(v, 'solid') || has(v, 'enhancement'))) {
              warnings.push('Bosniak I несовместим с перегородками/солидным компонентом/усилением — проверьте категорию.');
            }
            return {
              description: `${loc} определяется простая кортикальная киста ${sz}: тонкостенная, жидкостной плотности, без перегородок, кальцинатов, солидного компонента и патологического контрастного усиления. Bosniak ${roman || '___'}.`,
              conclusion: `Простая киста ${gen} Bosniak ${roman || '___'}.`,
              warnings: warnings.length ? warnings : undefined,
            };
          },
        },
      ],
    },

    // ─── ЧЛС и мочеточники ────────────────────────────────────────────────────
    {
      id: 'urinary',
      organ: 'Чашечно-лоханочная система и мочеточники',
      anchors: ['члс', 'чашечно-лоханочная', 'мочеточник', 'гидронефроз'],
      normal: {
        render: () => ({ description: 'Чашечно-лоханочные системы не расширены. Мочеточники в зоне исследования не расширены. Рентгенконтрастных конкрементов не выявлено.' }),
      },
    },

    // ─── Кишечник ─────────────────────────────────────────────────────────────
    {
      id: 'bowel',
      organ: 'Кишечник в зоне исследования',
      anchors: ['кишечник', 'кишка', 'кишки'],
      normal: {
        render: () => ({ description: 'Петли кишечника в зоне исследования обычного расположения, без патологического утолщения стенок и признаков нарушения пассажа.' }),
      },
    },

    // ─── Брюшина, свободная жидкость и газ ─────────────────────────────────────
    {
      id: 'peritoneum',
      organ: 'Брюшина, свободная жидкость и газ',
      anchors: ['жидкость', 'жидкости', 'газ', 'газа', 'асцит', 'пневмоперитонеум',
        'свободный газ', 'свободная жидкость', 'брюшина'],
      normal: {
        render: () => ({ description: 'Свободной жидкости и свободного газа в брюшной полости не определяется.' }),
      },
      findings: [
        {
          id: 'free_gas',
          triggers: ['пневмоперитонеум', 'свободный газ', 'газ под диафрагмой'],
          render: () => ({
            description: 'В брюшной полости определяется свободный газ (пневмоперитонеум).',
            conclusion: 'Свободный газ в брюшной полости (пневмоперитонеум).',
            critical: true,
          }),
        },
        {
          id: 'free_fluid',
          triggers: ['асцит', 'свободная жидкость'],
          flags: [
            { name: 'small', phrases: ['малый', 'малое количество', 'небольш'] },
            { name: 'moderate', phrases: ['умеренн'] },
            { name: 'large', phrases: ['значительн', 'большое количество'] },
          ],
          render: (v) => {
            const amount = has(v, 'large') ? 'значительное количество'
              : has(v, 'moderate') ? 'умеренное количество'
                : has(v, 'small') ? 'небольшое количество' : 'свободную жидкость';
            return {
              description: `В брюшной полости определяется ${amount} свободной жидкости.`,
              conclusion: has(v, 'small') ? undefined : 'КТ-признаки асцита.',
              addToConclusion: !has(v, 'small'),
            };
          },
        },
      ],
    },

    // ─── Лимфатические узлы ────────────────────────────────────────────────────
    {
      id: 'lymph',
      organ: 'Лимфатические узлы',
      anchors: ['лимфоузлы', 'лимфатические узлы', 'лимфоузел'],
      normal: {
        render: () => ({ description: 'Увеличенных внутрибрюшных и забрюшинных лимфатических узлов не выявлено.' }),
      },
    },

    // ─── Сосуды ───────────────────────────────────────────────────────────────
    {
      id: 'vessels',
      organ: 'Сосуды брюшной полости',
      anchors: ['аорта', 'сосуды', 'чревный ствол'],
      normal: {
        render: () => ({ description: 'Брюшная аорта не расширена. Висцеральные ветви проходимы. Аневризматического расширения и признаков диссекции не выявлено.' }),
      },
    },

    // ─── Кости и мягкие ткани ─────────────────────────────────────────────────
    {
      id: 'bones',
      organ: 'Кости и мягкие ткани',
      anchors: ['кости', 'мягкие ткани', 'костные структуры'],
      normal: {
        render: () => ({ description: 'Костных деструктивных изменений в зоне исследования не выявлено. Мягкие ткани без особенностей.' }),
      },
    },
  ],

  // ─── Контроль противоречий (раздел 34 ТЗ) ────────────────────────────────────
  conflicts: (state: ProtocolState): Conflict[] => {
    const out: Conflict[] = [];
    const S = state.sections;

    // (#3 селезёнка/индекс — движок сам выносит спленомегалию в заключение, не противоречие.)

    // #4 Bosniak I + перегородки/солид/усиление
    const kd = S.kidneys;
    if (kd?.status === 'pathology') {
      for (const inst of kd.instances) {
        const f = inst.view.flags;
        if (inst.view.slots.bosniak === 1 && (f.septa || f.solid || f.enhancement)) {
          out.push({ code: 'bosniak1_conflict', sectionId: 'kidneys', message: 'Bosniak I несовместим с перегородками/солидным компонентом/усилением.' });
        }
      }
    }

    // #9 LI-RADS без многофазного контраста
    const multiphase = state.technique.phases.includes('arterial')
      && (state.technique.phases.includes('portal') || state.technique.phases.includes('delayed'));
    const les = S.liver_lesions;
    if (les?.status === 'pathology') {
      for (const inst of les.instances) {
        if (inst.findingId === 'hcc' && inst.view.slots.lirads !== undefined && !multiphase) {
          out.push({ code: 'lirads_no_multiphase', sectionId: 'liver_lesions', message: 'LI-RADS выставлен без многофазного контрастирования.' });
        }
      }
    }

    // #10 аденома по washout без отсроченной фазы
    const ad = S.adrenals;
    if (ad?.status === 'pathology') {
      for (const inst of ad.instances) {
        const sl = inst.view.slots;
        if (inst.findingId === 'adenoma' && sl.huVenous !== undefined && sl.huDelayed === undefined) {
          out.push({ code: 'washout_no_delayed', sectionId: 'adrenals', message: 'Расчёт washout невозможен без отсроченной фазы.' });
        }
      }
    }

    return out;
  },
};
