#!/usr/bin/env node
/**
 * Структурирует один raw-text файл через /api/structure и рендерит его в PDF,
 * используя тот же React-PDF шаблон, что и UI (src/components/MedicalPDFDocument.tsx).
 *
 * Usage: node scripts/qa-to-pdf.cjs <input.txt> [output.pdf]
 */
const fs = require('fs');
const path = require('path');
const React = require('react');
const ReactPDF = require('@react-pdf/renderer');

const { Document, Page, Text, View, StyleSheet, Font } = ReactPDF;
const e = React.createElement;

const SERVER = process.env.SERVER_URL || 'http://localhost:3001';
const AUTH_PASS = process.env.AUTH_PASS || 'meddok2026';

// --- Шрифты (Arial c кириллицей из src/assets/fonts) -----------------------
const FONT_DIR = path.resolve(__dirname, '..', '..', 'src', 'assets', 'fonts');
Font.register({
  family: 'ArialCyr',
  fonts: [
    { src: path.join(FONT_DIR, 'arial.ttf'), fontWeight: 'normal' },
    { src: path.join(FONT_DIR, 'arialbd.ttf'), fontWeight: 'bold' },
  ],
});

// --- Стили (один-в-один с MedicalPDFDocument.tsx) --------------------------
const styles = StyleSheet.create({
  page: { padding: 40, fontFamily: 'ArialCyr', fontSize: 11, lineHeight: 1.5, color: '#1e293b' },
  header: { marginBottom: 24, paddingBottom: 16, borderBottomWidth: 2, borderBottomColor: '#339678', borderBottomStyle: 'solid' },
  title: { fontSize: 18, fontWeight: 'bold', textAlign: 'center', color: '#1e4d41', marginBottom: 4 },
  subtitle: { fontSize: 10, textAlign: 'center', color: '#64748b' },
  patientBlock: { backgroundColor: '#f0f9f6', padding: 12, marginBottom: 20, borderRadius: 4 },
  patientRow: { flexDirection: 'row', marginBottom: 4 },
  patientLabel: { width: 120, fontWeight: 'bold', color: '#21614f' },
  patientValue: { flex: 1 },
  section: { marginBottom: 16 },
  sectionTitle: { fontSize: 12, fontWeight: 'bold', color: '#267860', marginBottom: 6, paddingBottom: 4, borderBottomWidth: 1, borderBottomColor: '#d8f0e8', borderBottomStyle: 'solid' },
  sectionContent: { fontSize: 11, lineHeight: 1.6, textAlign: 'justify' },
  footer: { position: 'absolute', bottom: 30, left: 40, right: 40, paddingTop: 12, borderTopWidth: 1, borderTopColor: '#e2e8f0', borderTopStyle: 'solid', flexDirection: 'row', justifyContent: 'space-between' },
  footerText: { fontSize: 9, color: '#94a3b8' },
  signatureLine: { marginTop: 40, flexDirection: 'row', justifyContent: 'flex-end' },
  signatureBlock: { width: 200 },
  signaturePlaceholder: { borderBottomWidth: 1, borderBottomColor: '#1e293b', borderBottomStyle: 'solid', marginBottom: 4, height: 30 },
  signatureLabel: { fontSize: 9, color: '#64748b', textAlign: 'center' },
});

const formatDate = (s) => {
  if (!s) return '';
  const d = new Date(s);
  if (isNaN(d.getTime())) return s;
  return d.toLocaleDateString('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric' });
};

function buildPdfTree(doc) {
  const p = doc.patient || {};
  const ra = doc.riskAssessment;

  const sections = [
    { title: 'Жалобы', content: doc.complaints },
    { title: 'Анамнез заболевания', content: doc.anamnesis },
    { title: 'Амбулаторные обследования', content: doc.outpatientExams },
    { title: 'Анамнез жизни', content: doc.clinicalCourse },
    { title: 'Аллергологический анамнез', content: doc.allergyHistory },
    { title: 'Объективный статус', content: doc.objectiveStatus },
    { title: 'Неврологический статус', content: doc.neurologicalStatus },
    { title: 'Предварительный диагноз', content: doc.diagnosis },
    { title: 'Заключительный диагноз', content: doc.finalDiagnosis },
    { title: 'План обследования', content: doc.doctorNotes },
    { title: 'Рекомендации / План лечения', content: doc.recommendations },
    { title: 'Амбулаторная терапия', content: doc.conclusion },
  ].filter((s) => s.content);

  const row = (label, value) =>
    e(View, { style: styles.patientRow },
      e(Text, { style: styles.patientLabel }, label),
      e(Text, { style: styles.patientValue }, value || '-'),
    );

  return e(Document, null,
    e(Page, { size: 'A4', style: styles.page },
      e(View, { style: styles.header },
        e(Text, { style: styles.title }, 'КОНСУЛЬТАЦИЯ'),
        e(Text, { style: styles.subtitle }, `Дата составления: ${formatDate(p.complaintDate)}`),
      ),
      e(View, { style: styles.patientBlock },
        row('ФИО пациента:', p.fullName),
        row('Возраст:', p.age),
        row('Пол:', p.gender),
        row('Дата обращения:', formatDate(p.complaintDate)),
      ),
      ra && e(View, { style: styles.patientBlock },
        e(Text, { style: { ...styles.sectionTitle, marginBottom: 8 } }, 'Оценка риска (шкала Морзе)'),
        row('Падал (3 мес.):', ra.fallInLast3Months || 'нет'),
        row('Головокружение:', ra.dizzinessOrWeakness || 'нет'),
        row('Сопровождение:', ra.needsEscort || 'нет'),
        row('Оценка боли:', `${ra.painScore || '0'}б`),
      ),
      ...sections.map((sec, idx) =>
        e(View, { key: idx, style: styles.section },
          e(Text, { style: styles.sectionTitle }, sec.title),
          sec.content.includes('\n')
            ? sec.content.split('\n').filter((l) => l.trim()).map((line, i) =>
                e(Text, { key: i, style: styles.sectionContent }, line))
            : e(Text, { style: styles.sectionContent }, sec.content),
        ),
      ),
      e(View, { style: styles.signatureLine },
        e(View, { style: styles.signatureBlock },
          e(View, { style: styles.signaturePlaceholder }),
          e(Text, { style: styles.signatureLabel }, 'Подпись врача'),
        ),
      ),
      e(View, { style: styles.footer, fixed: true },
        e(Text, { style: styles.footerText }, 'Документ сформирован автоматически'),
        e(Text, { style: styles.footerText }, formatDate(p.complaintDate)),
      ),
    ),
  );
}

async function main() {
  const input = process.argv[2];
  if (!input) throw new Error('Usage: qa-to-pdf.cjs <input.txt> [output.pdf]');
  const text = fs.readFileSync(path.resolve(input), 'utf-8').trim();

  const base = path.basename(input, path.extname(input));
  const output = process.argv[3] || path.resolve(path.dirname(input), `${base}.pdf`);

  console.log(`Input:  ${input} (${text.length} chars)`);
  console.log(`Output: ${output}`);

  const loginResp = await fetch(`${SERVER}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password: AUTH_PASS }),
  });
  const { token } = await loginResp.json();
  if (!token) throw new Error('Login failed');

  const start = Date.now();
  const resp = await fetch(`${SERVER}/api/structure`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!resp.ok) throw new Error(`Structure failed: ${resp.status} ${await resp.text()}`);
  const data = await resp.json();
  console.log(`Structured in ${((Date.now() - start) / 1000).toFixed(1)}s`);

  // Server возвращает patient/riskAssessment как строки JSON — парсим.
  const doc = data.document || {};
  if (typeof doc.patient === 'string') {
    try { doc.patient = JSON.parse(doc.patient); } catch { doc.patient = {}; }
  }
  if (typeof doc.riskAssessment === 'string') {
    try { doc.riskAssessment = JSON.parse(doc.riskAssessment); } catch { doc.riskAssessment = null; }
  }

  await ReactPDF.renderToFile(buildPdfTree(doc), output);
  console.log(`PDF written: ${output}`);
}

main().catch((e) => { console.error('Fatal:', e); process.exit(1); });
