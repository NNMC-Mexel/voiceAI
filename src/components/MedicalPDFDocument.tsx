import { Document, Page, Text, View, StyleSheet, Font } from '@react-pdf/renderer';
import type { MedicalDocument } from '../types';
import arialRegular from '../assets/fonts/arial.ttf';
import arialBold from '../assets/fonts/arialbd.ttf';

Font.register({
  family: 'ArialCyr',
  fonts: [
    { src: arialRegular, fontWeight: 'normal' },
    { src: arialBold, fontWeight: 'bold' },
  ],
});

const styles = StyleSheet.create({
  page: {
    padding: 40,
    fontFamily: 'ArialCyr',
    fontSize: 11,
    lineHeight: 1.5,
    color: '#1e293b',
  },
  header: {
    marginBottom: 24,
    paddingBottom: 16,
    borderBottomWidth: 2,
    borderBottomColor: '#339678',
    borderBottomStyle: 'solid',
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
    color: '#1e4d41',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 10,
    textAlign: 'center',
    color: '#64748b',
  },
  patientBlock: {
    backgroundColor: '#f0f9f6',
    padding: 12,
    marginBottom: 20,
    borderRadius: 4,
  },
  patientRow: {
    flexDirection: 'row',
    marginBottom: 4,
  },
  patientLabel: {
    width: 120,
    fontWeight: 'bold',
    color: '#21614f',
  },
  patientValue: {
    flex: 1,
  },
  section: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#267860',
    marginBottom: 6,
    paddingBottom: 4,
    borderBottomWidth: 1,
    borderBottomColor: '#d8f0e8',
    borderBottomStyle: 'solid',
  },
  sectionContent: {
    fontSize: 11,
    lineHeight: 1.6,
    textAlign: 'justify',
  },
  footer: {
    position: 'absolute',
    bottom: 30,
    left: 40,
    right: 40,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
    borderTopStyle: 'solid',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  footerText: {
    fontSize: 9,
    color: '#94a3b8',
  },
  signatureLine: {
    marginTop: 40,
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  signatureBlock: {
    width: 200,
  },
  signaturePlaceholder: {
    borderBottomWidth: 1,
    borderBottomColor: '#1e293b',
    borderBottomStyle: 'solid',
    marginBottom: 4,
    height: 30,
  },
  signatureLabel: {
    fontSize: 9,
    color: '#64748b',
    textAlign: 'center',
  },
});

interface MedicalPDFDocumentProps {
  document: MedicalDocument;
}

export function MedicalPDFDocument({ document }: MedicalPDFDocumentProps) {
  const formatDate = (dateStr: string) => {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    return date.toLocaleDateString('ru-RU', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
    });
  };

  const sections = [
    { title: 'Жалобы', content: document.complaints },
    { title: 'Анамнез заболевания', content: document.anamnesis },
    { title: 'Объективный статус', content: document.objectiveStatus },
    { title: 'Диагноз', content: document.diagnosis },
    { title: 'Клиническое течение', content: document.clinicalCourse },
    { title: 'Заключение', content: document.conclusion },
    { title: 'Рекомендации', content: document.recommendations },
    { title: 'Заметки врача', content: document.doctorNotes },
  ].filter((s) => s.content);

  return (
    <Document>
      <Page size="A4" style={styles.page}>
        <View style={styles.header}>
          <Text style={styles.title}>МЕДИЦИНСКИЙ ПРОТОКОЛ</Text>
          <Text style={styles.subtitle}>Дата составления: {formatDate(new Date().toISOString())}</Text>
        </View>

        <View style={styles.patientBlock}>
          <View style={styles.patientRow}>
            <Text style={styles.patientLabel}>ФИО пациента:</Text>
            <Text style={styles.patientValue}>{document.patient.fullName || '-'}</Text>
          </View>
          <View style={styles.patientRow}>
            <Text style={styles.patientLabel}>Возраст:</Text>
            <Text style={styles.patientValue}>{document.patient.age || '-'}</Text>
          </View>
          <View style={styles.patientRow}>
            <Text style={styles.patientLabel}>Пол:</Text>
            <Text style={styles.patientValue}>{document.patient.gender || '-'}</Text>
          </View>
          <View style={styles.patientRow}>
            <Text style={styles.patientLabel}>Дата обращения:</Text>
            <Text style={styles.patientValue}>{formatDate(document.patient.complaintDate) || '-'}</Text>
          </View>
        </View>

        {sections.map((section, index) => (
          <View key={index} style={styles.section}>
            <Text style={styles.sectionTitle}>{section.title}</Text>
            <Text style={styles.sectionContent}>{section.content}</Text>
          </View>
        ))}

        <View style={styles.signatureLine}>
          <View style={styles.signatureBlock}>
            <View style={styles.signaturePlaceholder} />
            <Text style={styles.signatureLabel}>Подпись врача</Text>
          </View>
        </View>

        <View style={styles.footer} fixed>
          <Text style={styles.footerText}>Документ сформирован автоматически</Text>
          <Text style={styles.footerText}>{formatDate(new Date().toISOString())}</Text>
        </View>
      </Page>
    </Document>
  );
}
