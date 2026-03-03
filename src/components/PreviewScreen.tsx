import { useState } from 'react';
import { PDFDownloadLink, PDFViewer, pdf } from '@react-pdf/renderer';
import { ArrowLeft, Download, Printer, Edit, FileText, Loader2 } from 'lucide-react';
import type { MedicalDocument } from '../types';
import { MedicalPDFDocument } from './MedicalPDFDocument';

interface PreviewScreenProps {
  document: MedicalDocument;
  onEdit: () => void;
  onNewDocument: () => void;
}

export function PreviewScreen({ document, onEdit, onNewDocument }: PreviewScreenProps) {
  const [isPrinting, setIsPrinting] = useState(false);

  const handlePrint = async () => {
    setIsPrinting(true);
    try {
      const instance = pdf(<MedicalPDFDocument document={document} />);
      const blob = await instance.toBlob();
      const blobUrl = URL.createObjectURL(blob);

      const win = window.open(blobUrl, '_blank');
      if (win) {
        win.addEventListener('load', () => {
          win.focus();
          win.print();
        });
      }

      setTimeout(() => URL.revokeObjectURL(blobUrl), 60000);
    } catch (error) {
      console.error('Print error:', error);
      alert('Не удалось подготовить документ к печати.');
    } finally {
      setIsPrinting(false);
    }
  };

  const generateFileName = () => {
    const date = new Date().toISOString().split('T')[0];
    const patientName = document.patient.fullName ? `_${document.patient.fullName.split(' ')[0]}` : '';
    return `Протокол${patientName}_${date}.pdf`;
  };

  return (
    <div className="min-h-screen bg-slate-100 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-6 slide-up">
          <div>
            <button
              onClick={onEdit}
              className="flex items-center gap-2 text-text-secondary hover:text-medical-600 transition-colors mb-2"
            >
              <ArrowLeft className="w-4 h-4" />
              <span className="text-sm">Вернуться к редактированию</span>
            </button>
            <h1 className="text-3xl font-display font-bold text-medical-900">Предпросмотр документа</h1>
          </div>

          <div className="flex items-center gap-3">
            <button onClick={onEdit} className="btn-secondary flex items-center gap-2">
              <Edit className="w-5 h-5" />
              Редактировать
            </button>

            <PDFDownloadLink document={<MedicalPDFDocument document={document} />} fileName={generateFileName()}>
              {({ loading }) => (
                <button disabled={loading} className="btn-secondary flex items-center gap-2">
                  {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Download className="w-5 h-5" />}
                  Скачать PDF
                </button>
              )}
            </PDFDownloadLink>

            <button onClick={handlePrint} disabled={isPrinting} className="btn-primary flex items-center gap-2">
              {isPrinting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Printer className="w-5 h-5" />}
              Печать
            </button>
          </div>
        </div>

        <div className="glass-card rounded-2xl overflow-hidden slide-up" style={{ animationDelay: '0.1s' }}>
          <div className="bg-medical-700 px-6 py-4 flex items-center gap-3">
            <FileText className="w-5 h-5 text-white" />
            <span className="text-white font-medium">{generateFileName()}</span>
          </div>

          <div className="p-6 bg-slate-200">
            <div className="mx-auto" style={{ maxWidth: '800px', height: '1000px' }}>
              <PDFViewer width="100%" height="100%" showToolbar={false} className="rounded-lg shadow-2xl">
                <MedicalPDFDocument document={document} />
              </PDFViewer>
            </div>
          </div>
        </div>

        <div className="mt-8 flex justify-center slide-up" style={{ animationDelay: '0.2s' }}>
          <button onClick={onNewDocument} className="btn-secondary flex items-center gap-2">
            <FileText className="w-5 h-5" />
            Создать новый документ
          </button>
        </div>
      </div>
    </div>
  );
}
