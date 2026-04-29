import { useState } from 'react';
import { PDFDownloadLink, PDFViewer, pdf } from '@react-pdf/renderer';
import { ArrowLeft, Download, Printer, Edit, FileText, Loader2, Mic } from 'lucide-react';
import type { MedicalDocument } from '../types';
import { MedicalPDFDocument } from './MedicalPDFDocument';

interface PreviewScreenProps {
  document: MedicalDocument;
  audioBlob: Blob | null;
  onEdit: () => void;
  onNewDocument: () => void;
}

export function PreviewScreen({ document, audioBlob, onEdit, onNewDocument }: PreviewScreenProps) {
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

  const handleDownloadAudio = () => {
    if (!audioBlob) return;
    const date = new Date().toISOString().split('T')[0];
    const patientName = document.patient.fullName ? `_${document.patient.fullName.split(' ')[0]}` : '';
    const type = audioBlob.type.toLowerCase();
    const ext = type.includes('mp4') ? 'mp4' : type.includes('ogg') ? 'ogg' : type.includes('wav') ? 'wav' : 'webm';
    const filename = `Аудио${patientName}_${date}.${ext}`;

    const url = URL.createObjectURL(audioBlob);
    const a = window.document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  };

  return (
    <div className="min-h-screen bg-slate-100 py-6 px-3 sm:py-8 sm:px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header — на мобильном кнопки переезжают под заголовок и оборачиваются */}
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between mb-6 slide-up">
          <div className="min-w-0">
            <button
              onClick={onEdit}
              className="flex items-center gap-2 text-text-secondary hover:text-medical-600 transition-colors mb-2"
            >
              <ArrowLeft className="w-4 h-4" />
              <span className="text-sm">Вернуться к редактированию</span>
            </button>
            <h1 className="text-2xl sm:text-3xl font-display font-bold text-medical-900">
              Предпросмотр документа
            </h1>
          </div>

          {/* Action row: wrap на мобильном, кнопки 50% по две в ряд */}
          <div className="flex flex-wrap gap-2 sm:gap-3 sm:flex-nowrap">
            <button
              onClick={onEdit}
              className="btn-secondary flex items-center justify-center gap-2 flex-1 sm:flex-initial min-w-[140px] px-4 sm:px-6"
            >
              <Edit className="w-5 h-5" />
              <span className="whitespace-nowrap">Редактировать</span>
            </button>

            <PDFDownloadLink document={<MedicalPDFDocument document={document} />} fileName={generateFileName()}>
              {({ loading }) => (
                <button
                  disabled={loading}
                  className="btn-secondary flex items-center justify-center gap-2 flex-1 sm:flex-initial min-w-[140px] px-4 sm:px-6"
                >
                  {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Download className="w-5 h-5" />}
                  <span className="whitespace-nowrap">Скачать PDF</span>
                </button>
              )}
            </PDFDownloadLink>

            {audioBlob && (
              <button
                onClick={handleDownloadAudio}
                className="btn-secondary flex items-center justify-center gap-2 flex-1 sm:flex-initial min-w-[140px] px-4 sm:px-6"
              >
                <Mic className="w-5 h-5" />
                <span className="whitespace-nowrap">Скачать аудио</span>
              </button>
            )}

            <button
              onClick={handlePrint}
              disabled={isPrinting}
              className="btn-primary flex items-center justify-center gap-2 flex-1 sm:flex-initial min-w-[120px] px-4 sm:px-6"
            >
              {isPrinting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Printer className="w-5 h-5" />}
              <span className="whitespace-nowrap">Печать</span>
            </button>
          </div>
        </div>

        <div className="glass-card rounded-2xl overflow-hidden slide-up" style={{ animationDelay: '0.1s' }}>
          <div className="bg-medical-700 px-4 py-3 sm:px-6 sm:py-4 flex items-center gap-3">
            <FileText className="w-5 h-5 text-white shrink-0" />
            <span className="text-white font-medium truncate">{generateFileName()}</span>
          </div>

          {/* PDF preview: высота адаптивна.
              Мобильный: 70vh (короче, чтобы кнопки внизу оставались видны).
              Desktop: фиксированные 1000px по бывшей модели — но не больше 80vh. */}
          <div className="p-3 sm:p-6 bg-slate-200">
            <div
              className="mx-auto w-full"
              style={{ maxWidth: '800px', height: 'min(80vh, 1000px)' }}
            >
              <PDFViewer width="100%" height="100%" showToolbar={false} className="rounded-lg shadow-2xl">
                <MedicalPDFDocument document={document} />
              </PDFViewer>
            </div>
          </div>
        </div>

        <div className="mt-6 sm:mt-8 flex justify-center slide-up" style={{ animationDelay: '0.2s' }}>
          <button
            onClick={onNewDocument}
            className="btn-secondary flex items-center justify-center gap-2 w-full sm:w-auto"
          >
            <FileText className="w-5 h-5" />
            <span className="whitespace-nowrap">Создать новый документ</span>
          </button>
        </div>
      </div>
    </div>
  );
}
