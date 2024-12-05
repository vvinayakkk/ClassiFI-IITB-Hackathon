import { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { ArrowLeft, User, FileText, BookOpen, LineChart, Brain, Target, Languages, MessageSquare, Mail, Phone } from 'lucide-react';
import Section, { SubCard } from './components/Section';
import LoadingSpinner, { processingSteps } from './components/LoadingSpinner';
import { Tag, ContactItem, ProgressBar } from './components/UIComponents';

const SERVER_URL = import.meta.env.VITE_SERVER_URL;

const DetailedAnalysis = () => {
  const { uploadId } = useParams();
  const navigate = useNavigate();
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingStep, setLoadingStep] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    const stepInterval = loading ? setInterval(() => {
      setLoadingStep((prev) => {
        if (prev < processingSteps.length - 1) {
          return prev + 1;
        } else {
          clearInterval(stepInterval);
          return prev;
        }
      });
    }, 1500) : null;

    return () => {
      if (stepInterval) clearInterval(stepInterval);
    };
  }, [loading]);

  useEffect(() => {
    const performAnalysis = async () => {
      try {
        const file = window.uploadedFiles?.get(Number(uploadId));
        if (!file) {
          throw new Error('File not found');
        }

        const formData = new FormData();
        formData.append('file', file);

        const analysisResponse = await fetch(`${SERVER_URL}/api/analyze/`, {
          method: 'POST',
          body: formData
        });

        if (!analysisResponse.ok) {
          throw new Error('Analysis failed');
        }

        const analysisData = await analysisResponse.json();
        console.log(analysisData);
        
        setAnalysis(analysisData.analysis);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    performAnalysis();
  }, [uploadId]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6">
      {loading && <LoadingSpinner loadingStep={loadingStep} />}
      
      <div className="max-w-5xl mx-auto">
        <button 
          onClick={() => navigate('/dashboard')}
          className="flex items-center space-x-2 mb-6 text-gray-600 hover:text-gray-900 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          <span>Back to Dashboard</span>
        </button>

        {error && (
          <div className="bg-red-50 text-red-600 p-4 rounded-xl border border-red-100 shadow-md">
            {error}
          </div>
        )}

        {analysis && (
          <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <Section 
                title="Personal Information" 
                icon={<User className="w-6 h-6 text-blue-500" />}
                className="md:col-span-2"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <ContactItem icon={<User className="w-5 h-5 text-gray-500" />}>
                    {analysis?.personal_info?.full_name || 'N/A'}
                  </ContactItem>
                  <ContactItem icon={<Mail className="w-5 h-5 text-gray-500" />}>
                    {analysis?.personal_info?.email || 'N/A'}
                  </ContactItem>
                  <ContactItem icon={<Phone className="w-5 h-5 text-gray-500" />}>
                    {analysis?.personal_info?.phone || 'N/A'}
                  </ContactItem>
                  {analysis?.personal_info?.linkedin && (
                    <ContactItem icon={<Link className="w-5 h-5 text-gray-500" />}>
                      {analysis.personal_info.linkedin}
                    </ContactItem>
                  )}
                </div>
              </Section>

              <Section 
                title="Document Overview" 
                icon={<FileText className="w-6 h-6 text-blue-500" />}
                className="md:col-span-2"
              >
                <SubCard title="Document Details">
                  <div className="space-y-2">
                    <p className="text-gray-700">Type: {analysis?.document_overview?.type}</p>
                    <p className="text-gray-700">Purpose: {analysis?.document_overview?.purpose}</p>
                  </div>
                </SubCard>
                <SubCard title="Major Sections">
                  <div className="flex flex-wrap gap-2">
                    {analysis?.content_structure?.major_sections?.map((section, idx) => (
                      <Tag key={idx}>{section}</Tag>
                    ))}
                  </div>
                </SubCard>
                <SubCard title="Key Themes">
                  <div className="flex flex-wrap gap-2">
                    {analysis?.document_overview?.key_themes?.map((theme, idx) => (
                      <Tag key={idx}>{theme}</Tag>
                    ))}
                  </div>
                </SubCard>
              </Section>

              <Section 
                title="Linguistic Analysis" 
                icon={<Languages className="w-6 h-6 text-indigo-500" />}
              >
                <SubCard title="Readability">
                  <div className="flex justify-between mb-2">
                    <span>Score</span>
                    <span>{analysis?.linguistic_analysis?.readability_score}/100</span>
                  </div>
                  <ProgressBar value={analysis?.linguistic_analysis?.readability_score} />
                </SubCard>
                <SubCard title="Language Metrics">
                  <div className="space-y-2">
                    <p className="text-gray-700">Vocabulary: {analysis?.linguistic_analysis?.vocabulary_complexity}</p>
                    <p className="text-gray-700">Structure: {analysis?.linguistic_analysis?.sentence_structure}</p>
                  </div>
                </SubCard>
              </Section>

              <Section 
                title="Sentiment Analysis" 
                icon={<MessageSquare className="w-6 h-6 text-green-500" />}
              >
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <h3 className="font-semibold">Sentiment Score</h3>
                      <span>{(analysis?.sentiment_analysis?.sentiment_ratio * 100).toFixed(0)}%</span>
                    </div>
                    <ProgressBar value={analysis?.sentiment_analysis?.sentiment_ratio * 100} />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Emotional Undertones</h3>
                    <div className="flex flex-wrap gap-2">
                      {analysis?.sentiment_analysis?.emotional_undertones?.map((emotion, idx) => (
                        <Tag key={idx}>{emotion}</Tag>
                      ))}
                    </div>
                  </div>
                </div>
              </Section>

              <Section 
                title="Key Insights" 
                icon={<Brain className="w-6 h-6 text-purple-500" />}
              >
                <SubCard title="Top Points">
                  <ul className="list-disc list-inside space-y-2">
                    {analysis?.key_insights?.top_points?.map((point, idx) => (
                      <li key={idx} className="text-gray-700">{point}</li>
                    )) || <li className="text-gray-500">No insights available</li>}
                  </ul>
                </SubCard>
              </Section>

              <Section 
                title="Critical Takeaways" 
                icon={<Target className="w-6 h-6 text-yellow-500" />}
              >
                <ul className="list-disc list-inside space-y-2">
                  {analysis?.key_insights?.critical_takeaways?.map((takeaway, idx) => (
                    <li key={idx} className="text-gray-700">{takeaway}</li>
                  )) || <li className="text-gray-500">No takeaways available</li>}
                </ul>
              </Section>

              <Section 
                title="Potential Implications" 
                icon={<LineChart className="w-6 h-6 text-red-500" />}
              >
                <ul className="list-disc list-inside space-y-2">
                  {analysis?.key_insights?.potential_implications?.map((implication, idx) => (
                    <li key={idx} className="text-gray-700">{implication}</li>
                  )) || <li className="text-gray-500">No implications available</li>}
                </ul>
              </Section>

              <Section 
                title="Content Analysis" 
                icon={<BookOpen className="w-6 h-6 text-green-500" />}
              >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-semibold mb-2">Writing Style</h3>
                    <p className="text-gray-700">{analysis?.content_structure?.writing_style || 'Not available'}</p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Complexity</h3>
                    <p className="text-gray-700">{analysis?.content_structure?.complexity || 'Not available'}</p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Tone</h3>
                    <p className="text-gray-700">{analysis?.content_structure?.tone || 'Not available'}</p>
                  </div>
                </div>
              </Section>

              <Section 
                title="Potential Use Cases" 
                icon={<Target className="w-6 h-6 text-orange-500" />}
              >
                <div className="space-y-4">
                  <div>
                    <h3 className="font-semibold mb-2">Relevant Industries</h3>
                    <div className="flex flex-wrap gap-2">
                      {analysis?.potential_use_cases?.relevant_industries?.map((industry, idx) => (
                        <Tag key={idx}>{industry}</Tag>
                      )) || <span className="text-gray-500">No industries listed</span>}
                    </div>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Suggested Actions</h3>
                    <ul className="list-disc list-inside space-y-2">
                      {analysis?.potential_use_cases?.suggested_actions?.map((action, idx) => (
                        <li key={idx} className="text-gray-700">{action}</li>
                      )) || <li className="text-gray-500">No actions suggested</li>}
                    </ul>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Recommended Applications</h3>
                    <ul className="list-disc list-inside space-y-2">
                      {analysis?.potential_use_cases?.recommended_applications?.map((application, idx) => (
                        <li key={idx} className="text-gray-700">{application}</li>
                      )) || <li className="text-gray-500">No applications recommended</li>}
                    </ul>
                  </div>
                </div>
              </Section>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DetailedAnalysis;