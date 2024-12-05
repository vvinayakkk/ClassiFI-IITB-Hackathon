import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { FileText, ChevronRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const RecentUploads = ({ uploads, selectedUpload, onUploadSelect, uploadedFiles }) => {
  const navigate = useNavigate();

  const handleUploadClick = (upload) => {
    onUploadSelect(upload);
    if (uploadedFiles.has(upload.id)) {
      navigate(`/moreanalysis/${upload.id}`);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Uploads</CardTitle>
        <CardDescription>Latest documents processed by the system</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {uploads.map(upload => (
            <div 
              key={upload.id} 
              className={`
                flex items-center justify-between p-4 rounded-lg cursor-pointer
                transition-all duration-200
                ${selectedUpload?.id === upload.id ? 'bg-blue-50 ring-2 ring-blue-500' : 'bg-gray-50 hover:bg-gray-100'}
              `}
              onClick={() => handleUploadClick(upload)}
            >
              <div className="flex items-center space-x-4">
                <div className="h-10 w-10 rounded-lg bg-blue-100 flex items-center justify-center">
                  <FileText className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="font-medium">{upload.name}</p>
                  <p className="text-sm text-gray-500">{upload.status}</p>
                  {selectedUpload?.id === upload.id && (
                    <div className="mt-2 space-y-1">
                      <p className="text-sm"><span className="font-medium">Confidence:</span> {upload.confidence}%</p>
                      <p className="text-sm"><span className="font-medium">Key Skills:</span> {upload.skills.join(", ")}</p>
                    </div>
                  )}
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-500">{upload.time}</span>
                <ChevronRight className={`h-5 w-5 text-gray-400 transition-transform ${selectedUpload?.id === upload.id ? 'rotate-90' : ''}`} />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default RecentUploads;