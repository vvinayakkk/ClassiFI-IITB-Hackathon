
import React from 'react';
import { Loader2 } from 'lucide-react';

const processingSteps = [
  "Extracting document text",
  "Parsing document structure",
  "Identifying key information sections",
  "Generating comprehensive report"
];

const LoadingSpinner = ({ loadingStep }) => (
  <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
    <div className="animate-pulse mb-8">
      <Loader2 className="w-24 h-24 text-blue-500 animate-spin" />
    </div>
    <div className="text-center max-w-md">
      <h2 className="text-3xl font-bold text-gray-800 mb-6 tracking-tight">Analyzing Document</h2>
      <div className="text-xl text-gray-600 mb-6 min-h-[50px]">
        {processingSteps[loadingStep]}
      </div>
      <div className="flex justify-center space-x-2">
        {processingSteps.map((_, index) => (
          <div 
            key={index} 
            className={`h-2 rounded-full transition-all duration-300 ${
              index === loadingStep 
                ? 'bg-blue-600 w-6' 
                : 'bg-gray-300 w-2'
            }`}
          ></div>
        ))}
      </div>
    </div>
  </div>
);

export default LoadingSpinner;
export { processingSteps };