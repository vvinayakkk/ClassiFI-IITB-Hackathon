import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, Server, Brain, Database } from 'lucide-react';

const CheckMarkIcon = () => (
  <motion.svg
    viewBox="0 0 24 24"
    className="w-5 h-5"
    initial="hidden"
    animate="visible"
  >
    <motion.path
      d="M4 12.6111L8.92308 17.5L20 6.5"
      fill="transparent"
      strokeWidth="2"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      variants={{
        hidden: { pathLength: 0 },
        visible: { 
          pathLength: 1,
          transition: { duration: 0.3, ease: "easeOut" }
        }
      }}
    />
  </motion.svg>
);

const DocumentProcessingLoader = ({ isLoading, modelResults, error }) => {
  const [currentStage, setCurrentStage] = useState(0);

  // Remove the timer-based effects and only use modelResults
  useEffect(() => {
    if (isLoading && modelResults?.modelResponses) {
      setCurrentStage(modelResults.modelResponses.length - 1);
    }
  }, [isLoading, modelResults]);

  const modelStages = [
    { 
      name: "Initial Parsing", 
      icon: <Cpu className="w-12 h-12 text-blue-500" />,
      color: "bg-blue-100",
      description: "Extracting document structure"
    },
    { 
      name: "Feature Extraction", 
      icon: <Server className="w-12 h-12 text-green-500" />,
      color: "bg-green-100",
      description: "Identifying key information"
    },
    { 
      name: "AI Classification", 
      icon: <Brain className="w-12 h-12 text-purple-500" />,
      color: "bg-purple-100",
      description: "Deep learning analysis"
    },
    { 
      name: "Final Validation", 
      icon: <Database className="w-12 h-12 text-red-500" />,
      color: "bg-red-100",
      description: "Cross-referencing results"
    }
  ];

  if (!isLoading) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/20 backdrop-blur-sm p-3">
      <motion.div 
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white shadow-xl rounded-xl p-4 w-full max-w-3xl max-h-[90vh] overflow-y-auto"
      >
        {/* Header */}
        <div className="text-center mb-4">
          <h2 className={`text-xl font-bold bg-gradient-to-r ${
            error ? 'from-red-600 to-red-500' : 'from-blue-600 to-indigo-600'
          } text-transparent bg-clip-text`}>
            {error ? 'Processing Error' : 'Processing Document'}
          </h2>
          <p className="text-sm text-gray-500">
            {error ? 'An error occurred during processing' : 'AI-powered analysis in progress'}
          </p>
        </div>

        {error ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-50 border border-red-200 rounded-lg p-4 text-center"
          >
            <div className="text-red-600 text-xl mb-2">⚠️</div>
            <p className="text-sm text-red-700 font-medium mb-2">{error}</p>
            <p className="text-xs text-red-500">
              Please try again or contact support if the issue persists.
            </p>
          </motion.div>
        ) : (
          <div className="grid md:grid-cols-2 gap-4">
            {/* Processing Stages */}
            <div className="space-y-2">
              {modelStages.map((stage, index) => (
                <motion.div
                  key={stage.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ 
                    opacity: index <= currentStage ? 1 : 0.5, 
                    x: 0,
                    scale: index === currentStage ? 1.02 : 1
                  }}
                  className={`flex items-center gap-3 p-2 rounded-lg ${stage.color} ${
                    index === currentStage ? 'ring-2 ring-blue-300' : ''
                  }`}
                >
                  <div className={`p-2 rounded-full ${stage.color}`}>
                    {React.cloneElement(stage.icon, {
                      className: `w-6 h-6 ${stage.icon.props.className.split(' ').pop()}`
                    })}
                  </div>
                  <div className="min-w-0 flex-1">
                    <h3 className="font-medium text-sm truncate">{stage.name}</h3>
                    <p className="text-xs text-gray-600">{stage.description}</p>
                  </div>
                  {index <= currentStage && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="relative"
                    >
                      <motion.div
                        className="absolute inset-0 bg-green-400 rounded-full"
                        animate={{
                          scale: [1, 1.5, 1],
                          opacity: [0.2, 0.1, 0]
                        }}
                        transition={{
                          duration: 1,
                          repeat: Infinity,
                          repeatType: "loop"
                        }}
                      />
                      <div className="relative z-10 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-white">
                        <CheckMarkIcon />
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              ))}
            </div>

            {/* Results Preview */}
            {modelResults?.modelResponses && (
              <div className="bg-gray-50 rounded-lg p-3 space-y-2">
                <h3 className="text-sm font-bold text-gray-800">Results</h3>
                <div className="space-y-2">
                  {modelResults.modelResponses.map((result, index) => (
                    <motion.div
                      key={`model${index + 1}`}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="bg-white rounded-lg p-2 shadow-sm text-sm"
                    >
                      <div className="flex justify-between items-center">
                        <span className="font-medium">Model {index + 1}</span>
                        <span className="text-xs text-blue-600">{result.confidence}%</span>
                      </div>
                      <p className="text-xs text-gray-600 mt-1">Category: {result.category}</p>
                    </motion.div>
                  ))}
                  
                  {modelResults.finalClassification && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="bg-blue-100 rounded-lg p-2 mt-2"
                    >
                      <p className="text-sm font-medium text-blue-800">{modelResults.finalClassification}</p>
                      <p className="text-xs text-blue-600 mt-1">
                        Overall Confidence: {modelResults.overallConfidence}%
                      </p>
                    </motion.div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default DocumentProcessingLoader;