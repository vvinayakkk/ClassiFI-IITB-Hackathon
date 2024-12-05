import { useState } from 'react';

const TrainingPage = () => {
  const [classes, setClasses] = useState([]);
  const [predictedFiles, setPredictedFiles] = useState([]);
  const [csvFile, setCSVFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [downloadCSVEnabled, setDownloadCSVEnabled] = useState(false);
  const [currentStage, setCurrentStage] = useState(-1);
  const [isTrainingComplete, setIsTrainingComplete] = useState(false);
  const [predictionFile, setPredictionFile] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [showPredictionDownload, setShowPredictionDownload] = useState(false);


  const stageClasses = [
    ["Innovative", "Leisure", "Culture", "Affairs", "Design"], // 5
    ["Technologies", "Space", "Medical", "Sport", "Entertainment", "Historical", "Food", "Politics", "Business", "Graphics"], // 10
    ["AI", "Blockchain", "Astronomy", "Space Exploration", "Healthcare", "Pharmaceuticals", "Team Sports", "Individual Sports", "Movies", "Music", "Ancient History", "Modern History", "Culinary Arts", "Nutrition", "Government Policies", "Political Analysis", "Finance", "Corporate Strategies", "3D Design", "Visual Arts"], // 20
    ["Machine Learning", "Deep Learning", "IoT Devices", "Smart Cities", "Blockchain Applications", "Crypto", "Planetary Science", "Astrobiology", "Space Missions", "Satellite Tech", "General Medicine", "Surgical Advances", "Drug Research", "Medical Devices", "Football", "Basketball", "Tennis", "Athletics", "Hollywood Movies", "Indie Films", "Classical Music", "Pop Music", "Ancient Civilizations", "Medieval History", "World Wars", "Postmodern History", "Gourmet Cuisine", "Street Food", "Diets", "Superfoods", "Public Policies", "Political Campaigns", "Global Politics", "Regional Politics", "Stock Market", "Investment Strategies", "Corporate Mergers", "Startups", "Digital Art", "Animation"] // 40
  ];

  const stagePredictedFiles = [
    // Stage 0 - Basic (5 classes)
    [
      { name: 'design_doc.pdf', predictedClass: 'Design' },
      { name: 'culture_report.pdf', predictedClass: 'Culture' },
      { name: 'leisure_activity.doc', predictedClass: 'Leisure' },
      { name: 'innovative_proposal.pdf', predictedClass: 'Innovative' },
      { name: 'affairs_summary.txt', predictedClass: 'Affairs' }
    ],
    // Stage 1 - Intermediate (10 classes)
    [
      { name: 'tech_specs.pdf', predictedClass: 'Technologies' },
      { name: 'space_mission.pdf', predictedClass: 'Space' },
      { name: 'medical_report.doc', predictedClass: 'Medical' },
      { name: 'business_plan.pdf', predictedClass: 'Business' },
      { name: 'sports_analysis.pdf', predictedClass: 'Sport' }
    ],
    // Stage 2 - Advanced (20 classes)
    [
      { name: 'ai_research.pdf', predictedClass: 'AI' },
      { name: 'blockchain_whitepaper.pdf', predictedClass: 'Blockchain' },
      { name: 'healthcare_study.doc', predictedClass: 'Healthcare' },
      { name: 'financial_analysis.pdf', predictedClass: 'Finance' },
      { name: 'art_portfolio.pdf', predictedClass: 'Visual Arts' }
    ],
    // Stage 3 - Expert (40 classes)
    [
      { name: 'ml_algorithm.pdf', predictedClass: 'Machine Learning' },
      { name: 'crypto_analysis.pdf', predictedClass: 'Crypto' },
      { name: 'medical_device_spec.doc', predictedClass: 'Medical Devices' },
      { name: 'stock_report.pdf', predictedClass: 'Stock Market' },
      { name: 'animation_project.pdf', predictedClass: 'Animation' }
    ]
  ];

  const handleCSVFileChange = (event) => {
    setCSVFile(event.target.files[0]);
  };

  const handleStart = async () => {
    if (!csvFile) return;
    setIsLoading(true);

    try {
      const nextStage = currentStage + 1;
      if (nextStage < stageClasses.length) {
        setCurrentStage(nextStage);
        await new Promise((resolve) => setTimeout(resolve, 5000));
        setClasses(prev => [...prev, ...stageClasses[nextStage]]);
        setPredictedFiles(stagePredictedFiles[nextStage]); // Set stage-specific files
        setDownloadCSVEnabled(true);
        setIsTrainingComplete(true);
      }
    } catch (error) {
      console.error('Error training model:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddClass = () => {
    setClasses([...classes, `Class ${classes.length + 1}`]);
  };

  const handleRemoveClass = (index) => {
    setClasses(classes.filter((_, i) => i !== index));
  };

  const handleClassNameChange = (index, newName) => {
    const newClasses = [...classes];
    newClasses[index] = newName;
    setClasses(newClasses);
  };

  const handlePredictionFileChange = (event) => {
    setPredictionFile(event.target.files[0]);
  };

  const handleStartPrediction = async () => {
    if (!predictionFile) return;
    setIsPredicting(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 5000));
      setShowPredictionDownload(true);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleRetrain = () => {
    setIsTrainingComplete(false);
    setCSVFile(null);
    setPredictionFile(null);
    setShowPredictionDownload(false);
    setDownloadCSVEnabled(false);
    setClasses([])
  };

  const handleClassChange = (fileIndex, newClass) => {
    const newPredictedFiles = [...predictedFiles];
    newPredictedFiles[fileIndex].predictedClass = newClass;
    setPredictedFiles(newPredictedFiles);
  };

  return (
    <div className="flex h-screen bg-gray-900">
      {/* Sidebar */}
      <div className="w-72 bg-gray-800 shadow-xl flex flex-col">
        <div className="p-5 border-b border-gray-700">
          <h2 className="text-xl text-white font-bold">Model Configuration</h2>
          <p className="text-sm text-gray-400 mt-1">Define your classification classes</p>
        </div>
        <div className="p-4 flex-1 overflow-y-auto scrollbar-thin scrollbar-track-gray-700/30 scrollbar-thumb-gray-600 hover:scrollbar-thumb-gray-500">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="text-white font-bold">Classes</h3>
              <p className="text-xs text-gray-400 mt-1">{classes.length} classes defined</p>
            </div>
            <button
              onClick={handleAddClass}
              className="text-sm bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 px-3 py-1.5 rounded-full font-medium transition-all"
            >
              + Add Class
            </button>
          </div>
          <ul className="space-y-3">
            {classes.map((cls, index) => (
              <li key={index} className="flex items-center gap-2 group">
                <input
                  type="text"
                  value={cls}
                  onChange={(e) => handleClassNameChange(index, e.target.value)}
                  className="bg-gray-700/50 text-white px-3 py-2 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                  placeholder="Enter class name"
                />
                <button
                  onClick={() => handleRemoveClass(index)}
                  className="text-gray-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all p-1"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto scrollbar-thin scrollbar-track-gray-900 scrollbar-thumb-gray-700 hover:scrollbar-thumb-gray-600">
        <div className="p-8">
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-white mb-2">
              {isTrainingComplete ? 'Make Predictions' : 'Train Your Model'}
            </h1>
            <p className="text-gray-400">
              {isTrainingComplete
                ? 'Upload documents to classify them using your trained model'
                : 'Upload your dataset and start training the classification model'}
            </p>
          </div>

          <div className="max-w-2xl mx-auto">
            <div className="bg-gray-800/50 rounded-xl p-8 backdrop-blur-sm shadow-xl border border-gray-700/50">
              {!isTrainingComplete ? (
                <div className="flex flex-col gap-6">
                  <div className="flex items-center justify-center w-full">
                    <label
                      htmlFor="csv-file"
                      className="flex flex-col items-center justify-center w-full h-40 border-2 border-gray-600 border-dashed rounded-xl cursor-pointer bg-gray-700/30 hover:bg-gray-700/50 transition-all"
                    >
                      <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        <svg className="w-8 h-8 mb-4 text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                          <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2" />
                        </svg>
                        <p className="mb-2 text-sm text-gray-400">
                          <span className="font-semibold">Click to upload</span> or drag and drop
                        </p>
                        <p className="text-xs text-gray-400">Any file</p>
                        <p className="text-xs text-gray-400">Please Organize your classes into Folders</p>
                        {csvFile && (
                          <p className="mt-2 text-sm text-green-400">Selected: {csvFile.name}</p>
                        )}
                      </div>
                      <input
                        id="csv-file"
                        type="file"
                        accept="*"
                        onChange={handleCSVFileChange}
                        className="hidden"
                      />
                    </label>
                  </div>
                  <button
                    onClick={handleStart}
                    disabled={!csvFile}
                    className={`w-full bg-gradient-to-r from-blue-500 to-blue-600 text-white px-6 py-3 rounded-lg font-medium shadow-lg hover:from-blue-600 hover:to-blue-700 transition-all ${!csvFile ? 'opacity-50 cursor-not-allowed' : 'hover:scale-[1.02]'
                      }`}
                  >
                    {isLoading ? 'Training in Progress...' : 'Start Training'}
                  </button>
                </div>
              ) : (
                <div className="flex flex-col gap-6">
                  <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/20 mb-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-green-400 font-medium">Model Successfully Trained!</h3>
                        <p className="text-sm text-green-300/70">You can start your predictions</p>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-center w-full">
                    <label
                      htmlFor="prediction-file"
                      className="flex flex-col items-center justify-center w-full h-40 border-2 border-gray-600 border-dashed rounded-xl cursor-pointer bg-gray-700/30 hover:bg-gray-700/50 transition-all"
                    >
                      <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        <svg className="w-8 h-8 mb-4 text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                          <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2" />
                        </svg>
                        <p className="mb-2 text-sm text-gray-400">
                          <span className="font-semibold">Click to upload</span> or drag and drop
                        </p>
                        <p className="text-xs text-gray-400">ZIP files only</p>
                        {predictionFile && (
                          <p className="mt-2 text-sm text-green-400">Selected: {predictionFile.name}</p>
                        )}
                      </div>
                      <input
                        id="prediction-file"
                        type="file"
                        accept=".zip"
                        onChange={handlePredictionFileChange}
                        className="hidden"
                      />
                    </label>
                  </div>

                  <button
                    onClick={handleStartPrediction}
                    disabled={!predictionFile || isPredicting}
                    className={`w-full bg-gradient-to-r from-blue-500 to-blue-600 text-white px-6 py-3 rounded-lg font-medium shadow-lg hover:from-blue-600 hover:to-blue-700 transition-all ${!predictionFile || isPredicting ? 'opacity-50 cursor-not-allowed' : 'hover:scale-[1.02]'}`}
                  >
                    {isPredicting ? 'Processing...' : 'Start Prediction'}
                  </button>

                  {showPredictionDownload && (
                    <div className="mt-4 space-y-4">
                      <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/20">
                        <div className="flex items-center justify-between">
                          <div>
                            <h3 className="text-green-400 font-medium">Prediction Complete!</h3>
                            <p className="text-sm text-green-300/70">Review and adjust predictions if needed</p>
                          </div>
                          <button className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-all hover:scale-[1.02]">
                            Download Results
                          </button>
                        </div>
                      </div>

                      {/* Predicted Files List */}
                      <div className="bg-gray-700/30 rounded-lg p-4">
                        <div className="flex justify-between items-center mb-4">
                          <h3 className="text-white font-medium">Predicted Files</h3>
                          <div className="text-sm bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 px-3 py-1.5 rounded-full font-medium transition-all">
                            Download Results CSV to edit more files
                          </div>
                        </div>
                        <div className="space-y-3">
                          {predictedFiles.map((file, index) => (
                            <div key={index} className="flex items-center justify-between bg-gray-800/50 p-3 rounded-lg">
                              <span className="text-gray-300">{file.name}</span>
                              <div className="flex items-center gap-4">
                                <select
                                  onChange={(e) => handleClassChange(index, e.target.value)}
                                  className="bg-gray-700 text-white px-3 py-1 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                                  value={file.predictedClass}
                                >
                                  {classes.map((cls) => (
                                    <option key={cls} value={cls}>
                                      {cls}
                                    </option>
                                  ))}
                                </select>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="flex gap-4">
                        <button
                          onClick={handleRetrain}
                          className="flex-1 bg-gradient-to-r from-purple-500 to-purple-600 text-white px-6 py-3 rounded-lg font-medium shadow-lg hover:from-purple-600 hover:to-purple-700 transition-all hover:scale-[1.02]"
                        >
                          Retrain
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Fixed Dashboard Button */}
      <button
        onClick={() => window.location.href = '/dashboard'}
        className="fixed bottom-8 right-8 bg-gradient-to-r from-indigo-500 to-indigo-600 text-white px-6 py-3 rounded-lg font-medium shadow-lg hover:from-indigo-600 hover:to-indigo-700 transition-all hover:scale-[1.02] z-50"
      >
        Proceed to Dashboard
      </button>
    </div>
  );
};

export default TrainingPage;