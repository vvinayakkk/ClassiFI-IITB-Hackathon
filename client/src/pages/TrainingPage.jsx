import { useState } from 'react';

const TrainingPage = () => {
  const [classes, setClasses] = useState([]);
  const [csvFile, setCSVFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [downloadCSVEnabled, setDownloadCSVEnabled] = useState(false);
  const [currentStage, setCurrentStage] = useState(-1);

  const stageClasses = [
    ["Technology", "Space", "Medical", "Sport", "Entertainment"],
    ["Historical", "Food", "Politics", "Business", "Graphics"],
    ["Technology", "Space", "Medical", "Sport", "Entertainment", "Historical", "Food", "Politics", "Business", "Graphics"],
    ["AI", "IoT", "Blockchain", "Astronomy", "Space Exploration", "Healthcare", "Pharmaceuticals", "Team Sports", "Individual Sports", "Movies", "Music", "Ancient History", "Modern History", "Culinary Arts", "Nutrition", "Government Policies", "Political Analysis", "Finance", "Corporate Strategies", "3D Design", "Visual Arts"]
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
        await new Promise((resolve) => setTimeout(resolve, 2000));
        setClasses(prev => [...prev, ...stageClasses[nextStage]]);
        setDownloadCSVEnabled(true);
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
            <div className="text-gray-400 text-sm mb-2">Dashboard / Training</div>
            <h1 className="text-4xl font-bold text-white mb-2">Train Your Model</h1>
            <p className="text-gray-400">Upload your dataset and start training the classification model</p>
          </div>

          <div className="max-w-2xl mx-auto">
            <div className="bg-gray-800/50 rounded-xl p-8 backdrop-blur-sm shadow-xl border border-gray-700/50">
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
                      <p className="text-xs text-gray-400">CSV files only</p>
                      {csvFile && (
                        <p className="mt-2 text-sm text-green-400">Selected: {csvFile.name}</p>
                      )}
                    </div>
                    <input
                      id="csv-file"
                      type="file"
                      accept=".csv"
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

              {isLoading && (
                <div className="mt-8 p-4 bg-blue-500/10 rounded-lg border border-blue-500/20">
                  <div className="flex items-center gap-4">
                    <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-400"></div>
                    <div>
                      <h3 className="text-blue-400 font-medium">Training in Progress</h3>
                      <p className="text-sm text-blue-300/70">This might take a few minutes...</p>
                    </div>
                  </div>
                </div>
              )}

              {downloadCSVEnabled && (
                <>
                  <div className="mt-8 p-4 bg-green-500/10 rounded-lg border border-green-500/20">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-green-400 font-medium">Training Complete!</h3>
                        <p className="text-sm text-green-300/70">Your model is ready to use</p>
                      </div>
                      <button className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-all hover:scale-[1.02]">
                        Download Results
                      </button>
                    </div>
                  </div>
                  <div className="mt-4 flex justify-center">
                    <button 
                      onClick={() => window.location.href = '/dashboard'} 
                      className="bg-gradient-to-r from-purple-500 to-purple-600 text-white px-6 py-3 rounded-lg font-medium shadow-lg hover:from-purple-600 hover:to-purple-700 transition-all hover:scale-[1.02]"
                    >
                      Proceed to Dashboard
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingPage;