import { Upload, FileText, BarChart, Settings } from 'lucide-react';

const QuickActions = ({ setActiveTab, handleUploadClick }) => {
  return (
    <div className="mb-6 sm:mb-8 px-4 sm:px-0">
      <h2 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4">Quick Actions</h2>
      <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        <button 
          onClick={handleUploadClick}
          className="bg-gradient-to-r from-blue-500 to-indigo-500 p-3 sm:p-4 rounded-lg shadow-md hover:shadow-lg transition-all duration-200 flex flex-col sm:flex-row items-center sm:space-x-3 text-white"
        >
          <div className="h-8 w-8 sm:h-10 sm:w-10 rounded-lg bg-white/20 flex items-center justify-center mb-2 sm:mb-0">
            <Upload className="h-4 w-4 sm:h-5 sm:w-5 text-white" />
          </div>
          <span className="text-sm sm:text-base font-medium text-center sm:text-left">Upload Document</span>
        </button>
        <button 
          onClick={() => setActiveTab('classifications')}
          className="bg-white p-3 sm:p-4 rounded-lg shadow-md hover:shadow-lg transition-all duration-200 flex flex-col sm:flex-row items-center sm:space-x-3 hover:bg-gray-50"
        >
          <div className="h-8 w-8 sm:h-10 sm:w-10 rounded-lg bg-green-100 flex items-center justify-center mb-2 sm:mb-0">
            <FileText className="h-4 w-4 sm:h-5 sm:w-5 text-green-600" />
          </div>
          <span className="text-sm sm:text-base font-medium text-center sm:text-left">Classifications</span>
        </button>
        <button 
          onClick={() => setActiveTab('analytics')}
          className="bg-white p-3 sm:p-4 rounded-lg shadow-md hover:shadow-lg transition-all duration-200 flex flex-col sm:flex-row items-center sm:space-x-3 hover:bg-gray-50"
        >
          <div className="h-8 w-8 sm:h-10 sm:w-10 rounded-lg bg-purple-100 flex items-center justify-center mb-2 sm:mb-0">
            <BarChart className="h-4 w-4 sm:h-5 sm:w-5 text-purple-600" />
          </div>
          <span className="text-sm sm:text-base font-medium text-center sm:text-left">Analytics</span>
        </button>
        <button 
          onClick={() => setActiveTab('settings')}
          className="bg-white p-3 sm:p-4 rounded-lg shadow-md hover:shadow-lg transition-all duration-200 flex flex-col sm:flex-row items-center sm:space-x-3 hover:bg-gray-50"
        >
          <div className="h-8 w-8 sm:h-10 sm:w-10 rounded-lg bg-orange-100 flex items-center justify-center mb-2 sm:mb-0">
            <Settings className="h-4 w-4 sm:h-5 sm:w-5 text-orange-600" />
          </div>
          <span className="text-sm sm:text-base font-medium text-center sm:text-left">Settings</span>
        </button>
      </div>
    </div>
  );
};

export default QuickActions;