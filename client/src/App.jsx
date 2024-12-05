import { Routes, Route } from 'react-router-dom';
import Dashboard from "./pages/DashboardPage";
import LandingPage from "./pages/LandingPage";
import TrainingPage from './pages/TrainingPage';
import DetailedAnalysis from './pages/DetailedAnalysis/index';
import AnalyticsPage from './pages/AnalyticsPage';

function App() {
  return (
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/train" element={<TrainingPage />} />
        <Route path="/dashboard" element={<Dashboard />} />  
        <Route path="/analytics" element={<AnalyticsPage />} />  
        <Route path="/moreanalysis/:uploadId" element={<DetailedAnalysis />} />
      </Routes>
  )
}

export default App