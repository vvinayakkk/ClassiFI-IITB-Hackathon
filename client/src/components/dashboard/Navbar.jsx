import { useState } from 'react';
import { Bell, Search, Settings, User, Menu, X } from 'lucide-react';
import { NOTIFICATIONS } from '@/constants/dashboardData';

const Navbar = ({ setActiveTab }) => {
  const [showNotifications, setShowNotifications] = useState(false);
  const [showMobileMenu, setShowMobileMenu] = useState(false);

  return (
    <nav className="bg-white shadow-lg px-4 sm:px-6 py-4 sticky top-0 z-50">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center justify-between w-full sm:w-auto">
            <h1 
              className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 text-transparent bg-clip-text cursor-pointer"
              onClick={() => setActiveTab('dashboard')}
            >
              ClassifyMe.ai
            </h1>
            <button 
              className="sm:hidden p-2 hover:bg-gray-100 rounded-lg"
              onClick={() => setShowMobileMenu(!showMobileMenu)}
            >
              {showMobileMenu ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
          <div className="hidden sm:block relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
            <input
              type="text"
              placeholder="Search resumes..."
              className="pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 w-64 bg-gray-50"
            />
          </div>
        </div>

        {/* Desktop Navigation */}
        <div className="hidden sm:flex items-center space-x-4">
          <div className="relative">
            <button 
              className="p-2 rounded-full hover:bg-gray-100 relative"
              onClick={() => setShowNotifications(!showNotifications)}
            >
              <Bell className="h-5 w-5 text-gray-600" />
              <span className="absolute top-0 right-0 h-2 w-2 bg-red-500 rounded-full"></span>
            </button>
            {showNotifications && (
              <div className="absolute right-0 mt-2 w-80 bg-white rounded-lg shadow-lg p-4 z-50">
                <h3 className="font-semibold mb-2">Notifications</h3>
                <div className="space-y-2">
                  {NOTIFICATIONS.map(notification => (
                    <div 
                      key={notification.id} 
                      className="p-2 hover:bg-gray-50 rounded-lg cursor-pointer transition-colors"
                    >
                      <p className="text-sm font-medium">{notification.message}</p>
                      <p className="text-xs text-gray-500">{notification.time}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          <button 
            className="p-2 rounded-full hover:bg-gray-100"
            onClick={() => setActiveTab('settings')}
          >
            <Settings className="h-5 w-5 text-gray-600" />
          </button>
          <button className="flex items-center space-x-2 bg-gray-100 rounded-full px-3 py-1 hover:bg-gray-200">
            <div className="h-8 w-8 rounded-full bg-gradient-to-r from-blue-500 to-indigo-500 flex items-center justify-center">
              <User className="h-5 w-5 text-white" />
            </div>
            <span className="text-sm font-medium"></span>
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {showMobileMenu && (
        <div className="sm:hidden mt-4 space-y-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
            <input
              type="text"
              placeholder="Search resumes..."
              className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-gray-50"
            />
          </div>
          <div className="flex items-center justify-between p-2 hover:bg-gray-50 rounded-lg">
            <span>Notifications</span>
            <Bell className="h-5 w-5 text-gray-600" />
          </div>
          <div className="flex items-center justify-between p-2 hover:bg-gray-50 rounded-lg" onClick={() => setActiveTab('settings')}>
            <span>Settings</span>
            <Settings className="h-5 w-5 text-gray-600" />
          </div>
          <div className="flex items-center justify-between p-2 hover:bg-gray-50 rounded-lg">
            <span>Profile</span>
            <User className="h-5 w-5 text-gray-600" />
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;