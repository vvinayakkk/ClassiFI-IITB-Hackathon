
import React from 'react';

export const Tag = ({ children }) => (
  <span className="inline-block px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-700 m-1">
    {children}
  </span>
);

export const ContactItem = ({ icon, children }) => (
  <div className="flex items-center space-x-3 text-gray-700">
    {icon}
    <span className="text-base">{children}</span>
  </div>
);

export const ProgressBar = ({ value, max = 100 }) => (
  <div className="w-full bg-gray-200 rounded-full h-2.5">
    <div 
      className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
      style={{ width: `${(value/max) * 100}%` }}
    ></div>
  </div>
);