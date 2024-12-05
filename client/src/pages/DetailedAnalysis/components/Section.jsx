import React from 'react';

export const SubCard = ({ title, children }) => (
  <div className="bg-gray-50/80 rounded-xl p-4 backdrop-blur-sm">
    <h3 className="font-semibold mb-3 text-gray-800">{title}</h3>
    {children}
  </div>
);

const Section = ({ title, icon, children, className = "" }) => (
  <div className={`bg-white/95 backdrop-blur-xl rounded-3xl shadow-2xl p-6 mb-6 border border-gray-200/50 hover:shadow-3xl transition-all duration-300 ${className}`}>
    <div className="flex items-center mb-6 space-x-4">
      <div className="p-3 rounded-xl bg-gradient-to-br from-gray-50 to-gray-100 shadow-md">{icon}</div>
      <h2 className="text-2xl font-semibold text-gray-800 tracking-tight">{title}</h2>
    </div>
    <div className="grid grid-cols-1 gap-4">
      {children}
    </div>
  </div>
);

export default Section;