import { useState } from 'react';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels';

ChartJS.register(ArcElement, Tooltip, Legend, ChartDataLabels);

const AnalyticsPage = () => {
    const [isLoading, setIsLoading] = useState(true);
    const rawData = [120, 120, 118, 118, 118, 118, 118, 117, 117, 116, 115, 115,
        115, 112, 111, 110, 107, 103, 102, 97, 96, 63, 36, 22];
    
    const total = rawData.reduce((acc, curr) => acc + curr, 0);
    
    const data = {
        labels: [
            'IT', 'Business Dev', 'Advocate', 'Chef', 'Engineering', 
            'Accountant', 'Finance', 'Fitness', 'Aviation', 'Sales', 
            'Banking', 'Healthcare', 'Consultant', 'Construction',
            'PR', 'HR', 'Designer', 'Arts', 'Teacher', 'Apparel',
            'Digital Media', 'Agriculture', 'Automobile', 'BPO'
        ],
        datasets: [{
            data: rawData,
            backgroundColor: [
                '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
                '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
                '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
                '#000000', '#ffffff', '#800080', '#e6beff',
            ],
            borderWidth: 2,
            borderColor: '#ffffff',
        }]
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            datalabels: {
                color: '#fff',
                font: {
                    weight: 'bold',
                    size: 12
                },
                formatter: (value) => {
                    const percentage = ((value / total) * 100).toFixed(1);
                    return `${percentage}%`;
                }
            },
            legend: {
                position: 'right',
                labels: {
                    padding: 20,
                    font: {
                        size: 12
                    }
                }
            },
            tooltip: {
                callbacks: {
                    label: (context) => {
                        const value = context.raw;
                        const percentage = ((value / total) * 100).toFixed(1);
                        return `${context.label}: ${value} (${percentage}%)`;
                    }
                }
            }
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            {/* Header Section */}
            <div className="max-w-7xl mx-auto text-center mb-12">
                <h1 className="text-4xl font-bold text-gray-900 mb-4">Analytics Dashboard</h1>
                <p className="text-lg text-gray-600">Comprehensive analysis of professional distribution and model performance</p>
            </div>

            {/* Model Performance Section */}
            <section className="max-w-7xl mx-auto mb-16">
                <h2 className="text-2xl font-semibold text-gray-800 mb-6 px-4">Model Performance Metrics</h2>
                <div className='relative'>
                    {isLoading && (
                        <div className="absolute inset-0 flex items-center justify-center bg-gray-50/80 backdrop-blur-sm rounded-xl z-10">
                            <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent"></div>
                        </div>
                    )}
                    <div className="bg-white p-4 rounded-xl shadow-lg">
                        <iframe 
                            title="Wandb Report"
                            src="https://wandb.ai/vvinayakkk-sardar-patel-institute-of-technology/huggingface/reports/Untitled-Report--VmlldzoxMDQwOTQ5MA?accessToken=jlnvsord7v1pnxz5efh4492zmhahs7u8rfofyvdpstiu7xjdfr5jayy7lsozefej"
                            onLoad={() => setIsLoading(false)}
                            className="w-full transition-all duration-300 rounded-lg"
                            style={{
                                height: 'calc(90vh - 200px)',
                                minHeight: '600px',
                                border: '2px solid #e5e7eb',
                            }}
                        />
                    </div>
                </div>
            </section>

            {/* Distribution Chart Section */}
            <section className="max-w-7xl mx-auto">
                <h2 className="text-2xl font-semibold text-gray-800 mb-6 px-4">Professional Distribution</h2>
                <div className='bg-white rounded-xl shadow-lg p-6'>
                    <div className='h-[600px] md:h-[700px] w-full'>
                        <Pie 
                            data={data} 
                            options={{
                                ...options,
                                plugins: {
                                    ...options.plugins,
                                    legend: {
                                        ...options.plugins.legend,
                                        position: window.innerWidth < 768 ? 'bottom' : 'right',
                                    }
                                }
                            }} 
                        />
                    </div>
                </div>
            </section>
        </div>
    );
};

export default AnalyticsPage;