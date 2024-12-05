import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import { Upload, FileText, BriefcaseIcon, Shield, Layers, Cpu, Server, Brain, Database, ArrowRight, FileUp, Search, CheckCircle, RefreshCw, FolderOpen, Link2 } from 'lucide-react';

const LandingPage = () => {
  const mountRef = useRef(null);
  const navigate = useNavigate();
  const [selectedStep, setSelectedStep] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    // Three.js background animation setup
    const currentMount = mountRef.current;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true });

    renderer.setSize(window.innerWidth, window.innerHeight);
    currentMount.appendChild(renderer.domElement);

    // Create animated particles
    const geometry = new THREE.BufferGeometry();
    const particles = 2000;
    const positions = new Float32Array(particles * 3);

    for (let i = 0; i < particles * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 10;
      positions[i + 1] = (Math.random() - 0.5) * 10;
      positions[i + 2] = (Math.random() - 0.5) * 10;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      size: 0.02,
      color: '#4F46E5'
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);
    camera.position.z = 5;

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      points.rotation.x += 0.001;
      points.rotation.y += 0.001;
      renderer.render(scene, camera);
    };

    animate();

    // Handle resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      currentMount.removeChild(renderer.domElement);
    };
  }, []);

  useEffect(() => {
    if (selectedStep > 0) {
      const targetProgress = (selectedStep / (steps.length - 1)) * 100;
      const duration = 1000; // 1 second animation
      const startTime = Date.now();
      const startProgress = progress;

      const animate = () => {
        const currentTime = Date.now();
        const elapsed = currentTime - startTime;
        
        if (elapsed < duration) {
          const newProgress = startProgress + ((targetProgress - startProgress) * elapsed) / duration;
          setProgress(newProgress);
          requestAnimationFrame(animate);
        } else {
          setProgress(targetProgress);
        }
      };

      requestAnimationFrame(animate);
    }
  }, [selectedStep]);

  const handleGetStarted = () => {
    navigate('/train');
  };

  const steps = [
    {
      title: "Upload",
      icon: <FileUp className="w-8 h-8 " />,
      color: "from-blue-600 to-blue-600",
      description: "Securely upload your documents through our intuitive drag-and-drop interface.",
      details: "Our system accepts various document formats including PDFs, Word documents, and plain text files.",
    },
    {
      title: "Parse",
      icon: <Search className="w-8 h-8" />,
      color: "from-purple-500 to-purple-600",
      description: "Advanced parsing algorithms extract relevant information from your documents.",
      details: "Our AI system analyzes document structure, content, and metadata for comprehensive understanding.",
    },
    {
      title: "Analyze",
      icon: <Brain className="w-8 h-8" />,
      color: "from-green-500 to-green-600",
      description: "Deep learning models process and analyze the extracted content.",
      details: "Multiple AI models work together to understand context, terminology, and document patterns.",
    },
    {
      title: "Refine",
      icon: <RefreshCw className="w-8 h-8" />,
      color: "from-red-500 to-red-600",
      description: "Results are refined through iterative processing and validation.",
      details: "Advanced algorithms continuously improve classification accuracy through machine learning.",
    },
    {
      title: "Classify",
      icon: <CheckCircle className="w-8 h-8" />,
      color: "from-yellow-500 to-yellow-600",
      description: "Documents are automatically categorized based on content analysis.",
      details: "Our classification system uses both pre-defined categories and dynamic learning for accuracy.",
    }
  ];

  return (
    <div className="relative min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 overflow-hidden">
      <div ref={mountRef} className="absolute inset-0 opacity-50" />

      <div className="relative z-10">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4 md:py-8">
          {/* Responsive Navigation */}
          <nav className="flex flex-col sm:flex-row justify-between items-center mb-8 md:mb-16">
            <div className="text-2xl sm:text-3xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent mb-4 sm:mb-0">
              ClassifyMe.ai
            </div>
            <div className="flex flex-wrap justify-center gap-4 sm:gap-8 text-sm sm:text-base">
              <a href="#features" className="text-gray-300 hover:text-white transition">Features</a>
              <a href="#how-it-works" className="text-gray-300 hover:text-white transition">How It Works</a>
            </div>
          </nav>

          {/* Responsive Header */}
          <header className="text-center mb-12 md:mb-16 px-4">
            <h1 className="text-3xl sm:text-4xl md:text-6xl font-bold text-white mb-4 leading-tight">
              Revolutionize Your Document Management with AI
            </h1>
            <p className="text-base sm:text-lg md:text-xl text-gray-300 mb-8 md:mb-12 leading-relaxed max-w-3xl mx-auto">
              Effortlessly classify, refine, and organize your documents with a self-improving AI system.
            </p>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleGetStarted}
              className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-6 sm:px-8 py-3 sm:py-4 rounded-lg text-base sm:text-lg font-semibold hover:from-blue-600 hover:to-indigo-700 transition-all duration-300 shadow-lg hover:shadow-xl w-full sm:w-auto"
            >
              Get Started Now
            </motion.button>
          </header>

          {/* Responsive Features Section */}
          <section id="features" className="mb-12 md:mb-16 px-4">
            <h2 className="text-2xl sm:text-3xl font-bold text-white text-center mb-8 md:mb-12">Key Features</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:ga  p-8">
              {[
                {
                  icon: <Cpu className="w-12 h-12 text-blue-400" />,
                  title: "AI-Powered Classification",
                  description: "Advanced AI analyzes your documents for precise classification."
                },
                {
                  icon: <Brain className="w-12 h-12 text-purple-400" />,
                  title: "Iterative Learning",
                  description: "Our AI improves with every document it processes."
                },
                {
                  icon: <Shield className="w-12 h-12 text-green-400" />,
                  title: "Secure Handling",
                  description: "Your documents are handled with the utmost security and privacy."
                },
                {
                  icon: <Layers className="w-12 h-12 text-yellow-400" />,
                  title: "Integration",
                  description: "Seamlessly integrate with your internal repositories."
                }
              ].map((feature, index) => (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.4 + index * 0.2 }}
                  className="bg-gray-800/50 backdrop-blur-sm p-8 rounded-xl hover:bg-gray-800/60 transition-all duration-300 border border-gray-700/50"
                >
                  <div className="mb-6">{feature.icon}</div>
                  <h3 className="text-xl font-semibold text-white mb-3">{feature.title}</h3>
                  <p className="text-gray-400 leading-relaxed">{feature.description}</p>
                </motion.div>
              ))}
            </div>
          </section>

          {/* Responsive How It Works Section */}
          <section id="how-it-works" className="mb-12 md:mb-24 px-4">
            <h2 className="text-2xl sm:text-4xl font-bold text-center mb-8 md:mb-16 bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
              How It Works
            </h2>
            <div className="relative max-w-7xl mx-auto">
              {/* Mobile Timeline */}
              <div className="md:hidden space-y-8">
                {steps.map((step, index) => (
                  <motion.div
                    key={step.title}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    className="flex items-start space-x-4"
                  >
                    <motion.button
                      onClick={() => setSelectedStep(index)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className={`flex-shrink-0 w-16 h-16 rounded-xl bg-gradient-to-r ${step.color} 
                        flex items-center justify-center
                        ${selectedStep === index ? 'ring-2 ring-white shadow-lg' : ''}`}
                    >
                      <div className="text-white opacity-100">{step.icon}</div>
                    </motion.button>
                    <div className="flex-grow">
                      <h3 className="text-lg font-semibold text-white mb-2">{step.title}</h3>
                      {selectedStep === index && (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="text-sm text-gray-300"
                        >
                          <p>{step.description}</p>
                        </motion.div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Desktop Timeline */}
              <div className="hidden md:block relative">
                {/* Progress Line - Updated for fewer steps */}
                <div className="absolute top-10 left-0 right-0 flex items-center">
                  <div className="w-full h-2 bg-gray-800/50 backdrop-blur-sm mx-10 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 transition-all duration-500"
                      style={{ 
                        width: `${progress}%`,
                        boxShadow: '0 0 20px rgba(99, 102, 241, 0.3)',
                        transition: 'width 0.8s cubic-bezier(0.4, 0, 0.2, 1)'
                      }}
                    />
                  </div>
                </div>

                {/* Loading percentage display - Updated styling */}
                <div className="absolute top-0 right-10 text-white font-mono bg-gray-800/50 px-3 py-1 rounded-full text-sm">
                  {Math.round(progress)}%
                </div>

                {/* Steps - Update grid columns for fewer steps */}
                <div className="relative z-10 grid grid-cols-5 gap-4 md:gap-8 pt-0">
                  {steps.map((step, index) => (
                    <motion.div
                      key={step.title}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.8, delay: 0.2 + index * 0.1 }}
                      className="flex flex-col items-center relative"
                    >
                      <motion.button
                        onClick={() => setSelectedStep(index)}
                        whileHover={{ scale: 1.05, rotate: [0, -5, 5, 0] }}
                        whileTap={{ scale: 0.95 }}
                        className={`relative w-20 h-20 rounded-2xl bg-gradient-to-r ${step.color} 
                          flex items-center justify-center mb-6 z-20
                          ${selectedStep === index 
                            ? 'ring-4 ring-white shadow-xl shadow-current/50' 
                            : 'hover:opacity-90'}`}
                        style={{
                          transformStyle: 'preserve-3d',
                          boxShadow: selectedStep === index 
                            ? '0 0 20px rgba(59, 130, 246, 0.5)' 
                            : '0 0 0 rgba(0,0,0,0)'
                        }}
                      >
                        <div className="text-white opacity-100">{step.icon}</div>
                      </motion.button>

                      <motion.div
                        initial={false}
                        animate={{
                          scale: selectedStep === index ? 1 : 0.95,
                          opacity: selectedStep === index ? 1 : 1,
                        }}
                        className="text-center group"
                      >
                        <h3 className={`text-xl font-semibold mb-3 transition-colors duration-300
                          ${selectedStep === index 
                            ? 'bg-gradient-to-r from-white via-blue-100 to-white bg-clip-text text-transparent' 
                            : 'text-gray-300'}`}
                        >
                          {step.title}
                        </h3>
                        {selectedStep === index && (
                          <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="space-y-3 backdrop-blur-sm bg-gray-900/50 p-4 rounded-xl border border-gray-700/50"
                          >
                            <p className="text-gray-200 leading-relaxed">{step.description}</p>
                          </motion.div>
                        )}
                      </motion.div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;