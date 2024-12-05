export const NOTIFICATIONS = [
  { id: 1, message: "New document classification available", time: "2 min ago", type: "success" },
  { id: 2, message: "Document processed successfully", time: "1 hour ago", type: "info" },
  { id: 3, message: "System update scheduled", time: "2 hours ago", type: "warning" },
  { id: 4, message: "New feature: Batch processing", time: "1 day ago", type: "info" },
];

export const RECENT_UPLOADS = [
  {
    id: 1,
    name: "quantum_computing_research.pdf",
    status: "Classified as: Technologies",
    time: "Just now",
    confidence: 98.2,
    keywords: ["Quantum", "Computing", "Research"],
  },
  {
    id: 2,
    name: "deep_learning_study.pdf",
    status: "Classified as: AI",
    time: "5 min ago",
    confidence: 96.7,
    keywords: ["Neural Networks", "Deep Learning", "AI Models"],
  },
  {
    id: 3,
    name: "space_mission_report.pdf",
    status: "Classified as: Space-Exploration",
    time: "1 hour ago",
    confidence: 97.1,
    keywords: ["Space Mission", "Satellites", "Exploration"],
  },
];

export const STATS_CARDS = [
  {
    title: "Total Documents",
    icon: "Upload",
    value: "1,234",
    change: { value: "12%", type: "increase" },
    color: "text-blue-600",
  },
  {
    title: "Active Users",
    icon: "Users",
    value: "892",
    change: { value: "8%", type: "increase" },
    color: "text-green-600",
  },
  {
    title: "Categories",
    icon: "Briefcase",
    value: "40",
    change: { value: "+20", type: "new" },
    color: "text-purple-600",
  },
  {
    title: "Accuracy Rate",
    icon: "PieChart",
    value: "98.5%",
    change: { value: "2.3%", type: "increase" },
    color: "text-orange-600",
  },
];

export const PROCESSING_RESULTS = {
  model1: {
    confidence: 92.3,
    classification: "Technical Documentation",
    keywords: ["JavaScript", "React", "Node.js"],
  },
  model2: {
    confidence: 94.1,
    classification: "Full Stack Developer",
    skills: ["Python", "Docker", "AWS"],
  },
  model3: {
    confidence: 93.7,
    classification: "DevOps Engineer",
    skills: ["Kubernetes", "CI/CD", "Cloud Computing"],
  },
  model4: {
    confidence: 95.5,
    classification: "Cloud Architect",
    skills: ["Cloud Strategy", "Enterprise Architecture", "Security"],
  },
  overallConfidence: 94.4,
  finalClassification: "Technical Documentation",
};

export const CLASSIFICATION_DATA = [
  { category: "Technologies", count: 150, icon: "💻" },
  { category: "AI", count: 145, icon: "🤖" },
  { category: "Healthcare", count: 140, icon: "🏥" },
  { category: "Space-Exploration", count: 135, icon: "🚀" },
  { category: "Business", count: 130, icon: "💼" },
  { category: "Entertainment", count: 128, icon: "🎬" },
  { category: "Sports", count: 125, icon: "⚽" },
  { category: "History", count: 120, icon: "📜" },
  { category: "Food", count: 118, icon: "🍳" },
  { category: "Politics", count: 115, icon: "🏛️" },
  { category: "Graphics", count: 112, icon: "🎨" },
  { category: "Blockchain", count: 110, icon: "🔗" },
  { category: "IoT", count: 108, icon: "📱" },
  { category: "Astronomy", count: 105, icon: "🔭" },
  { category: "Music", count: 102, icon: "🎵" }
];

export const CATEGORY_DETAILS = {
  "Technologies": {
    totalPDFs: 150,
    averageConfidence: 96.5,
    commonKeywords: ["Innovation", "Digital", "Software"],
    documentTypes: "8-12 types",
    description: "Documents covering various technological advances and implementations.",
    typicalContent: ["Technical specifications", "Research papers", "Implementation guides"]
  },
  "Space": {
    totalPDFs: 135,
    averageConfidence: 94.8,
    commonKeywords: ["Space Missions", "Satellites", "Exploration"],
    documentTypes: "6-8 types",
    description: "Documentation related to space exploration and research.",
    typicalContent: ["Mission reports", "Research findings", "Space technology"]
  },
  "Medical": {
    totalPDFs: 140,
    averageConfidence: 95.2,
    commonKeywords: ["Healthcare", "Treatment", "Clinical"],
    documentTypes: "7-9 types",
    description: "Medical research and healthcare documentation.",
    typicalContent: ["Clinical studies", "Medical protocols", "Treatment guidelines"]
  },
  "Sport": {
    totalPDFs: 125,
    averageConfidence: 93.5,
    commonKeywords: ["Athletics", "Competition", "Training"],
    documentTypes: "5-7 types",
    description: "Sports-related documentation and analysis.",
    typicalContent: ["Training programs", "Competition reports", "Sports science"]
  },
  "Entertainment": {
    totalPDFs: 128,
    averageConfidence: 92.8,
    commonKeywords: ["Media", "Performance", "Production"],
    documentTypes: "6-8 types",
    description: "Entertainment industry documentation.",
    typicalContent: ["Production scripts", "Media analysis", "Industry reports"]
  },
  "Historical": {
    totalPDFs: 120,
    averageConfidence: 91.5,
    commonKeywords: ["History", "Archives", "Documentation"],
    documentTypes: "5-7 types",
    description: "Historical documents and research papers.",
    typicalContent: ["Historical records", "Research papers", "Archive materials"]
  },
  "Food": {
    totalPDFs: 118,
    averageConfidence: 93.2,
    commonKeywords: ["Cuisine", "Nutrition", "Recipes"],
    documentTypes: "4-6 types",
    description: "Food and culinary documentation.",
    typicalContent: ["Recipe collections", "Nutritional guides", "Food research"]
  },
  "Politics": {
    totalPDFs: 115,
    averageConfidence: 90.8,
    commonKeywords: ["Government", "Policy", "Analysis"],
    documentTypes: "6-8 types",
    description: "Political analysis and policy documentation.",
    typicalContent: ["Policy papers", "Political analysis", "Government documents"]
  },
  "Business": {
    totalPDFs: 130,
    averageConfidence: 94.5,
    commonKeywords: ["Strategy", "Management", "Finance"],
    documentTypes: "7-9 types",
    description: "Business and corporate documentation.",
    typicalContent: ["Business plans", "Corporate strategies", "Market analysis"]
  },
  "Graphics": {
    totalPDFs: 112,
    averageConfidence: 92.7,
    commonKeywords: ["Design", "Visual", "Creative"],
    documentTypes: "5-7 types",
    description: "Graphic design and visual arts documentation.",
    typicalContent: ["Design specs", "Visual guidelines", "Creative briefs"]
  },
  "AI": {
    totalPDFs: 145,
    averageConfidence: 95.8,
    commonKeywords: ["Machine Learning", "Neural Networks", "Deep Learning"],
    documentTypes: "6-8 types",
    description: "Artificial Intelligence documentation and research.",
    typicalContent: ["AI research", "ML models", "Algorithm documentation"]
  },
  "IoT": {
    totalPDFs: 108,
    averageConfidence: 93.4,
    commonKeywords: ["Connected Devices", "Sensors", "Smart Systems"],
    documentTypes: "5-7 types",
    description: "Internet of Things technology documentation.",
    typicalContent: ["IoT specifications", "Device protocols", "System architecture"]
  },
  "Blockchain": {
    totalPDFs: 110,
    averageConfidence: 94.1,
    commonKeywords: ["Cryptocurrency", "DLT", "Smart Contracts"],
    documentTypes: "5-7 types",
    description: "Blockchain technology and cryptocurrency documentation.",
    typicalContent: ["Technical whitepapers", "Protocol documentation", "Implementation guides"]
  },
  "Astronomy": {
    totalPDFs: 105,
    averageConfidence: 92.9,
    commonKeywords: ["Celestial", "Observatory", "Stars"],
    documentTypes: "4-6 types",
    description: "Astronomical research and observation documentation.",
    typicalContent: ["Research papers", "Observation data", "Astronomical studies"]
  },
  "Music": {
    totalPDFs: 102,
    averageConfidence: 91.2,
    commonKeywords: ["Composition", "Performance", "Theory"],
    documentTypes: "4-6 types",
    description: "Musical documentation and analysis.",
    typicalContent: ["Sheet music", "Music theory", "Performance analysis"]
  }
};
