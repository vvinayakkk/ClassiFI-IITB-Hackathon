export const NOTIFICATIONS = [
  { id: 1, message: "New document classification available", time: "2 min ago", type: "success" },
  { id: 2, message: "Document processed successfully", time: "1 hour ago", type: "info" },
  { id: 3, message: "System update scheduled", time: "2 hours ago", type: "warning" },
  { id: 4, message: "New feature: Batch processing", time: "1 day ago", type: "info" },
];

export const RECENT_UPLOADS = [
  {
    id: 1,
    name: "senior_dev_document.pdf",
    status: "Classified as: Java Developer",
    time: "Just now",
    confidence: 98.2,
    keywords: ["Java", "Spring Boot", "Microservices"],
  },
  {
    id: 2,
    name: "project_management_doc.pdf",
    status: "Classified as: Project Manager",
    time: "5 min ago",
    confidence: 95.7,
    keywords: ["Agile", "Scrum", "Team Leadership"],
  },
  {
    id: 3,
    name: "hr_policy_doc.pdf",
    status: "Classified as: HR Specialist",
    time: "1 hour ago",
    confidence: 97.1,
    keywords: ["Recruitment", "Employee Relations", "HR Policy"],
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
    title: "Job Categories",
    icon: "Briefcase",
    value: "15",
    change: { value: "+2", type: "new" },
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
  { category: "BUSINESS-DEVELOPMENT", count: 120, icon: "💼" },
  { category: "INFORMATION-TECHNOLOGY", count: 120, icon: "💻" },
  { category: "CHEF", count: 118, icon: "👨‍🍳" },
  { category: "ACCOUNTANT", count: 118, icon: "📊" },
  { category: "ENGINEERING", count: 118, icon: "🔧" },
  { category: "ADVOCATE", count: 118, icon: "⚖️" },
  { category: "FINANCE", count: 118, icon: "💰" },
  { category: "FITNESS", count: 117, icon: "💪" },
  { category: "AVIATION", count: 117, icon: "✈️" },
  { category: "SALES", count: 116, icon: "🤝" },
  { category: "CONSULTANT", count: 115, icon: "📈" },
  { category: "BANKING", count: 115, icon: "🏦" },
  { category: "HEALTHCARE", count: 115, icon: "🩺" },
  { category: "CONSTRUCTION", count: 112, icon: "🏗️" },
  { category: "PUBLIC-RELATIONS", count: 111, icon: "📣" },
  { category: "HR", count: 110, icon: "👥" },
  { category: "DESIGNER", count: 107, icon: "🎨" },
  { category: "ARTS", count: 103, icon: "🎭" },
  { category: "TEACHER", count: 102, icon: "📚" },
  { category: "APPAREL", count: 97, icon: "👗" },
  { category: "DIGITAL-MEDIA", count: 96, icon: "📱" },
  { category: "AGRICULTURE", count: 63, icon: "🌾" },
  { category: "AUTOMOBILE", count: 36, icon: "🚗" },
  { category: "BPO", count: 22, icon: "🖥️" },
];

export const CATEGORY_DETAILS = {
  "BUSINESS-DEVELOPMENT": {
    totalPDFs: 120,
    averageConfidence: 95.5,
    commonKeywords: ["Strategic Planning", "Business Analysis", "Negotiation"],
    documentTypes: "7-10 types",
    description: "Business Development documents focus on growth strategies, market expansion, and business opportunities.",
    typicalContent: ["Business proposals", "Strategic planning documents", "Market analysis reports"],
  },
  "INFORMATION-TECHNOLOGY": {
    totalPDFs: 120,
    averageConfidence: 97.2,
    commonKeywords: ["Software Development", "Cloud Computing", "Cybersecurity"],
    documentTypes: "5-8 types",
    description: "IT documentation covers system designs, technical specifications, and software documentation.",
    typicalContent: ["Technical specifications", "System architecture documents", "Security protocols"],
  },
  CHEF: {
    totalPDFs: 118,
    averageConfidence: 92.4,
    topSkills: ["Culinary Arts", "Menu Planning", "Food Safety"],
    averageExperience: "3-5 years",
    description: "Chefs are culinary professionals responsible for preparing, cooking, and presenting meals to high standards.",
    keyResponsibilities: [
      "Prepare and cook food according to recipes",
      "Manage kitchen staff and operations",
      "Ensure compliance with food safety standards",
    ],
  },
  ACCOUNTANT: {
    totalPDFs: 118,
    averageConfidence: 96.1,
    topSkills: ["Financial Reporting", "Taxation", "Budgeting"],
    averageExperience: "5-7 years",
    description: "Accountants are responsible for preparing and analyzing financial records, ensuring compliance with regulations.",
    keyResponsibilities: [
      "Prepare and examine financial records",
      "Ensure accuracy and compliance with laws",
      "Provide financial advice to clients or management",
    ],
  },
  ENGINEERING: {
    totalPDFs: 118,
    averageConfidence: 94.7,
    topSkills: ["Problem-Solving", "Technical Design", "Project Management"],
    averageExperience: "4-6 years",
    description: "Engineers apply science and mathematics to solve technical problems and design innovative solutions.",
    keyResponsibilities: [
      "Design and develop technical solutions",
      "Test and evaluate systems or products",
      "Collaborate with cross-functional teams",
    ],
  },
  ADVOCATE: {
    totalPDFs: 118,
    averageConfidence: 93.3,
    topSkills: ["Legal Research", "Litigation", "Client Advocacy"],
    averageExperience: "6-8 years",
    description: "Advocates provide legal representation and advice to clients, ensuring their rights are protected.",
    keyResponsibilities: [
      "Represent clients in legal proceedings",
      "Draft legal documents and contracts",
      "Conduct legal research and analysis",
    ],
  },
  FINANCE: {
    totalPDFs: 118,
    averageConfidence: 95.8,
    topSkills: ["Financial Analysis", "Investment Management", "Risk Assessment"],
    averageExperience: "5-8 years",
    description: "Finance professionals manage financial resources, plan investments, and assess financial risks.",
    keyResponsibilities: [
      "Develop financial strategies and plans",
      "Analyze and interpret financial data",
      "Advise on investment opportunities",
    ],
  },
  FITNESS: {
    totalPDFs: 117,
    averageConfidence: 91.9,
    topSkills: ["Personal Training", "Nutrition Planning", "Motivational Skills"],
    averageExperience: "3-5 years",
    description: "Fitness professionals help clients achieve their health and wellness goals through personalized training and guidance.",
    keyResponsibilities: [
      "Create personalized fitness plans",
      "Provide guidance on exercise techniques",
      "Motivate clients to achieve fitness goals",
    ],
  },
  AVIATION: {
    totalPDFs: 117,
    averageConfidence: 94.5,
    topSkills: ["Flight Operations", "Aircraft Maintenance", "Safety Procedures"],
    averageExperience: "5-8 years",
    description: "Aviation professionals ensure the safe and efficient operation of aircraft and related services.",
    keyResponsibilities: [
      "Operate and navigate aircraft",
      "Perform maintenance and inspections",
      "Ensure compliance with aviation safety regulations",
    ],
  },
  SALES: {
    totalPDFs: 116,
    averageConfidence: 93.2,
    topSkills: ["Client Relationship", "Negotiation", "Lead Generation"],
    averageExperience: "2-4 years",
    description: "Sales professionals identify customer needs, promote products or services, and close deals.",
    keyResponsibilities: ["Generate and follow up on leads", "Negotiate contracts and agreements", "Achieve sales targets and goals"],
  },
  CONSULTANT: {
    totalPDFs: 115,
    averageConfidence: 94.6,
    topSkills: ["Problem-Solving", "Data Analysis", "Client Communication"],
    averageExperience: "5-7 years",
    description: "Consultants provide expert advice to help organizations solve problems and achieve goals.",
    keyResponsibilities: [
      "Analyze client needs and challenges",
      "Develop and propose solutions",
      "Support implementation and change management",
    ],
  },
  BANKING: {
    totalPDFs: 115,
    averageConfidence: 96.0,
    topSkills: ["Financial Services", "Risk Management", "Customer Relations"],
    averageExperience: "6-8 years",
    description: "Banking professionals manage financial transactions, customer relationships, and risk assessments.",
    keyResponsibilities: [
      "Assist customers with financial services",
      "Analyze credit and loan applications",
      "Ensure regulatory compliance",
    ],
  },
  HEALTHCARE: {
    totalPDFs: 115,
    averageConfidence: 96.5,
    topSkills: ["Patient Care", "Medical Knowledge", "Attention to Detail"],
    averageExperience: "4-6 years",
    description: "Healthcare professionals provide medical care and support to patients, ensuring their well-being and recovery.",
    keyResponsibilities: ["Diagnose and treat illnesses", "Provide patient education and support", "Maintain medical records and reports"],
  },
  CONSTRUCTION: {
    totalPDFs: 112,
    averageConfidence: 92.8,
    topSkills: ["Blueprint Reading", "Project Management", "Safety Compliance"],
    averageExperience: "5-7 years",
    description: "Construction professionals build and manage infrastructure projects, ensuring quality and safety.",
    keyResponsibilities: [
      "Plan and oversee construction projects",
      "Ensure compliance with safety standards",
      "Manage teams and resources efficiently",
    ],
  },
  "PUBLIC-RELATIONS": {
    totalPDFs: 111,
    averageConfidence: 91.7,
    topSkills: ["Media Relations", "Crisis Management", "Content Creation"],
    averageExperience: "3-5 years",
    description: "Public Relations professionals maintain and promote a positive image for organizations and individuals.",
    keyResponsibilities: [
      "Develop and execute PR campaigns",
      "Manage media relations and communication",
      "Monitor public perception and feedback",
    ],
  },
  HR: {
    totalPDFs: 110,
    averageConfidence: 94.1,
    topSkills: ["Recruitment", "Employee Engagement", "Conflict Resolution"],
    averageExperience: "4-6 years",
    description: "Human Resources professionals manage employee relations, recruitment, and organizational development.",
    keyResponsibilities: [
      "Recruit and onboard employees",
      "Develop HR policies and strategies",
      "Handle employee grievances and conflicts",
    ],
  },
  DESIGNER: {
    totalPDFs: 107,
    averageConfidence: 93.8,
    topSkills: ["Graphic Design", "Creative Thinking", "User Experience"],
    averageExperience: "3-5 years",
    description: "Designers create visual concepts and designs to communicate ideas effectively.",
    keyResponsibilities: ["Develop visual designs and layouts", "Collaborate with clients and teams", "Ensure consistency in branding"],
  },
  ARTS: {
    totalPDFs: 103,
    averageConfidence: 90.5,
    topSkills: ["Artistic Skills", "Creativity", "Self-Expression"],
    averageExperience: "2-4 years",
    description: "Artists express creativity through various mediums such as painting, sculpture, or performance.",
    keyResponsibilities: [
      "Create original artworks",
      "Showcase art through exhibitions or performances",
      "Collaborate on creative projects",
    ],
  },
  TEACHER: {
    totalPDFs: 102,
    averageConfidence: 92.7,
    topSkills: ["Instructional Design", "Classroom Management", "Subject Knowledge"],
    averageExperience: "5-7 years",
    description: "Teachers educate and inspire students, preparing them for personal and professional growth.",
    keyResponsibilities: [
      "Develop and deliver lesson plans",
      "Assess and monitor student progress",
      "Engage students in interactive learning",
    ],
  },
  APPAREL: {
    totalPDFs: 97,
    averageConfidence: 89.3,
    topSkills: ["Fashion Design", "Trend Analysis", "Product Development"],
    averageExperience: "3-5 years",
    description: "Apparel professionals design and produce clothing and accessories that align with market trends.",
    keyResponsibilities: [
      "Design and develop clothing lines",
      "Analyze fashion trends and markets",
      "Manage production and quality control",
    ],
  },
  "DIGITAL-MEDIA": {
    totalPDFs: 96,
    averageConfidence: 93.8,
    topSkills: ["Content Creation", "Social Media Management", "Analytics"],
    averageExperience: "3-5 years",
    description:
      "Digital Media professionals create, manage, and optimize content across various online platforms to engage audiences and drive brand growth.",
    keyResponsibilities: [
      "Develop engaging digital content",
      "Manage social media channels and campaigns",
      "Analyze performance metrics and optimize strategies",
    ],
  },

  AGRICULTURE: {
    totalPDFs: 63,
    averageConfidence: 90.5,
    topSkills: ["Crop Management", "Soil Science", "Sustainable Farming"],
    averageExperience: "4-6 years",
    description:
      "Agriculture professionals specialize in crop production, livestock management, and sustainable farming practices to enhance productivity.",
    keyResponsibilities: [
      "Monitor and manage crop and soil health",
      "Implement sustainable farming practices",
      "Optimize agricultural techniques for higher yield",
    ],
  },

  AUTOMOBILE: {
    totalPDFs: 36,
    averageConfidence: 91.7,
    topSkills: ["Vehicle Maintenance", "Mechanical Engineering", "Diagnostics"],
    averageExperience: "5-7 years",
    description:
      "Automobile professionals ensure the design, maintenance, and efficient operation of vehicles, contributing to the automotive industry's growth.",
    keyResponsibilities: [
      "Diagnose and repair vehicle issues",
      "Design and test automotive systems",
      "Ensure compliance with safety standards",
    ],
  },

  BPO: {
    totalPDFs: 22,
    averageConfidence: 89.2,
    topSkills: ["Customer Support", "Process Optimization", "Communication"],
    averageExperience: "2-4 years",
    description:
      "BPO (Business Process Outsourcing) professionals handle outsourced tasks like customer service, technical support, and back-office operations.",
    keyResponsibilities: [
      "Provide customer service and support",
      "Optimize business processes for efficiency",
      "Handle client inquiries and resolve issues",
    ],
  },
};
