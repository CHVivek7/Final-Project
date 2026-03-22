
'use client'

import Link from 'next/link'
// import Image from 'next/image'
// import { motion } from 'framer-motion';


export default function ProjectOverview() {
  return (
    <main className="flex min-h-screen flex-col items-center p-6 md:p-24">
      <div className="z-10 max-w-5xl w-full">
        {/* Header section */}
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-amber-500 to-red-500 leading-tight">
          Project Overview
        </h1>

        <p className="text-2xl md:text-lg mb-10 max-w-3xl text-foreground/80 border-l-4 border-orange-500/50 pl-4">
          Our project is a Quantum + AI-powered drug discovery platform that enables users to simulate molecular
          energy using quantum algorithms and predict toxicity using machine learning. It combines quantum computing,
          cheminformatics, and AI to accelerate molecular analysis and provide intelligent insights for modern drug research.
        </p>

        {/* Problem Statement Section */}
        <div className="rounded-lg p-6 mb-8 transition-all duration-300 bg-background hover:bg-gradient-to-br hover:from-orange-500/5 hover:to-red-500/10 text-foreground border border-orange-500/30 shadow-sm">
          <h2 className="text-2xl font-semibold mb-3 group-hover:text-orange-500">Problem Statement</h2>

          <div className="flex flex-col md:flex-row gap-6">
            <div className="p-4 rounded-lg bg-background border border-orange-500/20">
              <div className="flex-1">
                <div className="p-4 rounded-lg bg-amber-900/50 mb-4">
                  <p className="font-medium">Traditional drug discovery is slow and computationally expensive.</p>
                </div>
                <p className="text-foreground/70 mb-4">
                  Simulating molecular behavior and predicting toxicity requires significant computational resources
                  and specialized tools. Existing systems often lack integration between simulation and prediction,
                  making the process inefficient and time-consuming.
                </p>
              </div>
            </div>
            <div className="p-4 rounded-lg bg-background border border-orange-500/20">
              <div className="flex-1">
                <div className="p-4 rounded-lg bg-amber-900/50 mb-4">
                  <p className="font-medium">Need for intelligent and accessible molecular analysis.</p>
                </div>
                <p className="text-foreground/70 mb-4">
                  Researchers and students need a unified platform that combines quantum simulation and AI-based prediction
                  to analyze molecules efficiently. Our solution bridges this gap by integrating quantum computing and
                  machine learning into a single interactive system.
                </p>
              </div>
            </div>
          </div>
        </div>
        {/* Team Members Section */}
        <div className="rounded-lg p-6 mb-8 transition-all duration-300 bg-background hover:bg-gradient-to-br hover:from-orange-500/5 hover:to-red-500/10 text-foreground border border-orange-500/30 shadow-sm">
          <h2 className="text-2xl font-semibold mb-4">Our Team</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-4 rounded-lg bg-background border border-orange-500/20">
              <h3 className="text-xl font-medium mb-2 text-orange-500">Vivek</h3>
              <p className="text-foreground/70 mb-2">
                <span className="font-medium">Role:</span> AI/ML Development
              </p>
              <p className="text-foreground/70">
                Developed the AI/ML components for personalized recommendations and implemented intelligent
                algorithms to enhance the user experience with smart discovery features.
              </p>
            </div>
            <div className="p-4 rounded-lg bg-background border border-orange-500/20">
              <h3 className="text-xl font-medium mb-2 text-orange-500">Siva Teja</h3>
              <p className="text-foreground/70 mb-2">
                <span className="font-medium">Role:</span> Backend Architecture
              </p>
              <p className="text-foreground/70">
                Designed and implemented the data models, API endpoints, and core platform functionality
                that powers the discovery capabilities and user interactions.
              </p>
            </div>
            <div className="p-4 rounded-lg bg-background border border-orange-500/20">
              <h3 className="text-xl font-medium mb-2 text-orange-500">Krishna & Karthik</h3>
              <p className="text-foreground/70 mb-2">
                <span className="font-medium">Role:</span> Frontend Development
              </p>
              <p className="text-foreground/70">
                Led the UI/UX design process and implemented the responsive frontend using React and TailwindCSS,
                creating an intuitive and engaging user experience.
              </p>
            </div>
            
          </div>
        </div>
        {/* Project Idea Section */}
        <div className="rounded-lg p-6 mb-8 transition-all duration-300 bg-background hover:bg-gradient-to-br hover:from-orange-500/5 hover:to-red-500/10 text-foreground border border-orange-500/30 shadow-sm">
          <h2 className="text-2xl font-semibold mb-4">Project Idea & Solution</h2>
          <p className="text-foreground/70 mb-4">
            Our platform integrates quantum computing and machine learning to create a unified drug discovery system.
            It allows users to simulate molecular energy using Variational Quantum Eigensolver (VQE) and predict
            toxicity across multiple biological targets using trained AI models.
          </p>

          <p className="text-foreground/70 mb-4">
            By combining classical and quantum features, the system provides deeper insights into molecular behavior,
            helping researchers make faster and more informed decisions in drug discovery workflows.
          </p>
        </div>

        {/* Origin Story Section */}
        <div className="rounded-lg p-6 mb-8 transition-all duration-300 bg-background hover:bg-gradient-to-br hover:from-orange-500/5 hover:to-red-500/10 text-foreground border border-orange-500/30 shadow-sm">
          <h2 className="text-2xl font-semibold mb-4">How We Got Started</h2>
          <p className="text-foreground/70 mb-4">
            By leveraging Qiskit for quantum simulations and machine learning models trained on real-world datasets,
            we built a system that demonstrates how hybrid quantum-classical approaches can enhance molecular analysis.
          </p>
        </div>


        {/* Impact Section */}
        <div className="rounded-lg p-6 mb-8 transition-all duration-300 bg-background hover:bg-gradient-to-br hover:from-orange-500/5 hover:to-red-500/10 text-foreground border border-orange-500/30 shadow-sm">
          <h2 className="text-2xl font-semibold mb-4">Impact & Vision</h2>
          <p className="text-foreground/70 mb-4">
            Our platform aims to accelerate drug discovery by reducing the time required for molecular analysis and
            toxicity prediction. By integrating quantum simulation with AI, we provide a powerful tool for researchers,
            students, and developers exploring computational chemistry.
          </p>
          <p className="text-foreground/70 mb-4">
            In the future, we envision expanding this system with real quantum hardware integration, improved models,
            and larger datasets to support more accurate and scalable drug discovery pipelines. 
          </p>
          <p className="text-foreground/70">
            Our long-term goal is to contribute to next-generation intelligent drug discovery systems that combine
            quantum computing, artificial intelligence, and big data.
          </p>
        </div>

        {/* Features & Benefits Section */}
        <div className="rounded-lg p-6 mb-8 transition-all duration-300 bg-background hover:bg-gradient-to-br hover:from-orange-500/5 hover:to-red-500/10 text-foreground border border-orange-500/30 shadow-sm">
          <h2 className="text-2xl font-semibold mb-4">Key Features & Benefits</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

            <div className="p-4 rounded-lg bg-background border border-orange-500/20">
              <h3 className="text-xl font-medium mb-2 text-orange-500">Quantum Simulation</h3>
              <p className="text-foreground/70">
                Simulates molecular ground state energy using Variational Quantum Eigensolver (VQE) with Qiskit.
              </p>
            </div>

            <div className="p-4 rounded-lg bg-background border border-orange-500/20">
              <h3 className="text-xl font-medium mb-2 text-orange-500">AI Toxicity Prediction</h3>
              <p className="text-foreground/70">
                Predicts molecular toxicity across 12 biological targets using machine learning models trained on Tox21 dataset.
              </p>
            </div>

            <div className="p-4 rounded-lg bg-background border border-orange-500/20">
              <h3 className="text-xl font-medium mb-2 text-orange-500">Feature Engineering</h3>
              <p className="text-foreground/70">
                Combines RDKit descriptors, molecular fingerprints, and quantum features into high-dimensional vectors.
              </p>
            </div>

            <div className="p-4 rounded-lg bg-background border border-orange-500/20">
              <h3 className="text-xl font-medium mb-2 text-orange-500">Visualization & Insights</h3>
              <p className="text-foreground/70">
                Displays molecular structures, energy comparisons, and prediction confidence for better analysis.
              </p>
            </div>

          </div>
        </div>


        {/* Navigation back to home */}
        <div className="mt-12 pt-6 border-t border-foreground/20">
          <div className="flex justify-between w-full">
            <Link href="/" className="text-foreground/60 hover:text-foreground transition-colors">
              &larr; Back to Home
            </Link>
            <Link href="/curated-list" className="text-foreground/60 hover:text-foreground transition-colors">
              To Interface &rarr;
            </Link>
          </div>
        </div>


      </div>
    </main>
  )
}