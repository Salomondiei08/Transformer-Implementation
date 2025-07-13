"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Brain, 
  Zap, 
  Target, 
  ArrowRight, 
  CheckCircle, 
  BarChart3,
  Play,
  Code,
  Sparkles,
  Copy,
  Cpu,
  Activity,
  ExternalLink
} from "lucide-react";
// REMOVE: import Sidebar from "@/components/Sidebar";

const tasks = [
  {
    id: "copy",
    name: "Copy",
    description: "Copy the input sequence exactly",
    icon: Copy,
    color: "#3b82f6",
    examples: ["hello → hello", "123 → 123", "abc → abc"]
  },
  {
    id: "reverse",
    name: "Reverse",
    description: "Reverse the input sequence",
    icon: ArrowRight,
    color: "#ef4444",
    examples: ["hello → olleh", "123 → 321", "abc → cba"]
  },
  {
    id: "sort",
    name: "Sort",
    description: "Sort characters in ascending order",
    icon: BarChart3,
    color: "#10b981",
    examples: ["hello → ehllo", "321 → 123", "cba → abc"]
  },
  {
    id: "shift",
    name: "Shift",
    description: "Shift each character by +1 position",
    icon: Zap,
    color: "#f59e0b",
    examples: ["hello → ifmmp", "abc → bcd", "123 → 234"]
  }
];

export default function Home() {
  const [selectedTask, setSelectedTask] = useState("reverse");
  const [input, setInput] = useState("hello world");
  const [prediction, setPrediction] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [isTraining, setIsTraining] = useState(false);

  const handlePredict = async () => {
    setIsLoading(true);
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const predictions = {
      copy: input,
      reverse: input.split('').reverse().join(''),
      sort: input.split('').sort().join(''),
      shift: input.split('').map(char => {
        if (char === ' ') return ' ';
        if (char >= 'a' && char <= 'z') {
          return String.fromCharCode(((char.charCodeAt(0) - 97 + 1) % 26) + 97);
        }
        if (char >= 'A' && char <= 'Z') {
          return String.fromCharCode(((char.charCodeAt(0) - 65 + 1) % 26) + 65);
        }
        if (char >= '0' && char <= '9') {
          return String.fromCharCode(((char.charCodeAt(0) - 48 + 1) % 10) + 48);
        }
        return char;
      }).join('')
    };
    
    setPrediction(predictions[selectedTask as keyof typeof predictions]);
    setIsLoading(false);
  };

  const startTraining = () => {
    setIsTraining(true);
    setTrainingProgress(0);
    
    const interval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          return 100;
        }
        return prev + Math.random() * 15;
      });
    }, 500);
  };

  const selectedTaskData = tasks.find(task => task.id === selectedTask);

  return (
    <div>
      {/* Dashboard content */}
      <div
        style={{
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%)',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Enhanced Background decoration */}
        <div style={{
          position: 'absolute',
          top: '-50%',
          right: '-50%',
          width: '100%',
          height: '100%',
          background: 'radial-gradient(circle, rgba(59, 130, 246, 0.08) 0%, transparent 70%)',
          pointerEvents: 'none',
          animation: 'pulse 4s ease-in-out infinite'
        }} />
        <div style={{
          position: 'absolute',
          bottom: '-50%',
          left: '-50%',
          width: '100%',
          height: '100%',
          background: 'radial-gradient(circle, rgba(16, 185, 129, 0.08) 0%, transparent 70%)',
          pointerEvents: 'none',
          animation: 'pulse 4s ease-in-out infinite 2s'
        }} />

        <div style={{ position: 'relative', zIndex: 1, padding: '40px 24px' }}>
          <div style={{ maxWidth: '1400px', margin: '0 auto', marginBottom: '48px' }}>
              {/* Enhanced Header */}
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '20px', 
                marginBottom: '16px',
                padding: '32px',
                background: 'rgba(255, 255, 255, 0.06)',
                borderRadius: '20px',
                backdropFilter: 'blur(24px)',
                border: '1px solid rgba(255, 255, 255, 0.12)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
              }}>
                <div style={{ 
                  padding: '16px', 
                  borderRadius: '16px', 
                  background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
                  boxShadow: '0 8px 32px rgba(59, 130, 246, 0.4)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <Brain style={{ height: '32px', width: '32px', color: 'white' }} />
                </div>
                <div style={{ flex: 1 }}>
                  <h1 style={{ 
                    fontSize: '36px', 
                    fontWeight: '800', 
                    color: 'white',
                    margin: 0,
                    marginBottom: '8px',
                    background: 'linear-gradient(135deg, #ffffff, #e2e8f0)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text',
                    letterSpacing: '-0.02em'
                  }}>
                    Transformer Playground
                  </h1>
                  <p style={{ 
                    color: '#94a3b8', 
                    fontSize: '18px',
                    margin: 0,
                    fontWeight: '400',
                    lineHeight: '1.5'
                  }}>
                    Interactive playground for testing custom Transformer models on sequence tasks
                  </p>
                </div>
              </div>
            </div>

            <div style={{ maxWidth: '1400px', margin: '0 auto', display: 'grid', gridTemplateColumns: '1fr 1.6fr', gap: '40px' }}>
              {/* Left Column */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
                {/* Enhanced Task Selection Card */}
                <Card style={{ 
                  background: 'rgba(255, 255, 255, 0.06)',
                  border: '1px solid rgba(255, 255, 255, 0.12)',
                  borderRadius: '20px',
                  backdropFilter: 'blur(24px)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.15)'
                }}>
                  <CardHeader style={{ padding: '28px 28px 20px 28px' }}>
                    <CardTitle style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '16px', 
                      color: 'white',
                      fontSize: '20px',
                      fontWeight: '700'
                    }}>
                      <div style={{ 
                        padding: '12px', 
                        borderRadius: '12px', 
                        background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <Target style={{ height: '20px', width: '20px', color: 'white' }} />
                      </div>
                      Select Task
                    </CardTitle>
                    <CardDescription style={{ color: '#94a3b8', fontSize: '15px', marginTop: '8px' }}>
                      Choose a sequence transformation task to test
                    </CardDescription>
                  </CardHeader>
                  <CardContent style={{ padding: '0 28px 28px 28px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
                    <Select value={selectedTask} onValueChange={setSelectedTask}>
                      <SelectTrigger style={{ 
                        width: '100%',
                        background: 'rgba(255, 255, 255, 0.06)',
                        border: '1px solid rgba(255, 255, 255, 0.12)',
                        borderRadius: '14px',
                        height: '56px',
                        fontSize: '16px',
                        fontWeight: '500'
                      }}>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent style={{ 
                        background: '#1e293b', 
                        border: '1px solid #334155',
                        borderRadius: '12px',
                        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
                      }}>
                        {tasks.map((task) => (
                          <SelectItem key={task.id} value={task.id} style={{ padding: '12px 16px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                              <div style={{ 
                                padding: '8px', 
                                borderRadius: '8px', 
                                backgroundColor: task.color + '20',
                                border: `1px solid ${task.color}40`,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                              }}>
                                <task.icon style={{ height: '16px', width: '16px', color: task.color }} />
                              </div>
                              <span style={{ color: 'white', fontWeight: '500' }}>{task.name}</span>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    {selectedTaskData && (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                        <div style={{ 
                          padding: '20px', 
                          background: 'rgba(255, 255, 255, 0.04)', 
                          borderRadius: '16px',
                          border: `1px solid ${selectedTaskData.color}30`
                        }}>
                          <p style={{ fontSize: '15px', color: '#cbd5e1', margin: '0 0 16px 0', lineHeight: '1.5' }}>
                            {selectedTaskData.description}
                          </p>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            <Label style={{ fontSize: '13px', fontWeight: '600', color: '#94a3b8' }}>Examples:</Label>
                            {selectedTaskData.examples.map((example, index) => (
                              <div key={index} style={{ 
                                fontSize: '13px', 
                                fontFamily: 'monospace', 
                                background: 'rgba(0, 0, 0, 0.3)', 
                                padding: '12px', 
                                borderRadius: '10px',
                                border: '1px solid rgba(255, 255, 255, 0.1)',
                                color: '#e2e8f0',
                                fontWeight: '500'
                              }}>
                                {example}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Enhanced Training Card */}
                <Card style={{ 
                  background: 'rgba(255, 255, 255, 0.06)',
                  border: '1px solid rgba(255, 255, 255, 0.12)',
                  borderRadius: '20px',
                  backdropFilter: 'blur(24px)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.15)'
                }}>
                  <CardHeader style={{ padding: '28px 28px 20px 28px' }}>
                    <CardTitle style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '16px', 
                      color: 'white',
                      fontSize: '20px',
                      fontWeight: '700'
                    }}>
                      <div style={{ 
                        padding: '12px', 
                        borderRadius: '12px', 
                        background: 'linear-gradient(135deg, #10b981, #059669)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <Activity style={{ height: '20px', width: '20px', color: 'white' }} />
                      </div>
                      Model Training
                    </CardTitle>
                    <CardDescription style={{ color: '#94a3b8', fontSize: '15px', marginTop: '8px' }}>
                      Monitor training progress and model status
                    </CardDescription>
                  </CardHeader>
                  <CardContent style={{ padding: '0 28px 28px 28px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '15px' }}>
                        <span style={{ color: '#94a3b8', fontWeight: '500' }}>Training Progress</span>
                        <span style={{ fontFamily: 'monospace', color: 'white', fontWeight: '700', fontSize: '16px' }}>{Math.round(trainingProgress)}%</span>
                      </div>
                      <div style={{ 
                        width: '100%', 
                        height: '10px', 
                        background: 'rgba(255, 255, 255, 0.1)', 
                        borderRadius: '6px',
                        overflow: 'hidden'
                      }}>
                        <div style={{ 
                          width: `${trainingProgress}%`, 
                          height: '100%', 
                          background: 'linear-gradient(90deg, #10b981, #34d399)',
                          borderRadius: '6px',
                          transition: 'width 0.3s ease',
                          boxShadow: '0 0 20px rgba(16, 185, 129, 0.3)'
                        }} />
                      </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '15px' }}>
                      <div style={{ 
                        width: '12px', 
                        height: '12px', 
                        borderRadius: '50%', 
                        backgroundColor: isTraining ? '#fbbf24' : '#10b981',
                        boxShadow: isTraining ? '0 0 16px rgba(251, 191, 36, 0.6)' : '0 0 16px rgba(16, 185, 129, 0.6)',
                        animation: isTraining ? 'pulse 2s ease-in-out infinite' : 'none'
                      }} />
                      <span style={{ color: '#cbd5e1', fontWeight: '500' }}>
                        {isTraining ? 'Training in progress...' : 'Model ready'}
                      </span>
                    </div>

                    <Button 
                      onClick={startTraining} 
                      disabled={isTraining}
                      style={{ 
                        width: '100%',
                        height: '56px',
                        background: isTraining ? 'rgba(251, 191, 36, 0.2)' : 'linear-gradient(135deg, #10b981, #059669)',
                        border: '1px solid rgba(255, 255, 255, 0.12)',
                        borderRadius: '14px',
                        color: 'white',
                        fontWeight: '600',
                        fontSize: '16px',
                        transition: 'all 0.3s ease'
                      }}
                      onMouseEnter={(e) => {
                        if (!isTraining) {
                          e.currentTarget.style.transform = 'translateY(-2px)';
                          e.currentTarget.style.boxShadow = '0 12px 32px rgba(16, 185, 129, 0.4)';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!isTraining) {
                          e.currentTarget.style.transform = 'translateY(0)';
                          e.currentTarget.style.boxShadow = 'none';
                        }
                      }}
                    >
                      <Play style={{ height: '18px', width: '18px', marginRight: '12px' }} />
                      {isTraining ? 'Training...' : 'Start Training'}
                    </Button>
                  </CardContent>
                </Card>

                {/* Enhanced Stats Card */}
                <Card style={{ 
                  background: 'rgba(255, 255, 255, 0.06)',
                  border: '1px solid rgba(255, 255, 255, 0.12)',
                  borderRadius: '20px',
                  backdropFilter: 'blur(24px)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.15)'
                }}>
                  <CardHeader style={{ padding: '28px 28px 20px 28px' }}>
                    <CardTitle style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '16px', 
                      color: 'white',
                      fontSize: '20px',
                      fontWeight: '700'
                    }}>
                      <div style={{ 
                        padding: '12px', 
                        borderRadius: '12px', 
                        background: 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <Cpu style={{ height: '20px', width: '20px', color: 'white' }} />
                      </div>
                      Model Statistics
                    </CardTitle>
                  </CardHeader>
                  <CardContent style={{ padding: '0 28px 28px 28px' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                      <div style={{ 
                        textAlign: 'center',
                        padding: '20px',
                        background: 'rgba(59, 130, 246, 0.12)',
                        borderRadius: '16px',
                        border: '1px solid rgba(59, 130, 246, 0.25)'
                      }}>
                        <div style={{ fontSize: '32px', fontWeight: '800', color: '#3b82f6', marginBottom: '8px' }}>12M</div>
                        <div style={{ fontSize: '13px', color: '#94a3b8', fontWeight: '600' }}>Parameters</div>
                      </div>
                      <div style={{ 
                        textAlign: 'center',
                        padding: '20px',
                        background: 'rgba(16, 185, 129, 0.12)',
                        borderRadius: '16px',
                        border: '1px solid rgba(16, 185, 129, 0.25)'
                      }}>
                        <div style={{ fontSize: '32px', fontWeight: '800', color: '#10b981', marginBottom: '8px' }}>99.8%</div>
                        <div style={{ fontSize: '13px', color: '#94a3b8', fontWeight: '600' }}>Accuracy</div>
                      </div>
                      <div style={{ 
                        textAlign: 'center',
                        padding: '20px',
                        background: 'rgba(139, 92, 246, 0.12)',
                        borderRadius: '16px',
                        border: '1px solid rgba(139, 92, 246, 0.25)'
                      }}>
                        <div style={{ fontSize: '32px', fontWeight: '800', color: '#8b5cf6', marginBottom: '8px' }}>8</div>
                        <div style={{ fontSize: '13px', color: '#94a3b8', fontWeight: '600' }}>Layers</div>
                      </div>
                      <div style={{ 
                        textAlign: 'center',
                        padding: '20px',
                        background: 'rgba(245, 158, 11, 0.12)',
                        borderRadius: '16px',
                        border: '1px solid rgba(245, 158, 11, 0.25)'
                      }}>
                        <div style={{ fontSize: '32px', fontWeight: '800', color: '#f59e0b', marginBottom: '8px' }}>512</div>
                        <div style={{ fontSize: '13px', color: '#94a3b8', fontWeight: '600' }}>Embedding</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Right Column */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
                {/* Enhanced Input Card */}
                <Card style={{ 
                  background: 'rgba(255, 255, 255, 0.06)',
                  border: '1px solid rgba(255, 255, 255, 0.12)',
                  borderRadius: '20px',
                  backdropFilter: 'blur(24px)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.15)'
                }}>
                  <CardHeader style={{ padding: '28px 28px 20px 28px' }}>
                    <CardTitle style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '16px', 
                      color: 'white',
                      fontSize: '20px',
                      fontWeight: '700'
                    }}>
                      <div style={{ 
                        padding: '12px', 
                        borderRadius: '12px', 
                        background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <Code style={{ height: '20px', width: '20px', color: 'white' }} />
                      </div>
                      Input Sequence
                    </CardTitle>
                    <CardDescription style={{ color: '#94a3b8', fontSize: '15px', marginTop: '8px' }}>
                      Enter a sequence to transform using the selected task
                    </CardDescription>
                  </CardHeader>
                  <CardContent style={{ padding: '0 28px 28px 28px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      <Label htmlFor="input" style={{ color: 'white', fontSize: '15px', fontWeight: '600' }}>Input Text</Label>
                      <Input
                        id="input"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Enter your sequence here..."
                        style={{ 
                          fontFamily: 'monospace',
                          height: '64px',
                          background: 'rgba(255, 255, 255, 0.06)',
                          border: '1px solid rgba(255, 255, 255, 0.12)',
                          borderRadius: '14px',
                          color: 'white',
                          fontSize: '16px',
                          padding: '0 20px',
                          fontWeight: '500'
                        }}
                      />
                    </div>
                    
                    <Button 
                      onClick={handlePredict} 
                      disabled={isLoading || !input.trim()}
                      style={{ 
                        width: '100%',
                        height: '64px',
                        background: isLoading || !input.trim() ? 'rgba(255, 255, 255, 0.1)' : 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
                        border: '1px solid rgba(255, 255, 255, 0.12)',
                        borderRadius: '14px',
                        color: 'white',
                        fontWeight: '600',
                        fontSize: '16px',
                        transition: 'all 0.3s ease'
                      }}
                      onMouseEnter={(e) => {
                        if (!isLoading && input.trim()) {
                          e.currentTarget.style.transform = 'translateY(-2px)';
                          e.currentTarget.style.boxShadow = '0 12px 32px rgba(59, 130, 246, 0.4)';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!isLoading && input.trim()) {
                          e.currentTarget.style.transform = 'translateY(0)';
                          e.currentTarget.style.boxShadow = 'none';
                        }
                      }}
                    >
                      {isLoading ? (
                        <>
                          <div style={{ 
                            animation: 'spin 1s linear infinite', 
                            borderRadius: '50%', 
                            height: '24px', 
                            width: '24px', 
                            border: '2px solid transparent', 
                            borderBottomColor: 'white', 
                            marginRight: '16px' 
                          }} />
                          Processing...
                        </>
                      ) : (
                        <>
                          <Sparkles style={{ height: '24px', width: '24px', marginRight: '16px' }} />
                          Generate Prediction
                        </>
                      )}
                    </Button>
                  </CardContent>
                </Card>

                {/* Enhanced Prediction Card */}
                <Card style={{ 
                  background: 'rgba(255, 255, 255, 0.06)',
                  border: '1px solid rgba(255, 255, 255, 0.12)',
                  borderRadius: '20px',
                  backdropFilter: 'blur(24px)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.15)'
                }}>
                  <CardHeader style={{ padding: '28px 28px 20px 28px' }}>
                    <CardTitle style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '16px', 
                      color: 'white',
                      fontSize: '20px',
                      fontWeight: '700'
                    }}>
                      <div style={{ 
                        padding: '12px', 
                        borderRadius: '12px', 
                        background: 'linear-gradient(135deg, #10b981, #059669)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <CheckCircle style={{ height: '20px', width: '20px', color: 'white' }} />
                      </div>
                      Model Prediction
                    </CardTitle>
                    <CardDescription style={{ color: '#94a3b8', fontSize: '15px', marginTop: '8px' }}>
                      The transformed sequence based on your input and selected task
                    </CardDescription>
                  </CardHeader>
                  <CardContent style={{ padding: '0 28px 28px 28px' }}>
                    {prediction ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                          <Label style={{ fontSize: '15px', color: '#94a3b8', fontWeight: '600' }}>Input</Label>
                          <div style={{ 
                            padding: '20px', 
                            background: 'rgba(0, 0, 0, 0.3)', 
                            borderRadius: '16px', 
                            fontFamily: 'monospace', 
                            fontSize: '16px', 
                            border: '1px solid rgba(255, 255, 255, 0.12)',
                            color: '#e2e8f0',
                            minHeight: '64px',
                            display: 'flex',
                            alignItems: 'center',
                            fontWeight: '500'
                          }}>
                            {input}
                          </div>
                        </div>
                        
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <div style={{ 
                            padding: '12px', 
                            borderRadius: '12px', 
                            background: 'rgba(255, 255, 255, 0.08)',
                            border: '1px solid rgba(255, 255, 255, 0.15)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                          }}>
                            <ArrowRight style={{ height: '24px', width: '24px', color: '#94a3b8' }} />
                          </div>
                        </div>
                        
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                          <Label style={{ fontSize: '15px', color: '#94a3b8', fontWeight: '600' }}>Output</Label>
                          <div style={{ 
                            padding: '20px', 
                            background: 'rgba(16, 185, 129, 0.12)', 
                            borderRadius: '16px', 
                            fontFamily: 'monospace', 
                            fontSize: '16px', 
                            border: '1px solid rgba(16, 185, 129, 0.3)',
                            color: '#10b981',
                            minHeight: '64px',
                            display: 'flex',
                            alignItems: 'center',
                            fontWeight: '600'
                          }}>
                            {prediction}
                          </div>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                          <Badge style={{ 
                            background: 'rgba(16, 185, 129, 0.2)', 
                            color: '#10b981', 
                            border: '1px solid rgba(16, 185, 129, 0.3)',
                            padding: '8px 16px',
                            borderRadius: '10px',
                            fontSize: '13px',
                            fontWeight: '600'
                          }}>
                            <CheckCircle style={{ height: '14px', width: '14px', marginRight: '8px' }} />
                            Success
                          </Badge>
                          <span style={{ fontSize: '13px', color: '#94a3b8', fontWeight: '500' }}>
                            Prediction completed in 1.2s
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div style={{ textAlign: 'center', padding: '48px 0' }}>
                        <div style={{ 
                          width: '96px', 
                          height: '96px', 
                          margin: '0 auto', 
                          marginBottom: '24px', 
                          borderRadius: '50%', 
                          background: 'rgba(255, 255, 255, 0.06)',
                          border: '1px solid rgba(255, 255, 255, 0.12)',
                          display: 'flex', 
                          alignItems: 'center', 
                          justifyContent: 'center'
                        }}>
                          <Sparkles style={{ height: '48px', width: '48px', color: '#94a3b8' }} />
                        </div>
                        <p style={{ color: '#94a3b8', fontSize: '16px', margin: 0, fontWeight: '500' }}>
                          Enter input and click &quot;Generate Prediction&quot; to see results
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Enhanced Examples Card */}
                <Card style={{ 
                  background: 'rgba(255, 255, 255, 0.06)',
                  border: '1px solid rgba(255, 255, 255, 0.12)',
                  borderRadius: '20px',
                  backdropFilter: 'blur(24px)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.15)'
                }}>
                  <CardHeader style={{ padding: '28px 28px 20px 28px' }}>
                    <CardTitle style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '16px', 
                      color: 'white',
                      fontSize: '20px',
                      fontWeight: '700'
                    }}>
                      <div style={{ 
                        padding: '12px', 
                        borderRadius: '12px', 
                        background: 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <Target style={{ height: '20px', width: '20px', color: 'white' }} />
                      </div>
                      Quick Examples
                    </CardTitle>
                    <CardDescription style={{ color: '#94a3b8', fontSize: '15px', marginTop: '8px' }}>
                      Try these examples to test different tasks
                    </CardDescription>
                  </CardHeader>
                  <CardContent style={{ padding: '0 28px 28px 28px' }}>
                    <Tabs defaultValue="copy" style={{ width: '100%' }}>
                      <TabsList style={{ 
                        display: 'grid', 
                        width: '100%', 
                        gridTemplateColumns: 'repeat(4, 1fr)',
                        background: 'rgba(255, 255, 255, 0.06)',
                        border: '1px solid rgba(255, 255, 255, 0.12)',
                        borderRadius: '14px',
                        padding: '6px'
                      }}>
                        {tasks.map((task) => (
                          <TabsTrigger 
                            key={task.id} 
                            value={task.id} 
                            style={{ 
                              fontSize: '13px',
                              fontWeight: '600',
                              borderRadius: '10px',
                              color: '#94a3b8',
                              background: 'transparent',
                              border: 'none',
                              padding: '10px 14px',
                              transition: 'all 0.2s ease'
                            }}
                          >
                            {task.name}
                          </TabsTrigger>
                        ))}
                      </TabsList>
                      {tasks.map((task) => (
                        <TabsContent key={task.id} value={task.id} style={{ marginTop: '24px' }}>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                            {task.examples.map((example, index) => (
                              <button
                                key={index}
                                onClick={() => {
                                  setSelectedTask(task.id);
                                  setInput(example.split(' → ')[0]);
                                  setPrediction('');
                                }}
                                style={{ 
                                  width: '100%', 
                                  textAlign: 'left', 
                                  padding: '20px', 
                                  borderRadius: '14px', 
                                  background: 'rgba(255, 255, 255, 0.06)', 
                                  border: '1px solid rgba(255, 255, 255, 0.12)', 
                                  transition: 'all 0.3s ease',
                                  cursor: 'pointer'
                                }}
                                onMouseEnter={(e) => {
                                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.2)';
                                  e.currentTarget.style.transform = 'translateY(-2px)';
                                  e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.2)';
                                }}
                                onMouseLeave={(e) => {
                                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.06)';
                                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.12)';
                                  e.currentTarget.style.transform = 'translateY(0)';
                                  e.currentTarget.style.boxShadow = 'none';
                                }}
                              >
                                <div style={{ 
                                  fontFamily: 'monospace', 
                                  fontSize: '15px', 
                                  color: '#e2e8f0',
                                  fontWeight: '500'
                                }}>
                                  {example}
                                </div>
                              </button>
                            ))}
                          </div>
                        </TabsContent>
                      ))}
                    </Tabs>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </div>
      {/* Signature Footer */}
      <div style={{
        position: 'relative',
        zIndex: 1,
        padding: '48px 24px 32px 24px',
        textAlign: 'center',
      }}>
        <div style={{
          maxWidth: '1400px',
          margin: '0 auto',
          padding: '28px',
          background: 'rgba(255, 255, 255, 0.04)',
          borderRadius: '20px',
          backdropFilter: 'blur(24px)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '12px',
            fontSize: '16px',
            fontWeight: '500',
            color: '#94a3b8'
          }}>
            <span>Made with</span>
            <span style={{
              fontSize: '24px',
              animation: 'pulse 2s ease-in-out infinite'
            }}>❤️</span>
            <span>by</span>
            <a 
              href="https://theguy.reinvent-labs.com/"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                color: 'white',
                fontWeight: '600',
                background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                textDecoration: 'none',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-1px)';
                e.currentTarget.style.filter = 'brightness(1.2)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.filter = 'brightness(1)';
              }}
            >
              Salomon Diei
              <ExternalLink style={{ height: '14px', width: '14px' }} />
            </a>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
