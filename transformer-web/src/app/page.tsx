"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
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
  Settings,
  Code,
  Sparkles,
  Copy,
  Cpu,
  Activity
} from "lucide-react";

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
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%)',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Background decoration */}
      <div style={{
        position: 'absolute',
        top: '-50%',
        right: '-50%',
        width: '100%',
        height: '100%',
        background: 'radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%)',
        pointerEvents: 'none'
      }} />
      <div style={{
        position: 'absolute',
        bottom: '-50%',
        left: '-50%',
        width: '100%',
        height: '100%',
        background: 'radial-gradient(circle, rgba(16, 185, 129, 0.1) 0%, transparent 70%)',
        pointerEvents: 'none'
      }} />

      <div style={{ position: 'relative', zIndex: 1, padding: '32px 24px' }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', marginBottom: '40px' }}>
          {/* Header */}
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '16px', 
            marginBottom: '12px',
            padding: '24px',
            background: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '16px',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(255, 255, 255, 0.1)'
          }}>
            <div style={{ 
              padding: '12px', 
              borderRadius: '12px', 
              background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
              boxShadow: '0 8px 32px rgba(59, 130, 246, 0.3)'
            }}>
              <Brain style={{ height: '28px', width: '28px', color: 'white' }} />
            </div>
            <div>
              <h1 style={{ 
                fontSize: '32px', 
                fontWeight: '800', 
                color: 'white',
                margin: 0,
                background: 'linear-gradient(135deg, #ffffff, #e2e8f0)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text'
              }}>
                Transformer Playground
              </h1>
              <p style={{ 
                color: '#94a3b8', 
                fontSize: '16px',
                margin: '4px 0 0 0',
                fontWeight: '400'
              }}>
                Interactive playground for testing custom Transformer models on sequence tasks
              </p>
            </div>
          </div>
        </div>

        <div style={{ maxWidth: '1400px', margin: '0 auto', display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: '32px' }}>
          {/* Left Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* Task Selection Card */}
            <Card style={{ 
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              backdropFilter: 'blur(20px)'
            }}>
              <CardHeader style={{ padding: '24px 24px 16px 24px' }}>
                <CardTitle style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '12px', 
                  color: 'white',
                  fontSize: '18px',
                  fontWeight: '600'
                }}>
                  <div style={{ 
                    padding: '8px', 
                    borderRadius: '8px', 
                    background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)'
                  }}>
                    <Target style={{ height: '16px', width: '16px', color: 'white' }} />
                  </div>
                  Select Task
                </CardTitle>
                <CardDescription style={{ color: '#94a3b8', fontSize: '14px' }}>
                  Choose a sequence transformation task to test
                </CardDescription>
              </CardHeader>
              <CardContent style={{ padding: '0 24px 24px 24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <Select value={selectedTask} onValueChange={setSelectedTask}>
                  <SelectTrigger style={{ 
                    width: '100%',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '12px',
                    height: '48px'
                  }}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent style={{ background: '#1e293b', border: '1px solid #334155' }}>
                    {tasks.map((task) => (
                      <SelectItem key={task.id} value={task.id}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                          <div style={{ 
                            padding: '6px', 
                            borderRadius: '6px', 
                            backgroundColor: task.color + '20',
                            border: `1px solid ${task.color}40`
                          }}>
                            <task.icon style={{ height: '14px', width: '14px', color: task.color }} />
                          </div>
                          <span style={{ color: 'white' }}>{task.name}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {selectedTaskData && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ 
                      padding: '16px', 
                      background: 'rgba(255, 255, 255, 0.03)', 
                      borderRadius: '12px',
                      border: `1px solid ${selectedTaskData.color}30`
                    }}>
                      <p style={{ fontSize: '14px', color: '#cbd5e1', margin: '0 0 12px 0' }}>
                        {selectedTaskData.description}
                      </p>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <Label style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8' }}>Examples:</Label>
                        {selectedTaskData.examples.map((example, index) => (
                          <div key={index} style={{ 
                            fontSize: '12px', 
                            fontFamily: 'monospace', 
                            background: 'rgba(0, 0, 0, 0.3)', 
                            padding: '10px', 
                            borderRadius: '8px',
                            border: '1px solid rgba(255, 255, 255, 0.1)',
                            color: '#e2e8f0'
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

            {/* Training Card */}
            <Card style={{ 
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              backdropFilter: 'blur(20px)'
            }}>
              <CardHeader style={{ padding: '24px 24px 16px 24px' }}>
                <CardTitle style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '12px', 
                  color: 'white',
                  fontSize: '18px',
                  fontWeight: '600'
                }}>
                  <div style={{ 
                    padding: '8px', 
                    borderRadius: '8px', 
                    background: 'linear-gradient(135deg, #10b981, #059669)'
                  }}>
                    <Activity style={{ height: '16px', width: '16px', color: 'white' }} />
                  </div>
                  Model Training
                </CardTitle>
                <CardDescription style={{ color: '#94a3b8', fontSize: '14px' }}>
                  Monitor training progress and model status
                </CardDescription>
              </CardHeader>
              <CardContent style={{ padding: '0 24px 24px 24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '14px' }}>
                    <span style={{ color: '#94a3b8' }}>Training Progress</span>
                    <span style={{ fontFamily: 'monospace', color: 'white', fontWeight: '600' }}>{Math.round(trainingProgress)}%</span>
                  </div>
                  <div style={{ 
                    width: '100%', 
                    height: '8px', 
                    background: 'rgba(255, 255, 255, 0.1)', 
                    borderRadius: '4px',
                    overflow: 'hidden'
                  }}>
                    <div style={{ 
                      width: `${trainingProgress}%`, 
                      height: '100%', 
                      background: 'linear-gradient(90deg, #10b981, #34d399)',
                      borderRadius: '4px',
                      transition: 'width 0.3s ease'
                    }} />
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '14px' }}>
                  <div style={{ 
                    width: '10px', 
                    height: '10px', 
                    borderRadius: '50%', 
                    backgroundColor: isTraining ? '#fbbf24' : '#10b981',
                    boxShadow: isTraining ? '0 0 12px rgba(251, 191, 36, 0.5)' : '0 0 12px rgba(16, 185, 129, 0.5)'
                  }} />
                  <span style={{ color: '#cbd5e1' }}>
                    {isTraining ? 'Training in progress...' : 'Model ready'}
                  </span>
                </div>

                <Button 
                  onClick={startTraining} 
                  disabled={isTraining}
                  style={{ 
                    width: '100%',
                    height: '48px',
                    background: isTraining ? 'rgba(251, 191, 36, 0.2)' : 'linear-gradient(135deg, #10b981, #059669)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '12px',
                    color: 'white',
                    fontWeight: '600',
                    transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    if (!isTraining) {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 8px 25px rgba(16, 185, 129, 0.3)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isTraining) {
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = 'none';
                    }
                  }}
                >
                  <Play style={{ height: '16px', width: '16px', marginRight: '8px' }} />
                  {isTraining ? 'Training...' : 'Start Training'}
                </Button>
              </CardContent>
            </Card>

            {/* Stats Card */}
            <Card style={{ 
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              backdropFilter: 'blur(20px)'
            }}>
              <CardHeader style={{ padding: '24px 24px 16px 24px' }}>
                <CardTitle style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '12px', 
                  color: 'white',
                  fontSize: '18px',
                  fontWeight: '600'
                }}>
                  <div style={{ 
                    padding: '8px', 
                    borderRadius: '8px', 
                    background: 'linear-gradient(135deg, #8b5cf6, #7c3aed)'
                  }}>
                    <Cpu style={{ height: '16px', width: '16px', color: 'white' }} />
                  </div>
                  Model Statistics
                </CardTitle>
              </CardHeader>
              <CardContent style={{ padding: '0 24px 24px 24px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                  <div style={{ 
                    textAlign: 'center',
                    padding: '16px',
                    background: 'rgba(59, 130, 246, 0.1)',
                    borderRadius: '12px',
                    border: '1px solid rgba(59, 130, 246, 0.2)'
                  }}>
                    <div style={{ fontSize: '28px', fontWeight: '800', color: '#3b82f6', marginBottom: '4px' }}>12M</div>
                    <div style={{ fontSize: '12px', color: '#94a3b8', fontWeight: '500' }}>Parameters</div>
                  </div>
                  <div style={{ 
                    textAlign: 'center',
                    padding: '16px',
                    background: 'rgba(16, 185, 129, 0.1)',
                    borderRadius: '12px',
                    border: '1px solid rgba(16, 185, 129, 0.2)'
                  }}>
                    <div style={{ fontSize: '28px', fontWeight: '800', color: '#10b981', marginBottom: '4px' }}>99.8%</div>
                    <div style={{ fontSize: '12px', color: '#94a3b8', fontWeight: '500' }}>Accuracy</div>
                  </div>
                  <div style={{ 
                    textAlign: 'center',
                    padding: '16px',
                    background: 'rgba(139, 92, 246, 0.1)',
                    borderRadius: '12px',
                    border: '1px solid rgba(139, 92, 246, 0.2)'
                  }}>
                    <div style={{ fontSize: '28px', fontWeight: '800', color: '#8b5cf6', marginBottom: '4px' }}>8</div>
                    <div style={{ fontSize: '12px', color: '#94a3b8', fontWeight: '500' }}>Layers</div>
                  </div>
                  <div style={{ 
                    textAlign: 'center',
                    padding: '16px',
                    background: 'rgba(245, 158, 11, 0.1)',
                    borderRadius: '12px',
                    border: '1px solid rgba(245, 158, 11, 0.2)'
                  }}>
                    <div style={{ fontSize: '28px', fontWeight: '800', color: '#f59e0b', marginBottom: '4px' }}>512</div>
                    <div style={{ fontSize: '12px', color: '#94a3b8', fontWeight: '500' }}>Embedding</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* Input Card */}
            <Card style={{ 
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              backdropFilter: 'blur(20px)'
            }}>
              <CardHeader style={{ padding: '24px 24px 16px 24px' }}>
                <CardTitle style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '12px', 
                  color: 'white',
                  fontSize: '18px',
                  fontWeight: '600'
                }}>
                  <div style={{ 
                    padding: '8px', 
                    borderRadius: '8px', 
                    background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)'
                  }}>
                    <Code style={{ height: '16px', width: '16px', color: 'white' }} />
                  </div>
                  Input Sequence
                </CardTitle>
                <CardDescription style={{ color: '#94a3b8', fontSize: '14px' }}>
                  Enter a sequence to transform using the selected task
                </CardDescription>
              </CardHeader>
              <CardContent style={{ padding: '0 24px 24px 24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <Label htmlFor="input" style={{ color: 'white', fontSize: '14px', fontWeight: '500' }}>Input Text</Label>
                  <Input
                    id="input"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Enter your sequence here..."
                    style={{ 
                      fontFamily: 'monospace',
                      height: '56px',
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '12px',
                      color: 'white',
                      fontSize: '16px',
                      padding: '0 16px'
                    }}
                  />
                </div>
                
                <Button 
                  onClick={handlePredict} 
                  disabled={isLoading || !input.trim()}
                  style={{ 
                    width: '100%',
                    height: '56px',
                    background: isLoading || !input.trim() ? 'rgba(255, 255, 255, 0.1)' : 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '12px',
                    color: 'white',
                    fontWeight: '600',
                    fontSize: '16px',
                    transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    if (!isLoading && input.trim()) {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 8px 25px rgba(59, 130, 246, 0.3)';
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
                        height: '20px', 
                        width: '20px', 
                        border: '2px solid transparent', 
                        borderBottomColor: 'white', 
                        marginRight: '12px' 
                      }} />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Sparkles style={{ height: '20px', width: '20px', marginRight: '12px' }} />
                      Generate Prediction
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Prediction Card */}
            <Card style={{ 
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              backdropFilter: 'blur(20px)'
            }}>
              <CardHeader style={{ padding: '24px 24px 16px 24px' }}>
                <CardTitle style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '12px', 
                  color: 'white',
                  fontSize: '18px',
                  fontWeight: '600'
                }}>
                  <div style={{ 
                    padding: '8px', 
                    borderRadius: '8px', 
                    background: 'linear-gradient(135deg, #10b981, #059669)'
                  }}>
                    <CheckCircle style={{ height: '16px', width: '16px', color: 'white' }} />
                  </div>
                  Model Prediction
                </CardTitle>
                <CardDescription style={{ color: '#94a3b8', fontSize: '14px' }}>
                  The transformed sequence based on your input and selected task
                </CardDescription>
              </CardHeader>
              <CardContent style={{ padding: '0 24px 24px 24px' }}>
                {prediction ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <Label style={{ fontSize: '14px', color: '#94a3b8', fontWeight: '500' }}>Input</Label>
                      <div style={{ 
                        padding: '16px', 
                        background: 'rgba(0, 0, 0, 0.3)', 
                        borderRadius: '12px', 
                        fontFamily: 'monospace', 
                        fontSize: '16px', 
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        color: '#e2e8f0',
                        minHeight: '56px',
                        display: 'flex',
                        alignItems: 'center'
                      }}>
                        {input}
                      </div>
                    </div>
                    
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <div style={{ 
                        padding: '8px', 
                        borderRadius: '8px', 
                        background: 'rgba(255, 255, 255, 0.1)',
                        border: '1px solid rgba(255, 255, 255, 0.2)'
                      }}>
                        <ArrowRight style={{ height: '20px', width: '20px', color: '#94a3b8' }} />
                      </div>
                    </div>
                    
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <Label style={{ fontSize: '14px', color: '#94a3b8', fontWeight: '500' }}>Output</Label>
                      <div style={{ 
                        padding: '16px', 
                        background: 'rgba(16, 185, 129, 0.1)', 
                        borderRadius: '12px', 
                        fontFamily: 'monospace', 
                        fontSize: '16px', 
                        border: '1px solid rgba(16, 185, 129, 0.3)',
                        color: '#10b981',
                        minHeight: '56px',
                        display: 'flex',
                        alignItems: 'center',
                        fontWeight: '600'
                      }}>
                        {prediction}
                      </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <Badge style={{ 
                        background: 'rgba(16, 185, 129, 0.2)', 
                        color: '#10b981', 
                        border: '1px solid rgba(16, 185, 129, 0.3)',
                        padding: '6px 12px',
                        borderRadius: '8px',
                        fontSize: '12px',
                        fontWeight: '600'
                      }}>
                        <CheckCircle style={{ height: '12px', width: '12px', marginRight: '6px' }} />
                        Success
                      </Badge>
                      <span style={{ fontSize: '12px', color: '#94a3b8' }}>
                        Prediction completed in 1.2s
                      </span>
                    </div>
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '40px 0' }}>
                    <div style={{ 
                      width: '80px', 
                      height: '80px', 
                      margin: '0 auto', 
                      marginBottom: '20px', 
                      borderRadius: '50%', 
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center'
                    }}>
                      <Sparkles style={{ height: '40px', width: '40px', color: '#94a3b8' }} />
                    </div>
                    <p style={{ color: '#94a3b8', fontSize: '16px', margin: 0 }}>
                      Enter input and click "Generate Prediction" to see results
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Examples Card */}
            <Card style={{ 
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              backdropFilter: 'blur(20px)'
            }}>
              <CardHeader style={{ padding: '24px 24px 16px 24px' }}>
                <CardTitle style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '12px', 
                  color: 'white',
                  fontSize: '18px',
                  fontWeight: '600'
                }}>
                  <div style={{ 
                    padding: '8px', 
                    borderRadius: '8px', 
                    background: 'linear-gradient(135deg, #8b5cf6, #7c3aed)'
                  }}>
                    <Target style={{ height: '16px', width: '16px', color: 'white' }} />
                  </div>
                  Quick Examples
                </CardTitle>
                <CardDescription style={{ color: '#94a3b8', fontSize: '14px' }}>
                  Try these examples to test different tasks
                </CardDescription>
              </CardHeader>
              <CardContent style={{ padding: '0 24px 24px 24px' }}>
                <Tabs defaultValue="copy" style={{ width: '100%' }}>
                  <TabsList style={{ 
                    display: 'grid', 
                    width: '100%', 
                    gridTemplateColumns: 'repeat(4, 1fr)',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '12px',
                    padding: '4px'
                  }}>
                    {tasks.map((task) => (
                      <TabsTrigger 
                        key={task.id} 
                        value={task.id} 
                        style={{ 
                          fontSize: '12px',
                          fontWeight: '600',
                          borderRadius: '8px',
                          color: '#94a3b8',
                          background: 'transparent',
                          border: 'none',
                          padding: '8px 12px'
                        }}
                      >
                        {task.name}
                      </TabsTrigger>
                    ))}
                  </TabsList>
                  {tasks.map((task) => (
                    <TabsContent key={task.id} value={task.id} style={{ marginTop: '20px' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
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
                              padding: '16px', 
                              borderRadius: '12px', 
                              background: 'rgba(255, 255, 255, 0.05)', 
                              border: '1px solid rgba(255, 255, 255, 0.1)', 
                              transition: 'all 0.2s ease',
                              cursor: 'pointer'
                            }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                              e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.2)';
                              e.currentTarget.style.transform = 'translateY(-2px)';
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                              e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                              e.currentTarget.style.transform = 'translateY(0)';
                            }}
                          >
                            <div style={{ 
                              fontFamily: 'monospace', 
                              fontSize: '14px', 
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

      {/* Signature Footer */}
      <div style={{
        position: 'relative',
        zIndex: 1,
        padding: '40px 24px 24px 24px',
        textAlign: 'center'
      }}>
        <div style={{
          maxWidth: '1400px',
          margin: '0 auto',
          padding: '24px',
          background: 'rgba(255, 255, 255, 0.03)',
          borderRadius: '16px',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.05)'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            fontSize: '16px',
            fontWeight: '500',
            color: '#94a3b8'
          }}>
            <span>Made with</span>
            <span style={{
              fontSize: '20px',
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
                transition: 'all 0.2s ease'
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
