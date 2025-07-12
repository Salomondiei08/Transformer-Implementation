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
  Copy
} from "lucide-react";

const tasks = [
  {
    id: "copy",
    name: "Copy",
    description: "Copy the input sequence exactly",
    icon: Copy,
    examples: ["hello → hello", "123 → 123", "abc → abc"]
  },
  {
    id: "reverse",
    name: "Reverse",
    description: "Reverse the input sequence",
    icon: ArrowRight,
    examples: ["hello → olleh", "123 → 321", "abc → cba"]
  },
  {
    id: "sort",
    name: "Sort",
    description: "Sort characters in ascending order",
    icon: BarChart3,
    examples: ["hello → ehllo", "321 → 123", "cba → abc"]
  },
  {
    id: "shift",
    name: "Shift",
    description: "Shift each character by +1 position",
    icon: Zap,
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
    <div style={{ minHeight: '100vh', padding: '24px', backgroundColor: 'black' }}>
      <div style={{ maxWidth: '1280px', margin: '0 auto', marginBottom: '32px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
          <div style={{ padding: '8px', borderRadius: '8px', backgroundColor: '#3b82f6' }}>
            <Brain style={{ height: '24px', width: '24px', color: 'white' }} />
          </div>
          <h1 style={{ fontSize: '30px', fontWeight: 'bold', color: 'white' }}>Transformer Playground</h1>
        </div>
        <p style={{ color: '#9ca3af', fontSize: '18px' }}>
          Interactive playground for testing custom Transformer models on sequence tasks
        </p>
      </div>

      <div style={{ maxWidth: '1280px', margin: '0 auto', display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '32px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <Card style={{ backgroundColor: '#1f2937', borderColor: '#374151' }}>
            <CardHeader>
              <CardTitle style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'white' }}>
                <Target style={{ height: '20px', width: '20px', color: '#60a5fa' }} />
                Select Task
              </CardTitle>
              <CardDescription style={{ color: '#9ca3af' }}>
                Choose a sequence transformation task to test
              </CardDescription>
            </CardHeader>
            <CardContent style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <Select value={selectedTask} onValueChange={setSelectedTask}>
                <SelectTrigger style={{ width: '100%' }}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {tasks.map((task) => (
                    <SelectItem key={task.id} value={task.id}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <task.icon style={{ height: '16px', width: '16px' }} />
                        {task.name}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {selectedTaskData && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <p style={{ fontSize: '14px', color: '#9ca3af' }}>
                    {selectedTaskData.description}
                  </p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <Label style={{ fontSize: '12px', fontWeight: '500', color: '#9ca3af' }}>Examples:</Label>
                    {selectedTaskData.examples.map((example, index) => (
                      <div key={index} style={{ fontSize: '12px', fontFamily: 'monospace', backgroundColor: '#4b5563', padding: '8px', borderRadius: '4px' }}>
                        {example}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card style={{ backgroundColor: '#1f2937', borderColor: '#374151' }}>
            <CardHeader>
              <CardTitle style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'white' }}>
                <Settings style={{ height: '20px', width: '20px', color: '#34d399' }} />
                Model Training
              </CardTitle>
              <CardDescription style={{ color: '#9ca3af' }}>
                Monitor training progress and model status
              </CardDescription>
            </CardHeader>
            <CardContent style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '14px' }}>
                  <span style={{ color: '#9ca3af' }}>Training Progress</span>
                  <span style={{ fontFamily: 'monospace', color: 'white' }}>{Math.round(trainingProgress)}%</span>
                </div>
                <Progress value={trainingProgress} style={{ height: '8px' }} />
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: isTraining ? '#fbbf24' : '#34d399' }} />
                <span style={{ color: '#9ca3af' }}>
                  {isTraining ? 'Training in progress...' : 'Model ready'}
                </span>
              </div>

              <Button 
                onClick={startTraining} 
                disabled={isTraining}
                style={{ width: '100%' }}
                variant="outline"
              >
                <Play style={{ height: '16px', width: '16px', marginRight: '8px' }} />
                {isTraining ? 'Training...' : 'Start Training'}
              </Button>
            </CardContent>
          </Card>

          <Card style={{ backgroundColor: '#1f2937', borderColor: '#374151' }}>
            <CardHeader>
              <CardTitle style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'white' }}>
                <BarChart3 style={{ height: '20px', width: '20px', color: '#a78bfa' }} />
                Model Statistics
              </CardTitle>
            </CardHeader>
            <CardContent style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#60a5fa' }}>12M</div>
                  <div style={{ fontSize: '12px', color: '#9ca3af' }}>Parameters</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#34d399' }}>99.8%</div>
                  <div style={{ fontSize: '12px', color: '#9ca3af' }}>Accuracy</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#a78bfa' }}>8</div>
                  <div style={{ fontSize: '12px', color: '#9ca3af' }}>Layers</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#fb923c' }}>512</div>
                  <div style={{ fontSize: '12px', color: '#9ca3af' }}>Embedding</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <Card style={{ backgroundColor: '#1f2937', borderColor: '#374151' }}>
            <CardHeader>
              <CardTitle style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'white' }}>
                <Code style={{ height: '20px', width: '20px', color: '#60a5fa' }} />
                Input Sequence
              </CardTitle>
              <CardDescription style={{ color: '#9ca3af' }}>
                Enter a sequence to transform using the selected task
              </CardDescription>
            </CardHeader>
            <CardContent style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <Label htmlFor="input" style={{ color: 'white' }}>Input Text</Label>
                <Input
                  id="input"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Enter your sequence here..."
                  style={{ fontFamily: 'monospace' }}
                />
              </div>
              
              <Button 
                onClick={handlePredict} 
                disabled={isLoading || !input.trim()}
                style={{ width: '100%' }}
                size="lg"
              >
                {isLoading ? (
                  <>
                    <div style={{ animation: 'spin 1s linear infinite', borderRadius: '50%', height: '16px', width: '16px', border: '2px solid transparent', borderBottomColor: 'white', marginRight: '8px' }} />
                    Processing...
                  </>
                ) : (
                  <>
                    <Sparkles style={{ height: '16px', width: '16px', marginRight: '8px' }} />
                    Generate Prediction
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          <Card style={{ backgroundColor: '#1f2937', borderColor: '#374151' }}>
            <CardHeader>
              <CardTitle style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'white' }}>
                <CheckCircle style={{ height: '20px', width: '20px', color: '#34d399' }} />
                Model Prediction
              </CardTitle>
              <CardDescription style={{ color: '#9ca3af' }}>
                The transformed sequence based on your input and selected task
              </CardDescription>
            </CardHeader>
            <CardContent style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {prediction ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <Label style={{ fontSize: '14px', color: '#9ca3af' }}>Input</Label>
                    <div style={{ padding: '12px', backgroundColor: '#4b5563', borderRadius: '8px', fontFamily: 'monospace', fontSize: '14px', border: '1px solid #6b7280' }}>
                      {input}
                    </div>
                  </div>
                  
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <ArrowRight style={{ height: '20px', width: '20px', color: '#9ca3af' }} />
                  </div>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <Label style={{ fontSize: '14px', color: '#9ca3af' }}>Output</Label>
                    <div style={{ padding: '12px', backgroundColor: 'rgba(34, 197, 94, 0.1)', borderRadius: '8px', fontFamily: 'monospace', fontSize: '14px', border: '1px solid rgba(34, 197, 94, 0.3)' }}>
                      {prediction}
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Badge variant="secondary" style={{ backgroundColor: 'rgba(34, 197, 94, 0.2)', color: '#34d399', border: '1px solid rgba(34, 197, 94, 0.3)' }}>
                      <CheckCircle style={{ height: '12px', width: '12px', marginRight: '4px' }} />
                      Success
                    </Badge>
                    <span style={{ fontSize: '12px', color: '#9ca3af' }}>
                      Prediction completed in 1.2s
                    </span>
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '32px 0' }}>
                  <div style={{ width: '64px', height: '64px', margin: '0 auto', marginBottom: '16px', borderRadius: '50%', backgroundColor: '#4b5563', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Sparkles style={{ height: '32px', width: '32px', color: '#9ca3af' }} />
                  </div>
                  <p style={{ color: '#9ca3af' }}>
                    Enter input and click "Generate Prediction" to see results
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card style={{ backgroundColor: '#1f2937', borderColor: '#374151' }}>
            <CardHeader>
              <CardTitle style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'white' }}>
                <Target style={{ height: '20px', width: '20px', color: '#a78bfa' }} />
                Quick Examples
              </CardTitle>
              <CardDescription style={{ color: '#9ca3af' }}>
                Try these examples to test different tasks
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="copy" style={{ width: '100%' }}>
                <TabsList style={{ display: 'grid', width: '100%', gridTemplateColumns: 'repeat(4, 1fr)' }}>
                  {tasks.map((task) => (
                    <TabsTrigger key={task.id} value={task.id} style={{ fontSize: '12px' }}>
                      {task.name}
                    </TabsTrigger>
                  ))}
                </TabsList>
                {tasks.map((task) => (
                  <TabsContent key={task.id} value={task.id} style={{ marginTop: '16px' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {task.examples.map((example, index) => (
                        <button
                          key={index}
                          onClick={() => {
                            setSelectedTask(task.id);
                            setInput(example.split(' → ')[0]);
                            setPrediction('');
                          }}
                          style={{ width: '100%', textAlign: 'left', padding: '12px', borderRadius: '8px', backgroundColor: '#4b5563', border: '1px solid #6b7280', transition: 'background-color 0.2s' }}
                          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#6b7280'}
                          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#4b5563'}
                        >
                          <div style={{ fontFamily: 'monospace', fontSize: '14px', color: 'white' }}>{example}</div>
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
  );
}
