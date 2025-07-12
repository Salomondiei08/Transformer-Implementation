# Transformer Tasks Web App

A beautiful, interactive web application for testing custom Transformer models on sequence tasks. Built with Next.js, Shadcn/ui, and dark mode design.

## 🚀 Features

- **🎨 Beautiful Dark Mode UI** - Modern, sleek interface with gradient backgrounds
- **🤖 Four Sequence Tasks** - Copy, Reverse, Sort, and Shift operations
- **⚡ Interactive Testing** - Real-time sequence input and prediction
- **📊 Visual Results** - Color-coded badges showing input, prediction, and expected output
- **🔄 Training Simulation** - Progress bar showing model training (simulated)
- **📱 Responsive Design** - Works perfectly on desktop and mobile

## 🎯 Available Tasks

### 📋 Copy (Easy)

- **Input**: `1 2 3 4 5 6 7 8`
- **Output**: `1 2 3 4 5 6 7 8`
- **Description**: Replicate the input sequence exactly

### 🔄 Reverse (Medium)

- **Input**: `1 2 3 4 5 6 7 8`
- **Output**: `8 7 6 5 4 3 2 1`
- **Description**: Reverse the order of the sequence

### 🔢 Sort (Hard)

- **Input**: `8 3 1 6 4 7 2 5`
- **Output**: `1 2 3 4 5 6 7 8`
- **Description**: Sort the numbers in ascending order

### ➡️ Shift (Hard)

- **Input**: `1 2 3 4 5 6 7 8`
- **Output**: `8 1 2 3 4 5 6 7`
- **Description**: Shift the sequence by one position

## 🛠️ Tech Stack

- **Frontend**: Next.js 14, TypeScript, Tailwind CSS
- **UI Components**: Shadcn/ui
- **Styling**: Dark mode with gradient backgrounds
- **State Management**: React hooks
- **API**: Next.js API routes (ready for Python backend integration)

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

1. **Navigate to the web app directory:**

   ```bash
   cd transformer-web
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Run the development server:**

   ```bash
   npm run dev
   ```

4. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

## 🎮 How to Use

1. **Select a Task**: Choose from Copy, Reverse, Sort, or Shift from the dropdown
2. **Enter Sequence**: Type 8 numbers (1-9) separated by spaces
3. **Get Prediction**: Click "Get Prediction" to see the model's output
4. **Compare Results**: View input, prediction, and expected output side by side
5. **Train Model**: Click "Train Model" to simulate training (optional)

## 🔧 Configuration

### Input Validation

- Must be exactly 8 numbers
- Numbers must be between 1-9
- Separated by spaces

### Example Inputs

```
1 2 3 4 5 6 7 8
9 8 7 6 5 4 3 2
5 3 7 1 9 2 8 4
```

## 🔗 Future Integration

The web app is designed to easily integrate with your Python Transformer backend:

1. **API Route**: `/api/predict` is ready to connect to Python models
2. **Backend**: Can call your trained models from `tasks/[task]/model_[task].pth`
3. **Real Training**: Replace simulation with actual model training calls

### Example Backend Integration

```typescript
// In handlePredict function
const response = await fetch('/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ task: selectedTask, sequence: inputSequence })
});
```

## 🎨 Customization

### Colors

Each task has its own color scheme:

- **Copy**: Green theme
- **Reverse**: Blue theme  
- **Sort**: Purple theme
- **Shift**: Orange theme

### Styling

- Dark mode with gradient backgrounds
- Glassmorphism card effects
- Responsive grid layout
- Smooth animations and transitions

## 📱 Responsive Design

The app works perfectly on:

- **Desktop**: Full 3-column layout
- **Tablet**: Responsive grid adjustments
- **Mobile**: Stacked single-column layout

## 🚀 Deployment

### Vercel (Recommended)

```bash
npm run build
# Deploy to Vercel
```

### Other Platforms

The app can be deployed to any platform that supports Next.js:

- Netlify
- Railway
- DigitalOcean App Platform
- AWS Amplify

## 🤝 Contributing

Feel free to contribute by:

- Adding new sequence tasks
- Improving the UI/UX
- Adding real backend integration
- Enhancing the training simulation

## 📄 License

This project is part of the Transformer implementation demo.

---

**Enjoy exploring the Transformer tasks! 🚀**
