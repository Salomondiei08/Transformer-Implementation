import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
    try {
        const { task, sequence } = await request.json();

        // Validate input
        if (!task || !sequence) {
            return NextResponse.json(
                { error: 'Missing task or sequence' },
                { status: 400 }
            );
        }

        const numbers = sequence.trim().split(/\s+/).map(Number);

        if (numbers.length !== 8) {
            return NextResponse.json(
                { error: 'Sequence must contain exactly 8 numbers' },
                { status: 400 }
            );
        }

        if (!numbers.every(num => num >= 1 && num <= 9)) {
            return NextResponse.json(
                { error: 'Numbers must be between 1 and 9' },
                { status: 400 }
            );
        }

        // For now, simulate the prediction
        // In a real implementation, this would call your Python backend
        let result: number[] = [];

        switch (task) {
            case 'copy':
                result = [...numbers];
                break;
            case 'reverse':
                result = [...numbers].reverse();
                break;
            case 'sort':
                result = [...numbers].sort((a, b) => a - b);
                break;
            case 'shift':
                result = [numbers[numbers.length - 1], ...numbers.slice(0, -1)];
                break;
            default:
                return NextResponse.json(
                    { error: 'Invalid task' },
                    { status: 400 }
                );
        }

        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 500));

        return NextResponse.json({
            task,
            input: numbers,
            prediction: result,
            success: true
        });

    } catch (error) {
        console.error('Prediction error:', error);
        return NextResponse.json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
} 