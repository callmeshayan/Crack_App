#!/bin/bash

# Script to deploy the trained model to Raspberry Pi
# Usage: ./deploy_model_to_pi.sh [pi_hostname] [pi_username]

# Default values
PI_HOST="${1:-raspberrypi.local}"
PI_USER="${2:-pi}"
MODEL_PATH="/Users/shayannaghashpour/Desktop/--/pipe_crack_ai/runs/detect/train_20260314_134701/weights/best.pt"
DEST_DIR="/home/$PI_USER/crack_app/models"

echo "================================================"
echo "Deploying Model to Raspberry Pi"
echo "================================================"
echo ""
echo "Source Model: $MODEL_PATH"
echo "Destination: $PI_USER@$PI_HOST:$DEST_DIR"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found at $MODEL_PATH"
    echo "   Please train your model first or update MODEL_PATH in this script"
    exit 1
fi

# Get model file size
MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
echo "Model size: $MODEL_SIZE"
echo ""

# Create models directory on Pi
echo "Creating models directory on Raspberry Pi..."
ssh "$PI_USER@$PI_HOST" "mkdir -p $DEST_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Error: Could not connect to Raspberry Pi"
    echo "   Check that:"
    echo "   1. Pi is powered on and connected to network"
    echo "   2. Hostname is correct: $PI_HOST"
    echo "   3. SSH is enabled on the Pi"
    exit 1
fi

# Copy model
echo "Copying model file..."
scp "$MODEL_PATH" "$PI_USER@$PI_HOST:$DEST_DIR/best.pt"

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to copy model file"
    exit 1
fi

echo ""
echo "✓ Model successfully copied!"
echo ""

# Update .env file on Pi
echo "Updating .env configuration..."
ssh "$PI_USER@$PI_HOST" "cd $DEST_DIR/../ && \
    sed -i 's|^MODEL_MODE=.*|MODEL_MODE=offline|' .env && \
    sed -i 's|^LOCAL_MODEL_PATH=.*|LOCAL_MODEL_PATH=$DEST_DIR/best.pt|' .env && \
    echo '✓ .env updated'"

echo ""
echo "================================================"
echo "Deployment Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. SSH into your Pi: ssh $PI_USER@$PI_HOST"
echo "2. Navigate to crack_app: cd ~/crack_app"
echo "3. Verify .env settings: cat .env"
echo "4. Run the app: python realtime_pi5_dual_web.py"
echo ""
echo "The app will now use your offline YOLO model!"
