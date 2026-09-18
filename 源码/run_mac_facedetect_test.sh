#!/bin/zsh
cd "/Users/emmaliu/Desktop/HKUST/RBM-project/Model/momask-codes/源码"
source .venv-emotion-arm/bin/activate
echo "==== FaceDetect Mac one-click test ===="
echo "1) Relax face for calibration"
echo "2) Smile / surprised / unhappy"
echo "3) Watch for: stable happy -> would plan wave"
echo "Window opens; press q to quit. Auto-stop in 60s."
echo
python facedetect_mac_demo.py --duration 60
echo
echo "==== Done. Exit code: $? ===="
read -n 1 -s -r -p "Press any key to close..."
