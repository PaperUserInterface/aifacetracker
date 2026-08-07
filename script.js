const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');

const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const emptyState = document.getElementById('emptyState');
const messageBox = document.getElementById('messageBox');
const statusBadge = document.getElementById('statusBadge');
const cameraStatus = document.getElementById('cameraStatus');
const faceStatus = document.getElementById('faceStatus');
const movementStatus = document.getElementById('movementStatus');
const handsDetected = document.getElementById('handsDetected');
const facesDetected = document.getElementById('facesDetected');
const fpsValue = document.getElementById('fpsValue');
const resolutionValue = document.getElementById('resolutionValue');

let camera = null;
let faceDetection = null;
let hands = null;
let pose = null;
let stream = null;
let isRunning = false;
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;
let latestFaceResults = null;
let latestHandsResults = null;
let latestPoseResults = null;
let lastFaceCenter = null;
let lastHandCenters = [];
let lastPoseCenter = null;
let movementTriggered = false;
let messageTimer = null;
let HAND_CONNECTIONS = [];
let POSE_CONNECTIONS = [];
let renderScaleX = 1;
let renderScaleY = 1;
let inferenceFrameCount = 0;

const faceColor = '#34d399';
const handColor = '#60a5fa';
const poseColor = '#fbbf24';
const processingCanvas = document.createElement('canvas');
const processingCtx = processingCanvas.getContext('2d');
const displayScale = { x: 1, y: 1 };

function setStatus(label, type, message) {
  statusBadge.textContent = label;
  statusBadge.className = `status-badge ${type}`;
  messageBox.textContent = message;
}

function updateValue(id, value) {
  document.getElementById(id).textContent = value;
}

function showEmptyState(show) {
  emptyState.style.display = show ? 'grid' : 'none';
}

function clearCanvas() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function releaseMediaStream() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  if (video.srcObject && video.srcObject !== stream) {
    video.srcObject.getTracks?.().forEach((track) => track.stop());
  }

  video.srcObject = null;
  stream = null;
}

function describeCameraError(error) {
  const message = error?.message || '';
  const lowerMessage = message.toLowerCase();

  if (error?.name === 'NotAllowedError' || error?.name === 'PermissionDeniedError') {
    return 'Camera permission was denied. Please allow camera access and try again.';
  }

  if (error?.name === 'NotFoundError') {
    return 'No webcam was found. Please connect a camera and try again.';
  }

  if (error?.name === 'NotReadableError' || lowerMessage.includes('in use') || lowerMessage.includes('busy')) {
    return 'The camera is already being used by another app or browser tab. Close that app and try again.';
  }

  if (error?.name === 'OverconstrainedError') {
    return 'The camera settings are unsupported by this browser.';
  }

  return 'Unable to access the camera. Please refresh the page and try again.';
}

function setIndicator(element, type) {
  element.className = `value ${type}`;
}

function setResolution() {
  const width = video.videoWidth || 0;
  const height = video.videoHeight || 0;
  resolutionValue.textContent = width && height ? `${width} × ${height}` : '--';
}

function updateFps() {
  frameCount += 1;
  const now = performance.now();
  const elapsed = now - lastFrameTime;
  if (elapsed >= 500) {
    fps = Math.round((frameCount * 1000) / elapsed);
    frameCount = 0;
    lastFrameTime = now;
  }
  fpsValue.textContent = fps || 0;
}

function drawFaceBoxes(results) {
  if (!results || !results.detections) {
    return;
  }

  const detections = results.detections || [];
  facesDetected.textContent = detections.length;

  if (detections.length > 0) {
    setIndicator(faceStatus, 'success');
    faceStatus.textContent = 'Face Detected';
    setStatus('Tracking', 'success', 'Head and hands are being tracked.');
  } else {
    setIndicator(faceStatus, 'danger');
    faceStatus.textContent = 'No Face';
  }

  detections.forEach((det, index) => {
    // Try to use landmarks if available for better positioning
    if (det.landmarks && det.landmarks.length > 0) {
      const landmarks = det.landmarks;
      
      // Draw facial expression tracking
      drawFacialExpressions(landmarks);
      
      // Get all x and y coordinates from landmarks
      const xs = landmarks.map(lm => lm.x);
      const ys = landmarks.map(lm => lm.y);
      
      // Calculate bounding box from landmarks with padding
      const xMin = Math.min(...xs) - 0.05;
      const yMin = Math.min(...ys) - 0.1;
      const xMax = Math.max(...xs) + 0.05;
      const yMax = Math.max(...ys) + 0.05;
      
      const width = xMax - xMin;
      const height = yMax - yMin;
      
      // Mirror horizontally and convert to canvas coordinates
      const x = (1 - xMax) * overlay.width;
      const y = yMin * overlay.height;
      const w = width * overlay.width;
      const h = height * overlay.height;
      
      // Draw rectangle around face
      ctx.strokeStyle = faceColor;
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, w, h);

      // Fill with semi-transparent color
      ctx.fillStyle = 'rgba(52, 211, 153, 0.14)';
      ctx.fillRect(x, y, w, h);

      // Draw label
      ctx.font = '16px Segoe UI';
      ctx.fillStyle = faceColor;
      ctx.fillText('Face', x + 6, Math.max(20, y - 10));
      
    } else {
      // Fallback to bounding box approach
      const box = det.boundingBox;
      if (!box) return;

      let xMin = box.xMin ?? 0;
      let yMin = box.yMin ?? 0;
      let width = box.width ?? 0;
      let height = box.height ?? 0;

      // Mirror horizontally and convert to canvas coordinates
      const x = (1 - xMin - width) * overlay.width;
      const y = yMin * overlay.height;
      const w = width * overlay.width;
      const h = height * overlay.height;

      // Draw rectangle around face
      ctx.strokeStyle = faceColor;
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, w, h);

      // Fill with semi-transparent color
      ctx.fillStyle = 'rgba(52, 211, 153, 0.14)';
      ctx.fillRect(x, y, w, h);

      // Draw label
      ctx.font = '16px Segoe UI';
      ctx.fillStyle = faceColor;
      ctx.fillText('Face', x + 6, Math.max(20, y - 10));
    }
  });
}

function drawFacialExpressions(landmarks) {
  if (!landmarks || landmarks.length < 6) return;
  
  // MediaPipe Face Detection provides 6 key landmarks:
  // 0: right eye, 1: left eye, 2: nose tip, 3: mouth center, 4: right ear, 5: left ear
  
  const expressionColors = {
    eyes: '#00ff88',      // Green
    mouth: '#4ecdc4',     // Teal
    nose: '#ffd93d',      // Yellow
    ears: '#ff6b6b'       // Red
  };

  // Convert landmarks to screen coordinates
  const points = landmarks.map((point) => ({
    x: (1 - point.x) * overlay.width,
    y: point.y * overlay.height,
  }));

  // Draw facial feature landmarks
  function drawFeatureLandmark(pointIndex, color, label, size = 6) {
    if (!points[pointIndex]) return;
    
    const point = points[pointIndex];
    
    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    
    // Draw landmark circle
    ctx.beginPath();
    ctx.arc(point.x, point.y, size, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    
    // Draw label
    ctx.font = '12px Segoe UI';
    ctx.fillText(label, point.x + 10, point.y - 10);
    
    ctx.restore();
  }

  // Draw individual facial features
  drawFeatureLandmark(0, expressionColors.eyes, 'R-EYE', 4);
  drawFeatureLandmark(1, expressionColors.eyes, 'L-EYE', 4);
  drawFeatureLandmark(2, expressionColors.nose, 'NOSE', 4);
  drawFeatureLandmark(3, expressionColors.mouth, 'MOUTH', 5);
  drawFeatureLandmark(4, expressionColors.ears, 'R-EAR', 3);
  drawFeatureLandmark(5, expressionColors.ears, 'L-EAR', 3);

  // Draw connections between key features
  ctx.save();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.lineWidth = 1;
  
  // Eye to eye connection
  if (points[0] && points[1]) {
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    ctx.lineTo(points[1].x, points[1].y);
    ctx.stroke();
  }
  
  // Nose to mouth connection
  if (points[2] && points[3]) {
    ctx.beginPath();
    ctx.moveTo(points[2].x, points[2].y);
    ctx.lineTo(points[3].x, points[3].y);
    ctx.stroke();
  }
  
  ctx.restore();

  // Simple expression detection based on landmark positions
  if (points.length >= 4) {
    const eyeDistance = Math.abs(points[0].x - points[1].x);
    const noseToMouthDistance = Math.abs(points[2].y - points[3].y);
    
    // Display expression status
    ctx.font = '14px Segoe UI';
    let yOffset = 50;
    
    // Simple "expression active" indicator
    if (eyeDistance > 50) {
      ctx.fillStyle = expressionColors.eyes;
      ctx.fillText('👁️ EYES ACTIVE', 10, yOffset);
      yOffset += 20;
    }
    
    if (noseToMouthDistance < 30) {
      ctx.fillStyle = expressionColors.mouth;
      ctx.fillText('😊 EXPRESSION', 10, yOffset);
      yOffset += 20;
    }
    
    // Show landmark count
    ctx.fillStyle = '#ffffff';
    ctx.fillText(`📍 ${landmarks.length} LANDMARKS`, 10, yOffset);
  }
}

function drawHands(results) {
  if (!results || !results.multiHandLandmarks) {
    handsDetected.textContent = 0;
    return;
  }

  const handsList = results.multiHandLandmarks || [];
  handsDetected.textContent = handsList.length;

  handsList.forEach((landmarks, index) => {
    renderSkeleton(landmarks, handColor, index + 1);
  });
}

function drawPose(results) {
  if (!results || !results.poseLandmarks) {
    return;
  }

  const poseLandmarks = results.poseLandmarks;
  
  // Draw exoskeleton skeletons for body parts (same style as hands)
  drawBodyPartExoskeletons(poseLandmarks);
  
  // Still draw the skeleton lines and points
  const posePoints = [
    poseLandmarks[0], poseLandmarks[1], poseLandmarks[2], poseLandmarks[3], poseLandmarks[4], poseLandmarks[5],
    poseLandmarks[6], poseLandmarks[7], poseLandmarks[8], poseLandmarks[9], poseLandmarks[10], poseLandmarks[11],
    poseLandmarks[12], poseLandmarks[13], poseLandmarks[14], poseLandmarks[15], poseLandmarks[16], poseLandmarks[17],
    poseLandmarks[18], poseLandmarks[19], poseLandmarks[20], poseLandmarks[23], poseLandmarks[24], poseLandmarks[25],
    poseLandmarks[26], poseLandmarks[27], poseLandmarks[28], poseLandmarks[29], poseLandmarks[30]
  ].filter(Boolean);

  if (posePoints.length >= 3) {
    renderSkeleton(posePoints, poseColor, 'Pose');
  }
}

function drawBodyPartExoskeletons(landmarks) {
  if (!landmarks || landmarks.length < 24) return;

  // Define body part colors
  const neckColor = '#ff6b6b';     // Red
  const chestColor = '#4ecdc4';    // Teal  
  const leftArmColor = '#45b7d1';  // Blue
  const rightArmColor = '#96ceb4'; // Green
  
  // Helper function to render body part skeleton like hands
  function renderBodyPartSkeleton(landmarkIndices, connections, color, label) {
    if (!landmarkIndices || landmarkIndices.length === 0) return;
    
    // Get the landmarks and convert to screen coordinates
    const points = landmarkIndices
      .map(index => landmarks[index])
      .filter(Boolean)
      .map((point) => ({
        x: (1 - point.x) * overlay.width,
        y: point.y * overlay.height,
      }));

    if (points.length === 0) return;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;

    // Draw connections between landmarks
    if (connections && connections.length > 0) {
      drawConnections(points, connections, color, 3);
    }
    
    // Draw landmark points
    drawPointMarkers(points, color, 4);
    
    // Draw label near the first point
    if (points[0]) {
      const labelX = points[0].x;
      const labelY = points[0].y - 15;
      ctx.font = '13px Segoe UI';
      ctx.fillText(label, labelX, labelY);
    }

    ctx.restore();
  }

  // LEFT ARM - shoulder to hand with connections like hand skeleton
  const leftArmIndices = [11, 13, 15, 17, 19, 21]; // shoulder -> elbow -> wrist -> hand landmarks
  const leftArmConnections = [[0, 1], [1, 2], [2, 3], [2, 4], [2, 5]]; // shoulder-elbow-wrist-fingers
  renderBodyPartSkeleton(leftArmIndices, leftArmConnections, leftArmColor, 'L-ARM');

  // RIGHT ARM - shoulder to hand with connections like hand skeleton
  const rightArmIndices = [12, 14, 16, 18, 20, 22]; // shoulder -> elbow -> wrist -> hand landmarks
  const rightArmConnections = [[0, 1], [1, 2], [2, 3], [2, 4], [2, 5]]; // shoulder-elbow-wrist-fingers
  renderBodyPartSkeleton(rightArmIndices, rightArmConnections, rightArmColor, 'R-ARM');

  // CHEST/TORSO - shoulder and hip connections with skeleton style
  const chestIndices = [11, 12, 23, 24]; // shoulders and hips
  const chestConnections = [[0, 1], [0, 2], [1, 3], [2, 3]]; // shoulder-shoulder, shoulder-hip connections
  renderBodyPartSkeleton(chestIndices, chestConnections, chestColor, 'CHEST');

  // NECK/HEAD - head landmarks with connections like hand skeleton
  const neckIndices = [0, 7, 8, 9, 10]; // nose, ears, mouth corners
  const neckConnections = [[0, 1], [0, 2], [0, 3], [0, 4]]; // nose to facial features
  renderBodyPartSkeleton(neckIndices, neckConnections, neckColor, 'HEAD');
}

function drawConnections(points, connections, color, lineWidth) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  connections.forEach(([start, end]) => {
    const startPoint = points[start];
    const endPoint = points[end];
    if (!startPoint || !endPoint) {
      return;
    }

    ctx.beginPath();
    ctx.moveTo(startPoint.x, startPoint.y);
    ctx.lineTo(endPoint.x, endPoint.y);
    ctx.stroke();
  });
}

function drawPointMarkers(points, color, radius) {
  ctx.fillStyle = color;
  points.forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
  });
}

function renderSkeleton(landmarks, color, label) {
  if (!landmarks || !landmarks.length) {
    return;
  }

  const points = landmarks.map((point) => ({
    x: (1 - point.x) * overlay.width,
    y: point.y * overlay.height,
  }));

  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;

  if (label === 'Pose') {
    drawConnections(points, POSE_CONNECTIONS, color, 3);
    drawPointMarkers(points, color, 4);
  } else {
    drawConnections(points, HAND_CONNECTIONS, color, 2);
    drawPointMarkers(points, color, 3);
    const labelX = points[0].x;
    const labelY = points[0].y - 12;
    ctx.font = '13px Segoe UI';
    ctx.fillText(`Hand ${label}`, labelX, labelY);
  }

  ctx.restore();
}

function detectMovement() {
  const faceCenter = latestFaceResults?.detections?.[0]?.boundingBox
    ? {
        x: latestFaceResults.detections[0].boundingBox.originX + latestFaceResults.detections[0].boundingBox.width / 2,
        y: latestFaceResults.detections[0].boundingBox.originY + latestFaceResults.detections[0].boundingBox.height / 2,
      }
    : null;

  const poseCenter = latestPoseResults?.poseLandmarks?.length
    ? latestPoseResults.poseLandmarks[0]
    : null;

  const handCenters = (latestHandsResults?.multiHandLandmarks || []).map((landmarks) => {
    const first = landmarks[0];
    return first ? { x: first.x, y: first.y } : null;
  }).filter(Boolean);

  let moved = false;

  if (faceCenter && lastFaceCenter) {
    moved = moved || Math.hypot(faceCenter.x - lastFaceCenter.x, faceCenter.y - lastFaceCenter.y) > 0.02;
  }
  if (poseCenter && lastPoseCenter) {
    moved = moved || Math.hypot(poseCenter.x - lastPoseCenter.x, poseCenter.y - lastPoseCenter.y) > 0.02;
  }

  if (handCenters.length && lastHandCenters.length) {
    for (let index = 0; index < Math.min(handCenters.length, lastHandCenters.length); index += 1) {
      const current = handCenters[index];
      const previous = lastHandCenters[index];
      if (Math.hypot(current.x - previous.x, current.y - previous.y) > 0.02) {
        moved = true;
        break;
      }
    }
  }

  lastFaceCenter = faceCenter;
  lastPoseCenter = poseCenter;
  lastHandCenters = handCenters;

  movementTriggered = moved;
  if (movementTriggered) {
    setIndicator(movementStatus, 'success');
    movementStatus.textContent = 'Tracking Movement';
  } else if (latestFaceResults?.detections?.length || latestHandsResults?.multiHandLandmarks?.length || latestPoseResults?.poseLandmarks?.length) {
    setIndicator(movementStatus, 'warning');
    movementStatus.textContent = 'Stable';
  } else {
    setIndicator(movementStatus, 'warning');
    movementStatus.textContent = 'No Motion';
  }
}

function drawScene() {
  if (!isRunning) {
    return;
  }

  if (video.readyState >= 2) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.drawImage(video, 0, 0, overlay.width, overlay.height);
    drawFaceBoxes(latestFaceResults);
    drawHands(latestHandsResults);
    drawPose(latestPoseResults);
    detectMovement();
    updateFps();
    setResolution();
  }

  requestAnimationFrame(drawScene);
}

async function initializeMediaPipe() {
  try {
    const [faceDetectionModule, handsModule, poseModule, cameraUtilsModule] = await Promise.all([
      import('https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/+esm'),
      import('https://cdn.jsdelivr.net/npm/@mediapipe/hands/+esm'),
      import('https://cdn.jsdelivr.net/npm/@mediapipe/pose/+esm'),
      import('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/+esm'),
    ]);

    const FaceDetection = faceDetectionModule.FaceDetection || faceDetectionModule.default?.FaceDetection;
    const Hands = handsModule.Hands || handsModule.default?.Hands;
    const Pose = poseModule.Pose || poseModule.default?.Pose;
    const Camera = cameraUtilsModule.Camera || cameraUtilsModule.default?.Camera;
    HAND_CONNECTIONS = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [5, 9], [9, 10], [10, 11], [11, 12],
      [9, 13], [13, 14], [14, 15], [15, 16],
      [0, 17], [17, 18], [18, 19], [19, 20],
      [5, 9], [9, 13], [13, 17], [17, 0]
    ];
    POSE_CONNECTIONS = [
      [11, 12], [11, 13], [13, 15], [15, 17], [17, 19],
      [12, 14], [14, 16], [16, 18], [18, 20], [11, 23],
      [12, 24], [23, 24], [23, 25], [24, 26], [25, 27],
      [26, 28], [27, 29], [28, 30], [11, 24], [12, 23]
    ];

    faceDetection = new FaceDetection({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`,
    });
    faceDetection.setOptions({
      selfieMode: true,
      model: 'short',
      minDetectionConfidence: 0.5,
    });
    faceDetection.onResults((results) => {
      latestFaceResults = results;
    });

    hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    hands.setOptions({
      selfieMode: true,
      maxNumHands: 10,
      modelComplexity: 0,
      minDetectionConfidence: 0.45,
      minTrackingConfidence: 0.45,
    });
    hands.onResults((results) => {
      latestHandsResults = results;
    });

    pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });
    pose.setOptions({
      selfieMode: true,
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.55,
      minTrackingConfidence: 0.55,
    });
    pose.onResults((results) => {
      latestPoseResults = results;
    });

    return { Camera };
  } catch (error) {
    console.error(error);
    throw new Error('MediaPipe failed to load. Please refresh and try again.');
  }
}

function resetUi() {
  clearCanvas();
  showEmptyState(true);
  cameraStatus.textContent = 'Idle';
  setIndicator(cameraStatus, 'warning');
  faceStatus.textContent = 'No Face';
  setIndicator(faceStatus, 'warning');
  movementStatus.textContent = 'Waiting';
  setIndicator(movementStatus, 'warning');
  handsDetected.textContent = '0';
  facesDetected.textContent = '0';
  fpsValue.textContent = '0';
  resolutionValue.textContent = '--';
  latestFaceResults = null;
  latestHandsResults = null;
  latestPoseResults = null;
  lastFaceCenter = null;
  lastHandCenters = [];
  lastPoseCenter = null;
  movementTriggered = false;
}

async function startCamera() {
  if (isRunning) {
    return;
  }

  try {
    if (camera) {
      camera.stop();
      camera = null;
    }

    releaseMediaStream();
    resetUi();
    setStatus('Loading', 'warning', 'Requesting camera access and loading MediaPipe models...');
    showEmptyState(true);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('This browser does not support camera access.');
    }

    const streamResponse = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' },
      audio: false,
    });

    stream = streamResponse;
    video.srcObject = stream;
    await video.play();

    const { Camera } = await initializeMediaPipe();

    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    processingCanvas.width = overlay.width;
    processingCanvas.height = overlay.height;
    displayScale.x = overlay.width / processingCanvas.width;
    displayScale.y = overlay.height / processingCanvas.height;
    setResolution();

    camera = new Camera(video, {
      onFrame: async () => {
        if (!video.videoWidth || !video.videoHeight) {
          return;
        }

        processingCtx.clearRect(0, 0, processingCanvas.width, processingCanvas.height);
        processingCtx.drawImage(video, 0, 0, processingCanvas.width, processingCanvas.height);

        inferenceFrameCount += 1;
        const runHands = inferenceFrameCount % 2 === 0;
        const runPose = inferenceFrameCount % 3 === 0;

        if (faceDetection) {
          await faceDetection.send({ image: processingCanvas });
        }
        if (hands && runHands) {
          await hands.send({ image: processingCanvas });
        }
        if (pose && runPose) {
          await pose.send({ image: processingCanvas });
        }
      },
      width: overlay.width,
      height: overlay.height,
    });

    await camera.start();
    isRunning = true;
    showEmptyState(false);
    cameraStatus.textContent = 'Camera Ready';
    setIndicator(cameraStatus, 'success');
    setStatus('Live', 'success', 'Camera is active. MediaPipe is tracking your face and hands.');
    drawScene();
  } catch (error) {
    console.error(error);
    isRunning = false;
    if (camera) {
      camera.stop();
      camera = null;
    }
    releaseMediaStream();
    setStatus('Error', 'danger', describeCameraError(error));
    cameraStatus.textContent = 'Camera Error';
    setIndicator(cameraStatus, 'danger');
    faceStatus.textContent = 'No Face';
    setIndicator(faceStatus, 'warning');
    movementStatus.textContent = 'Waiting';
    setIndicator(movementStatus, 'warning');
    showEmptyState(true);
  }
}

function stopCamera() {
  if (!isRunning && !stream) {
    resetUi();
    setStatus('Stopped', 'warning', 'Camera stopped. Press Start Camera to continue.');
    return;
  }

  isRunning = false;
  if (camera) {
    camera.stop();
  }
  releaseMediaStream();
  clearCanvas();
  if (camera) {
    camera = null;
  }
  faceDetection = null;
  hands = null;
  pose = null;
  resetUi();
  setStatus('Stopped', 'warning', 'Camera stopped. Resources were released.');
  cameraStatus.textContent = 'Stopped';
  setIndicator(cameraStatus, 'warning');
  showEmptyState(true);
}

startCameraBtn.addEventListener('click', startCamera);
stopCameraBtn.addEventListener('click', stopCamera);
window.addEventListener('beforeunload', stopCamera);
