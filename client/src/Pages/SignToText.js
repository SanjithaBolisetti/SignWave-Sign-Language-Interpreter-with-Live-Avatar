import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import { Hands } from "@mediapipe/hands";
import { speak } from "../Utils/speak";

function SignToText() {
  const webcamRef = useRef(null);
  const [model, setModel] = useState(null);
  const [outputText, setOutputText] = useState("");

  // Load ML model
  useEffect(() => {
    tf.loadLayersModel("/model/model.json")
      .then(setModel)
      .catch(() => console.log("Model not loaded yet"));
  }, []);

  // Setup MediaPipe
  useEffect(() => {
    const hands = new Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });

    hands.onResults(onResults);

    if (webcamRef.current) {
      const camera = new window.Camera(webcamRef.current.video, {
        onFrame: async () => {
          await hands.send({ image: webcamRef.current.video });
        },
        width: 640,
        height: 480,
      });
      camera.start();
    }
  }, [model]);

  const onResults = (results) => {
    if (!model || !results.multiHandLandmarks) return;

    const landmarks = results.multiHandLandmarks[0]
      .flatMap((p) => [p.x, p.y, p.z]);

    const input = tf.tensor([landmarks]);
    const prediction = model.predict(input);
    const index = prediction.argMax(1).dataSync()[0];

    const letter = String.fromCharCode(65 + index);
    setOutputText((prev) => prev + letter);
  };

  return (
    <div className="container text-center mt-5">
      <h2>Sign to Text & Voice</h2>

      <Webcam
        ref={webcamRef}
        mirrored
        style={{ width: 400, borderRadius: 12 }}
      />

      <textarea
        className="form-control mt-3"
        rows={3}
        value={outputText}
        readOnly
      />

      <button
        className="btn btn-primary mt-3"
        onClick={() => speak(outputText)}
      >
        Speak
      </button>
    </div>
  );
}

export default SignToText;
