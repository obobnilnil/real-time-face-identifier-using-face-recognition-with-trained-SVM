package models

import "time"

type InputFaceRecognitionDetails struct {
	Name       string    `json:"name"`
	CameraID   string    `json:"camera_id"`
	Timestamp  time.Time `json:"timestamp"`
	Embedding  []float64 `json:"embedding"`
	ImagePath  string    `json:"image_path"`
	Confidence float64   `json:"confidence,omitempty"`
}
