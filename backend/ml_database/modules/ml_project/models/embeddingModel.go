package models

import "time"

type EmbeddingDetails struct {
	Name       string    `bson:"name"`
	CameraID   string    `bson:"camera_id"`
	Timestamp  time.Time `bson:"timestamp"`
	Embedding  []float64 `bson:"embedding"`
	ImagePath  string    `bson:"image_path"`
	Confidence float64   `bson:"confidence"`
}
