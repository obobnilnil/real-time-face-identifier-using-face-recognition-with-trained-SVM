package repositories

import (
	"context"
	"log"
	"ml_database/modules/ml_project/models"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
)

type RepositoryPort interface {
	InputFaceRecognitionDetailsRepositories(inputFaceRecognition models.InputFaceRecognitionDetails) error
	GetImagesByTimeRangeRepositories(cameraID string, baseTime time.Time) ([]string, error)
	GetImagePathsByNameAndTimeRepositories(name string, baseTime *time.Time, isAll bool) ([]string, error)
	FetchEmbeddingDetailsForModelTrainerRepositories(ctx context.Context, name string) ([]*models.EmbeddingDetails, error)
}

type repositoryAdapter struct {
	collection          *mongo.Collection
	embeddingCollection *mongo.Collection
}

func NewRepositoryAdapter(mongoDB *mongo.Database) RepositoryPort {
	return &repositoryAdapter{collection: mongoDB.Collection("face_logs"),
		embeddingCollection: mongoDB.Collection("face_logs")}
}

func (r *repositoryAdapter) InputFaceRecognitionDetailsRepositories(inputFaceRecognition models.InputFaceRecognitionDetails) error {
	doc := bson.M{
		"name":       inputFaceRecognition.Name,
		"camera_id":  inputFaceRecognition.CameraID,
		"timestamp":  inputFaceRecognition.Timestamp,
		"embedding":  inputFaceRecognition.Embedding,
		"image_path": inputFaceRecognition.ImagePath,
		"confidence": inputFaceRecognition.Confidence,
	}

	_, err := r.collection.InsertOne(context.TODO(), doc)
	if err != nil {
		log.Println("[ERROR] InsertOne failed:", err)
		return err
	}

	log.Println("[INFO] Face data inserted into MongoDB")
	return nil
}

func (r *repositoryAdapter) GetImagesByTimeRangeRepositories(cameraID string, baseTime time.Time) ([]string, error) {
	startTime := baseTime.Add(-1 * time.Hour)
	endTime := baseTime.Add(1 * time.Hour)

	filter := bson.M{
		"camera_id": cameraID,
		"timestamp": bson.M{
			"$gte": startTime,
			"$lt":  endTime,
		},
	}

	cursor, err := r.collection.Find(context.TODO(), filter)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(context.TODO())

	var paths []string
	for cursor.Next(context.TODO()) {
		var doc struct {
			ImagePath string `bson:"image_path"`
		}
		if err := cursor.Decode(&doc); err != nil {
			return nil, err
		}
		paths = append(paths, doc.ImagePath)
	}

	return paths, nil
}

func (r *repositoryAdapter) GetImagePathsByNameAndTimeRepositories(name string, baseTime *time.Time, isAll bool) ([]string, error) {
	filter := bson.M{
		"name": name,
	}

	if !isAll && baseTime != nil {
		startTime := baseTime.Add(-1 * time.Hour)
		endTime := baseTime.Add(1 * time.Hour)

		filter["timestamp"] = bson.M{
			"$gte": startTime,
			"$lt":  endTime,
		}
	}

	cursor, err := r.collection.Find(context.TODO(), filter)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(context.TODO())

	var paths []string
	for cursor.Next(context.TODO()) {
		var doc struct {
			ImagePath string `bson:"image_path"`
		}
		if err := cursor.Decode(&doc); err != nil {
			continue
		}
		paths = append(paths, doc.ImagePath)
	}

	return paths, nil
}

func (r *repositoryAdapter) FetchEmbeddingDetailsForModelTrainerRepositories(ctx context.Context, name string) ([]*models.EmbeddingDetails, error) {
	var results []*models.EmbeddingDetails
	filter := bson.M{"name": name}

	cursor, err := r.embeddingCollection.Find(ctx, filter)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	for cursor.Next(ctx) {
		var embedding models.EmbeddingDetails
		if err := cursor.Decode(&embedding); err != nil {
			return nil, err
		}
		results = append(results, &embedding)
	}

	return results, nil
}
