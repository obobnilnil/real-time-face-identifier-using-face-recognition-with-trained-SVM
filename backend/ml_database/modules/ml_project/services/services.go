package services

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"ml_database/modules/ml_project/models"
	"ml_database/modules/ml_project/repositories"
	"os"
	"path/filepath"
	"time"
)

type ServicePort interface {
	InputFaceRecognitionDetailsServices(inputFaceRecognition models.InputFaceRecognitionDetails) error
	UploadImageAndReturnPathServices(inputImage models.UploadImageAndReturnPath) ([]string, error)
	GetImagesByTimeRangeServices(cameraID string, baseTime time.Time) ([]string, error)
	GetImagesByNameAndTimeServices(name string, baseTime *time.Time, isAll bool) ([]string, error)
	FetchEmbeddingDetailsForModelTrainerServices(ctx context.Context, name string) ([]*models.EmbeddingDetails, error)
}
type serviceAdapter struct {
	r repositories.RepositoryPort
}

func NewServiceAdapter(r repositories.RepositoryPort) ServicePort {
	return &serviceAdapter{r: r}
}

func (s *serviceAdapter) InputFaceRecognitionDetailsServices(inputFaceRecognition models.InputFaceRecognitionDetails) error {
	if inputFaceRecognition.Name == "" || inputFaceRecognition.CameraID == "" || inputFaceRecognition.ImagePath == "" {
		return errors.New("name, camera_id, and image_path must not be empty")
	}

	if len(inputFaceRecognition.Embedding) != 128 {
		return errors.New("embedding must contain exactly 128 float values")
	}
	if inputFaceRecognition.Timestamp.IsZero() {
		return errors.New("timestamp must not be empty")
	}
	if inputFaceRecognition.Confidence < 0.0 || inputFaceRecognition.Confidence > 1.0 {
		return errors.New("confidence must be between 0 and 1")
	}
	err := s.r.InputFaceRecognitionDetailsRepositories(inputFaceRecognition)
	if err != nil {
		log.Println("[ERROR] Failed to save face data:", err)
		return err
	}

	return nil
}

func (s *serviceAdapter) UploadImageAndReturnPathServices(inputImage models.UploadImageAndReturnPath) ([]string, error) { // with timestamp + cam number
	var savedPaths []string

	outputDir := "./snapshots"
	err := os.MkdirAll(outputDir, os.ModePerm)
	if err != nil {
		log.Printf("[ERROR] Cannot create output directory: %v", err)
		return nil, err
	}

	for _, file := range inputImage.Images {
		openedFile, err := file.Open()
		if err != nil {
			log.Printf("[ERROR] Cannot open file: %v", err)
			return nil, err
		}
		defer openedFile.Close()

		timestamp := time.Now().UnixNano()
		filename := fmt.Sprintf("face_%s_%d%s", inputImage.CameraID, timestamp, filepath.Ext(file.Filename))
		targetPath := filepath.Join(outputDir, filename)

		out, err := os.Create(targetPath)
		if err != nil {
			log.Printf("[ERROR] Cannot create file: %v", err)
			return nil, err
		}
		defer out.Close()

		if _, err := io.Copy(out, openedFile); err != nil {
			log.Printf("[ERROR] Failed to copy file content: %v", err)
			return nil, err
		}

		savedPaths = append(savedPaths, targetPath)
	}

	return savedPaths, nil
}

func (s *serviceAdapter) GetImagesByTimeRangeServices(cameraID string, baseTime time.Time) ([]string, error) {
	paths, err := s.r.GetImagesByTimeRangeRepositories(cameraID, baseTime)
	if err != nil {
		log.Println("[ERROR] QueryImagesByTimeRangeRepositories:", err)
		return nil, err
	}
	return paths, nil
}

func (s *serviceAdapter) GetImagesByNameAndTimeServices(name string, baseTime *time.Time, isAll bool) ([]string, error) {
	paths, err := s.r.GetImagePathsByNameAndTimeRepositories(name, baseTime, isAll)
	if err != nil {
		log.Println("[Error] QueryImagesByNameRepositories:", err)
		return nil, err
	}
	return paths, nil
}

func (s *serviceAdapter) FetchEmbeddingDetailsForModelTrainerServices(ctx context.Context, name string) ([]*models.EmbeddingDetails, error) {
	embeddings, err := s.r.FetchEmbeddingDetailsForModelTrainerRepositories(ctx, name)
	if err != nil {
		log.Println("[Error] FetchEmbeddingsByNameRepositories:", err)
		return nil, err
	}
	return embeddings, nil
}
