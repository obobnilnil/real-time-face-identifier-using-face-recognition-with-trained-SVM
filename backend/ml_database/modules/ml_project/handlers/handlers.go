package handlers

import (
	"log"
	"ml_database/modules/ml_project/models"
	"ml_database/modules/ml_project/services"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

type HandlerPort interface {
	InputFaceRecognitionDetailsHandlers(c *gin.Context)
	UploadImageAndReturnPathHandlers(c *gin.Context)
	GetImagesByTimeRangeHandlers(c *gin.Context)
	GetImagesByNameAndTimeHandlers(c *gin.Context)
	FetchEmbeddingDetailsForModelTrainerHandlers(c *gin.Context)
}

type handlerAdapter struct {
	s services.ServicePort
}

func NewHanerhandlerAdapter(s services.ServicePort) HandlerPort {
	return &handlerAdapter{s: s}
}

func (h *handlerAdapter) InputFaceRecognitionDetailsHandlers(c *gin.Context) {
	var inputFaceRecogniion models.InputFaceRecognitionDetails
	if err := c.ShouldBindJSON(&inputFaceRecogniion); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"status": "Error", "message": err.Error()})
		return
	}
	err := h.s.InputFaceRecognitionDetailsServices(inputFaceRecogniion)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"status": "Error", "message": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"status": "OK", "message": "Face Recognition Details have been successfully added."})
}

func (h *handlerAdapter) UploadImageAndReturnPathHandlers(c *gin.Context) {
	var inputImage models.UploadImageAndReturnPath
	if err := c.ShouldBind(&inputImage); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"status": "Error", "message": err.Error()})
		return
	}

	if len(inputImage.Images) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"status": "Error", "message": "Please insert at least one image"})
		return
	}
	paths, err := h.s.UploadImageAndReturnPathServices(inputImage)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"status": "Error", "message": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "OK", "paths": paths})
}

func (h *handlerAdapter) GetImagesByTimeRangeHandlers(c *gin.Context) {
	cameraID := c.Query("cameraID")
	timestampStr := c.Query("timestamp")
	if cameraID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "Missing required parameter: cameraID"})
		return
	}
	if timestampStr == "" {
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "Missing required parameter: timestamp"})
		return
	}
	log.Println("[DEBUG] Raw timestamp string:", timestampStr)

	parsedTime, err := time.Parse(time.RFC3339, timestampStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid timestamp format"})
		return
	}

	paths, err := h.s.GetImagesByTimeRangeServices(cameraID, parsedTime)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if len(paths) == 0 {
		c.JSON(http.StatusNotFound, gin.H{"message": "No images found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "OK", "paths": paths})
}

func (h *handlerAdapter) GetImagesByNameAndTimeHandlers(c *gin.Context) {
	name := c.Query("name")
	timestampStr := c.Query("timestamp")

	if name == "" {
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "Missing required parameter: name"})
		return
	}

	log.Println("[DEBUG] Name:", name)
	log.Println("[DEBUG] Raw timestamp string:", timestampStr)

	var parsedTime *time.Time
	var isAll bool

	if timestampStr != "" {
		if timestampStr == "All" {
			isAll = true
		} else {
			t, err := time.Parse(time.RFC3339, timestampStr)
			if err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid timestamp format"})
				return
			}
			parsedTime = &t
		}
	}

	paths, err := h.s.GetImagesByNameAndTimeServices(name, parsedTime, isAll)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if len(paths) == 0 {
		c.JSON(http.StatusNotFound, gin.H{"message": "No images found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "OK", "paths": paths})
}

func (h *handlerAdapter) FetchEmbeddingDetailsForModelTrainerHandlers(c *gin.Context) {
	ctx := c.Request.Context()

	name := c.Query("name")
	if name == "" {
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "Missing required parameter: name"})
		return
	}
	log.Println("[DEBUG] Name query param:", name)

	embeddings, err := h.s.FetchEmbeddingDetailsForModelTrainerServices(ctx, name)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": err.Error()})
		return
	}
	if len(embeddings) == 0 {
		c.JSON(http.StatusNotFound, gin.H{"status": "error", "message": "No embeddings found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "OK", "embeddings": embeddings})
}
