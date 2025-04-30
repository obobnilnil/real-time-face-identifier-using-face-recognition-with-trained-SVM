package servers

import (
	"ml_database/modules/ml_project/handlers"
	"ml_database/modules/ml_project/repositories"
	"ml_database/modules/ml_project/services"

	"github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/mongo"
)

func SetupRoutesMLProject(router *gin.Engine, conn *mongo.Database) {

	r := repositories.NewRepositoryAdapter(conn)
	s := services.NewServiceAdapter(r)
	h := handlers.NewHanerhandlerAdapter(s)

	router.POST("/api/InputFaceRecognitionDetails", h.InputFaceRecognitionDetailsHandlers)
	router.POST("/api/UploadImageAndReturnPath", h.UploadImageAndReturnPathHandlers)
	router.GET("/api/GetImagesByCamNumber", h.GetImagesByTimeRangeHandlers)
	router.GET("/api/GetImagesByName", h.GetImagesByNameAndTimeHandlers)
	router.GET("/api/FetchEmbeddingDetailsForModelTrainer", h.FetchEmbeddingDetailsForModelTrainerHandlers)
}
