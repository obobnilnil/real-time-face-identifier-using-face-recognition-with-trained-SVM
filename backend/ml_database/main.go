package main

import (
	"context"
	"ml_database/modules/ml_project/servers"
	"ml_database/pkg/database/mongodb"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func main() {

	conn := mongodb.MongoDB()
	defer conn.Client().Disconnect(context.Background())

	router := gin.Default()
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"*"}
	config.AllowMethods = []string{"GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Type", "X-Auth-Token", "Authorization"}
	router.Use(cors.New(config))

	servers.SetupRoutesMLProject(router, conn)

	err := router.Run(":8889")
	if err != nil {
		panic(err.Error())
	}
}
