package mongodb

import (
	"fmt"
	"os"

	"context"
	"log"
	"time"

	"github.com/joho/godotenv"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/mongo/readpref"
)

var Collection *mongo.Collection

func MongoDB() *mongo.Database {
	err := godotenv.Load()
	if err != nil {
		log.Fatalf("Error loading .env file")
	}

	connectionURI := os.Getenv("MONGODB")
	fmt.Println("MONGODB URI =", os.Getenv("MONGODB"))
	clientOptions := options.Client().ApplyURI(connectionURI)
	fmt.Println(connectionURI)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	client, err := mongo.Connect(ctx, clientOptions)
	if err != nil {
		log.Fatalf("Failed to connect to MongoDB: %v", err)
	}

	err = client.Ping(ctx, readpref.Primary())
	if err != nil {
		log.Fatal(err)
	}
	dbName := os.Getenv("MONGO_DB")
	conn := client.Database(dbName)
	fmt.Println("MongoDB Connected")
	return conn
}
