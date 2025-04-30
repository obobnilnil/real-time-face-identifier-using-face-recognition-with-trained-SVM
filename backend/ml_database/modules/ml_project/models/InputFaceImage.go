package models

import "mime/multipart"

type UploadImageAndReturnPath struct {
	CameraID string                  `form:"camera_id"`
	Images   []*multipart.FileHeader `form:"images"`
}
