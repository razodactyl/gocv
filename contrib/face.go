package contrib

/*
#include <stdlib.h>
#include "face.h"
*/
import "C"
import (
	"image"

	"gocv.io/x/gocv"
)

// PredictResponse represents a predicted label and associated confidence.
type PredictResponse struct {
	Label      int32   `json:"label"`
	Confidence float32 `json:"confidence"`
}

var _ FaceRecognizer = (*LBPHFaceRecognizer)(nil)
var _ FaceRecognizer = (*FisherFaceRecognizer)(nil)
var _ BasicFaceRecognizer = (*FisherFaceRecognizer)(nil)
var _ FaceRecognizer = (*EigenFaceRecognizer)(nil)
var _ BasicFaceRecognizer = (*EigenFaceRecognizer)(nil)

// LBPHFaceRecognizer is a wrapper for the OpenCV Local Binary Patterns
// Histograms face recognizer.
type LBPHFaceRecognizer struct {
	p C.LBPHFaceRecognizer
}

// Empty returns true if the model is empty.
func (fr *LBPHFaceRecognizer) Empty() bool {
	return faceRecognizer_Empty(C.FaceRecognizer(fr.p))
}

// NewLBPHFaceRecognizer creates a new LBPH Recognizer model.
//
// For further information, see:
// https://docs.opencv.org/master/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html
func NewLBPHFaceRecognizer() *LBPHFaceRecognizer {
	return &LBPHFaceRecognizer{p: C.CreateLBPHFaceRecognizer()}
}

// Train loaded model with images and their labels
//
// see https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#ac8680c2aa9649ad3f55e27761165c0d6
func (fr *LBPHFaceRecognizer) Train(images []gocv.Mat, labels []int) error {
	return faceRecognizer_Train(C.FaceRecognizer(fr.p), images, labels)
}

// Update updates the existing trained model with new images and labels.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#a8a4e73ea878dcd0c235d0487189d25f3
func (fr *LBPHFaceRecognizer) Update(newImages []gocv.Mat, newLabels []int) error {
	return faceRecognizer_Update(C.FaceRecognizer(fr.p), newImages, newLabels)
}

// Predict predicts a label for a given input image. It returns the label for
// correctly predicted image or -1 if not found.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#aa2d2f02faffab1bf01317ae6502fb631
func (fr *LBPHFaceRecognizer) Predict(sample gocv.Mat) int {
	label := faceRecognizer_Predict(C.FaceRecognizer(fr.p), sample)

	return int(label)
}

// PredictExtendedResponse returns a label and associated confidence (e.g.
// distance) for a given input image. It is the extended version of
// `Predict()`.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#ab0d593e53ebd9a0f350c989fcac7f251
func (fr *LBPHFaceRecognizer) PredictExtendedResponse(sample gocv.Mat) PredictResponse {
	resp := faceRecognizer_PredictExtendedResponse(C.FaceRecognizer(fr.p), sample)

	return resp
}

// GetThreshold gets the threshold value of the model, i.e. the threshold
// applied in the prediction.
//
// For further information, see:
// https://docs.opencv.org/4.x/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html#acf2a6993eb4347b3f89009da693a3f70
func (fr *LBPHFaceRecognizer) GetThreshold() float32 {
	t := faceRecognizer_GetThreshold(C.FaceRecognizer(fr.p))
	return float32(t)
}

// SetThreshold sets the threshold value of the model, i.e. the threshold
// applied in the prediction.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#a3182081e5f8023e658ad8ab96656dd63
func (fr *LBPHFaceRecognizer) SetThreshold(threshold float32) {
	faceRecognizer_SetThreshold(C.FaceRecognizer(fr.p), threshold)
}

// SetNeighbors sets the neighbors value of the model, i.e. the number of
// sample points to build a Circular Local Binary Pattern from. Note that wrong
// neighbors can raise OpenCV exception!
//
// For further information, see:
// https://docs.opencv.org/master/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html#ab225f7bf353ce8697a506eda10124a92
func (fr *LBPHFaceRecognizer) SetNeighbors(neighbors int) {
	C.LBPHFaceRecognizer_SetNeighbors(fr.p, C.int(neighbors))
}

// GetNeighbors returns the neighbors value of the model.
//
// For further information, see:
// https://docs.opencv.org/master/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html#a50a3e2ca6e8464166e153c9df84b0a77
func (fr *LBPHFaceRecognizer) GetNeighbors() int {

	n := C.LBPHFaceRecognizer_GetNeighbors(fr.p)

	return int(n)
}

// SetRadius sets the radius used for building the Circular Local Binary
// Pattern.
//
// For further information, see:
// https://docs.opencv.org/master/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html#a62d94c75cade902fd3b487b1ef9883fc
func (fr *LBPHFaceRecognizer) SetRadius(radius int) {
	C.LBPHFaceRecognizer_SetRadius(fr.p, C.int(radius))
}

// SaveFile saves the trained model data to file.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#a2adf2d555550194244b05c91fefcb4d6
func (fr *LBPHFaceRecognizer) SaveFile(fname string) error {
	return faceRecognizer_SaveFile(C.FaceRecognizer(fr.p), fname)
}

// LoadFile loads a trained model data from file.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#acc42e5b04595dba71f0777c7179af8c3
func (fr *LBPHFaceRecognizer) LoadFile(fname string) error {
	return faceRecognizer_LoadFile(C.FaceRecognizer(fr.p), fname)
}

// SetGridX sets grid's X value
//
// For further information, see:
// https://docs.opencv.org/4.x/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html#ad65975baee31dbf3bd2a750feef74831
func (fr *LBPHFaceRecognizer) SetGridX(x int) {
	C.LBPHFaceRecognizer_SetGridX(fr.p, C.int(x))
}

// SetGridY sets grid's Y value
//
// For further information, see:
// https://docs.opencv.org/4.x/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html#a9cebb0138dbb3553b27beb2df3924ae6
func (fr *LBPHFaceRecognizer) SetGridY(y int) {
	C.LBPHFaceRecognizer_SetGridY(fr.p, C.int(y))
}

// GetGridX gets grid's X value
//
// For further information, see:
// https://docs.opencv.org/4.x/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html#ada6839bed931a8f68c5127e1af7a8b83
func (fr *LBPHFaceRecognizer) GetGridX() int {
	x := C.LBPHFaceRecognizer_GetGridX(fr.p)
	return int(x)
}

// GetGridY gets grid's Y value
//
// For further information, see:
// https://docs.opencv.org/4.x/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html#a22c68c0baf3eb9e852f47ae9241dbf15
func (fr *LBPHFaceRecognizer) GetGridY() int {
	y := C.LBPHFaceRecognizer_GetGridY(fr.p)
	return int(y)
}

// SetGrid helper for SetGrid* functions
func (fr *LBPHFaceRecognizer) SetGrid(p image.Point) {
	fr.SetGridX(p.X)
	fr.SetGridY(p.Y)
}

// GetGrid helper for GetGrid* functions
func (fr *LBPHFaceRecognizer) GetGrid() image.Point {
	return image.Pt(fr.GetGridX(), fr.GetGridY())
}

func (fr *LBPHFaceRecognizer) Close() error {
	C.LBPHFaceRecognizer_Close(fr.p)
	fr.p = nil
	return nil
}

type FisherFaceRecognizer struct {
	p C.FisherFaceRecognizer
}

// NewFisherFaceRecognizer creates a new Fisher Recognizer model.
//
// For further information, see:
// https://docs.opencv.org/4.x/d2/de9/classcv_1_1face_1_1FisherFaceRecognizer.html#ac6e204df6d7e526f4c77d3e0389dfbaa
func NewFisherFaceRecognizer() *FisherFaceRecognizer {
	return &FisherFaceRecognizer{p: C.FisherFaceRecognizer_Create()}
}

// NewFisherFaceRecognizerWithParams creates a new Fisher Recognizer model.
//
// [num_components]	The number of components (read: Fisherfaces) kept for this Linear Discriminant Analysis with the Fisherfaces criterion. It's useful to keep all components, that means the number of your classes c (read: subjects, persons you want to recognize). If you leave this at the default (0) or set it to a value less-equal 0 or greater (c-1), it will be set to the correct number (c-1) automatically.
//
// [threshold] The threshold applied in the prediction. If the distance to the nearest neighbor is larger than the threshold, this method returns -1.
//
// For further information, see:
// https://docs.opencv.org/4.x/d2/de9/classcv_1_1face_1_1FisherFaceRecognizer.html#ac6e204df6d7e526f4c77d3e0389dfbaa
func NewFisherFaceRecognizerWithParams(numComponents int, threshold float32) *FisherFaceRecognizer {
	return &FisherFaceRecognizer{p: C.FisherFaceRecognizer_CreateWithParams(C.int(numComponents), C.float(threshold))}
}

// Empty returns true if the model is empty.
func (fr *FisherFaceRecognizer) Empty() bool {
	b := faceRecognizer_Empty(C.FaceRecognizer(fr.p))
	return bool(b)
}

func (fr *FisherFaceRecognizer) GetEigenValues() gocv.Mat {
	return basicFaceRecognizer_GetEigenValues(C.BasicFaceRecognizer(fr.p))
}

func (fr *FisherFaceRecognizer) GetEigenVectors() gocv.Mat {
	return basicFaceRecognizer_GetEigenVectors(C.BasicFaceRecognizer(fr.p))
}

func (fr *FisherFaceRecognizer) GetLabels() gocv.Mat {
	return basicFaceRecognizer_GetLabels(C.BasicFaceRecognizer(fr.p))
}

func (fr *FisherFaceRecognizer) GetMean() gocv.Mat {
	return basicFaceRecognizer_GetMean(C.BasicFaceRecognizer(fr.p))
}

func (fr *FisherFaceRecognizer) GetNumComponents() int {
	return basicFaceRecognizer_GetNumComponents(C.BasicFaceRecognizer(fr.p))
}

func (fr *FisherFaceRecognizer) SetNumComponents(val int) {
	basicFaceRecognizer_SetNumComponents(C.BasicFaceRecognizer(fr.p), val)
}

func (fr *FisherFaceRecognizer) GetProjections() []gocv.Mat {
	return basicFaceRecognizer_GetProjections(C.BasicFaceRecognizer(fr.p))
}

// GetThreshold gets the threshold value of the model, i.e. the threshold
// applied in the prediction.
//
// For further information, see:
// https://docs.opencv.org/4.x/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html#acf2a6993eb4347b3f89009da693a3f70
func (fr *FisherFaceRecognizer) GetThreshold() float32 {
	return faceRecognizer_GetThreshold(C.FaceRecognizer(fr.p))
}

// SetThreshold sets the threshold value of the model, i.e. the threshold
// applied in the prediction.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#a3182081e5f8023e658ad8ab96656dd63
func (fr *FisherFaceRecognizer) SetThreshold(threshold float32) {
	faceRecognizer_SetThreshold(C.FaceRecognizer(fr.p), threshold)
}

// LoadFile loads a trained model data from file.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#acc42e5b04595dba71f0777c7179af8c3
func (fr *FisherFaceRecognizer) LoadFile(filename string) error {
	return basicFaceRecognizer_LoadFile(C.BasicFaceRecognizer(fr.p), filename)
}

// SaveFile saves the trained model data to file.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#a2adf2d555550194244b05c91fefcb4d6
func (fr *FisherFaceRecognizer) SaveFile(filename string) error {
	return basicFaceRecognizer_SaveFile(C.BasicFaceRecognizer(fr.p), filename)
}

// Predict predicts a label for a given input image. It returns the label for
// correctly predicted image or -1 if not found.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#aa2d2f02faffab1bf01317ae6502fb631
func (fr *FisherFaceRecognizer) Predict(sample gocv.Mat) int {
	return faceRecognizer_Predict(C.FaceRecognizer(fr.p), sample)
}

// PredictExtendedResponse returns a label and associated confidence (e.g.
// distance) for a given input image. It is the extended version of
// `Predict()`.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#ab0d593e53ebd9a0f350c989fcac7f251
func (fr *FisherFaceRecognizer) PredictExtendedResponse(sample gocv.Mat) PredictResponse {
	return faceRecognizer_PredictExtendedResponse(C.FaceRecognizer(fr.p), sample)
}

// Train loaded model with images and their labels
//
// see https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#ac8680c2aa9649ad3f55e27761165c0d6
func (fr *FisherFaceRecognizer) Train(images []gocv.Mat, labels []int) error {
	return basicFaceRecognizer_Train(C.BasicFaceRecognizer(fr.p), images, labels)
}

// Update This model does not support updating.
//
// For further information, see:
// https://docs.opencv.org/4.x/d2/de9/classcv_1_1face_1_1FisherFaceRecognizer.html#ac6e204df6d7e526f4c77d3e0389dfbaa
func (fr *FisherFaceRecognizer) Update(newImages []gocv.Mat, newLabels []int) error {
	return faceRecognizer_Train(C.FaceRecognizer(fr.p), newImages, newLabels)

}

func (fr *FisherFaceRecognizer) Close() error {
	C.FisherFaceRecognizer_Close(fr.p)
	fr.p = nil
	return nil
}

type EigenFaceRecognizer struct {
	p C.EigenFaceRecognizer
}

// NewEigenFaceRecognizer creates a new Eigen Recognizer model.
//
// For further information, see:
// https://docs.opencv.org/4.x/dd/d7c/classcv_1_1face_1_1EigenFaceRecognizer.html#a22c8392f27a20b24d04351b675e7b6db
func NewEigenFaceRecognizer() *EigenFaceRecognizer {
	return &EigenFaceRecognizer{p: C.EigenFaceRecognizer_Create()}
}

// NewEigenFaceRecognizerWithParams creates a new Eigen Recognizer model.
//
// [num_components]	The number of components (read: Eigenfaces) kept for this Principal Component Analysis.
// [threshold]	The threshold applied in the prediction.
//
// For further information, see:
// https://docs.opencv.org/4.x/dd/d7c/classcv_1_1face_1_1EigenFaceRecognizer.html#a22c8392f27a20b24d04351b675e7b6db
func NewEigenFaceRecognizerWithParams(numComponents int, threshold float32) *EigenFaceRecognizer {
	return &EigenFaceRecognizer{p: C.EigenFaceRecognizer_CreateWithParams(C.int(numComponents), C.float(threshold))}
}

// Empty returns true if the model is empty.
func (fr *EigenFaceRecognizer) Empty() bool {
	b := faceRecognizer_Empty(C.FaceRecognizer(fr.p))
	return bool(b)
}

func (fr *EigenFaceRecognizer) GetEigenValues() gocv.Mat {
	return basicFaceRecognizer_GetEigenValues(C.BasicFaceRecognizer(fr.p))
}

func (fr *EigenFaceRecognizer) GetEigenVectors() gocv.Mat {
	return basicFaceRecognizer_GetEigenVectors(C.BasicFaceRecognizer(fr.p))
}

func (fr *EigenFaceRecognizer) GetLabels() gocv.Mat {
	return basicFaceRecognizer_GetLabels(C.BasicFaceRecognizer(fr.p))
}

func (fr *EigenFaceRecognizer) GetMean() gocv.Mat {
	return basicFaceRecognizer_GetMean(C.BasicFaceRecognizer(fr.p))
}

func (fr *EigenFaceRecognizer) GetNumComponents() int {
	return basicFaceRecognizer_GetNumComponents(C.BasicFaceRecognizer(fr.p))
}

func (fr *EigenFaceRecognizer) SetNumComponents(val int) {
	basicFaceRecognizer_SetNumComponents(C.BasicFaceRecognizer(fr.p), val)
}

func (fr *EigenFaceRecognizer) GetProjections() []gocv.Mat {
	return basicFaceRecognizer_GetProjections(C.BasicFaceRecognizer(fr.p))
}

// GetThreshold gets the threshold value of the model, i.e. the threshold
// applied in the prediction.
//
// For further information, see:
// https://docs.opencv.org/4.x/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html#acf2a6993eb4347b3f89009da693a3f70
func (fr *EigenFaceRecognizer) GetThreshold() float32 {
	return faceRecognizer_GetThreshold(C.FaceRecognizer(fr.p))
}

// SetThreshold sets the threshold value of the model, i.e. the threshold
// applied in the prediction.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#a3182081e5f8023e658ad8ab96656dd63
func (fr *EigenFaceRecognizer) SetThreshold(threshold float32) {
	faceRecognizer_SetThreshold(C.FaceRecognizer(fr.p), threshold)
}

// LoadFile loads a trained model data from file.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#acc42e5b04595dba71f0777c7179af8c3
func (fr *EigenFaceRecognizer) LoadFile(filename string) error {
	return basicFaceRecognizer_LoadFile(C.BasicFaceRecognizer(fr.p), filename)
}

// SaveFile saves the trained model data to file.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#a2adf2d555550194244b05c91fefcb4d6
func (fr *EigenFaceRecognizer) SaveFile(filename string) error {
	return basicFaceRecognizer_SaveFile(C.BasicFaceRecognizer(fr.p), filename)
}

// Predict predicts a label for a given input image. It returns the label for
// correctly predicted image or -1 if not found.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#aa2d2f02faffab1bf01317ae6502fb631
func (fr *EigenFaceRecognizer) Predict(sample gocv.Mat) int {
	return faceRecognizer_Predict(C.FaceRecognizer(fr.p), sample)
}

// PredictExtendedResponse returns a label and associated confidence (e.g.
// distance) for a given input image. It is the extended version of
// `Predict()`.
//
// For further information, see:
// https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#ab0d593e53ebd9a0f350c989fcac7f251
func (fr *EigenFaceRecognizer) PredictExtendedResponse(sample gocv.Mat) PredictResponse {
	return faceRecognizer_PredictExtendedResponse(C.FaceRecognizer(fr.p), sample)
}

// Train loaded model with images and their labels
//
// see https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#ac8680c2aa9649ad3f55e27761165c0d6
func (fr *EigenFaceRecognizer) Train(images []gocv.Mat, labels []int) error {
	return basicFaceRecognizer_Train(C.BasicFaceRecognizer(fr.p), images, labels)
}

// Update This model does not support updating.
//
// For further information, see:
// https://docs.opencv.org/4.x/dd/d7c/classcv_1_1face_1_1EigenFaceRecognizer.html#a22c8392f27a20b24d04351b675e7b6db
func (fr *EigenFaceRecognizer) Update(newImages []gocv.Mat, newLabels []int) error {
	return basicFaceRecognizer_Train(C.BasicFaceRecognizer(fr.p), newImages, newLabels)
}

func (fr *EigenFaceRecognizer) Close() error {
	C.EigenFaceRecognizer_Close(fr.p)
	fr.p = nil
	return nil
}
