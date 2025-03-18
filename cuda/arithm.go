package cuda

/*
#include <stdlib.h>
#include "../core.h"
#include "core.h"
#include "arithm.h"
*/
import "C"

import (
	"unsafe"

	"gocv.io/x/gocv"
)

// Abs computes an absolute value of each matrix element.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga54a72bd772494ab34d05406fd76df2b6
func Abs(src GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuAbs(src.p, dst.p, nil))
}

// AbsWithStream computes an absolute value of each matrix element
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga54a72bd772494ab34d05406fd76df2b6
func AbsWithStream(src GpuMat, dst *GpuMat, stream Stream) error {
	return OpenCVResult(C.GpuAbs(src.p, dst.p, stream.p))
}

// AbsDiff computes per-element absolute difference of two matrices
// (or of a matrix and scalar) using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#gac062b283cf46ee90f74a773d3382ab54
func AbsDiff(src1, src2 GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuAbsDiff(src1.p, src2.p, dst.p, nil))
}

// AbsDiffWithStream computes an absolute value of each matrix element
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#gac062b283cf46ee90f74a773d3382ab54
func AbsDiffWithStream(src1, src2 GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuAbsDiff(src1.p, src2.p, dst.p, s.p))
}

// Add computes a matrix-matrix or matrix-scalar sum.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga5d9794bde97ed23d1c1485249074a8b1
func Add(src1, src2 GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuAdd(src1.p, src2.p, dst.p, nil))
}

// AddWithStream computes a matrix-matrix or matrix-scalar sum
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga5d9794bde97ed23d1c1485249074a8b1
func AddWithStream(src1, src2 GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuAdd(src1.p, src2.p, dst.p, s.p))
}

// BitwiseAnd performs a per-element bitwise conjunction of two matrices
// (or of matrix and scalar).
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga78d7c1a013877abd4237fbfc4e13bd76
func BitwiseAnd(src1, src2 GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuBitwiseAnd(src1.p, src2.p, dst.p, nil))
}

// BitwiseAndWithStream performs a per-element bitwise conjunction of two matrices
// (or of matrix and scalar) using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga78d7c1a013877abd4237fbfc4e13bd76
func BitwiseAndWithStream(src1, src2 GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuBitwiseAnd(src1.p, src2.p, dst.p, s.p))
}

// BitwiseNot performs a per-element bitwise inversion.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#gae58159a2259ae1acc76b531c171cf06a
func BitwiseNot(src GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuBitwiseNot(src.p, dst.p, nil))
}

// BitwiseNotWithStream performs a per-element bitwise inversion
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#gae58159a2259ae1acc76b531c171cf06a
func BitwiseNotWithStream(src GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuBitwiseNot(src.p, dst.p, s.p))
}

// BitwiseOr performs a per-element bitwise disjunction of two matrices
// (or of matrix and scalar).
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#gafd098ee3e51c68daa793999c1da3dfb7
func BitwiseOr(src1, src2 GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuBitwiseOr(src1.p, src2.p, dst.p, nil))
}

// BitwiseOrWithStream performs a per-element bitwise disjunction of two matrices
// (or of matrix and scalar) using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#gafd098ee3e51c68daa793999c1da3dfb7
func BitwiseOrWithStream(src1, src2 GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuBitwiseXor(src1.p, src2.p, dst.p, s.p))
}

// BitwiseXor performs a per-element exclusive or of two matrices
// (or of matrix and scalar).
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga3d95d4faafb099aacf18e8b915a4ad8d
func BitwiseXor(src1, src2 GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuBitwiseXor(src1.p, src2.p, dst.p, nil))
}

// BitwiseXorWithStream performs a per-element exclusive or of two matrices
// (or of matrix and scalar) using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga3d95d4faafb099aacf18e8b915a4ad8d
func BitwiseXorWithStream(src1, src2 GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuBitwiseXor(src1.p, src2.p, dst.p, s.p))
}

// Divide computes a matrix-matrix or matrix-scalar division.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga124315aa226260841e25cc0b9ea99dc3
func Divide(src1, src2 GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuDivide(src1.p, src2.p, dst.p, nil))
}

// DivideWithStream computes a matrix-matrix or matrix-scalar division
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga124315aa226260841e25cc0b9ea99dc3
func DivideWithStream(src1, src2 GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuDivide(src1.p, src2.p, dst.p, s.p))
}

// Exp computes an exponent of each matrix element.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#gac6e51541d3bb0a7a396128e4d5919b61
func Exp(src GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuExp(src.p, dst.p, nil))
}

// ExpWithStream computes an exponent of each matrix element
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#gac6e51541d3bb0a7a396128e4d5919b61
func ExpWithStream(src GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuExp(src.p, dst.p, s.p))
}

// Log computes natural logarithm of absolute value of each matrix element.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#gac6e51541d3bb0a7a396128e4d5919b61
func Log(src GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuLog(src.p, dst.p, nil))
}

// LogWithStream computes natural logarithm of absolute value of each matrix element
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#gac6e51541d3bb0a7a396128e4d5919b61
func LogWithStream(src GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuLog(src.p, dst.p, s.p))
}

// Max computes the per-element maximum of two matrices (or a matrix and a scalar).
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#gadb5dd3d870f10c0866035755b929b1e7
func Max(src1, src2 GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuMax(src1.p, src2.p, dst.p, nil))
}

// MaxWithStream computes the per-element maximum of two matrices (or a matrix and a scalar).
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#gadb5dd3d870f10c0866035755b929b1e7
func MaxWithStream(src1, src2 GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuMax(src1.p, src2.p, dst.p, s.p))
}

// Min computes the per-element minimum of two matrices (or a matrix and a scalar).
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga74f0b05a65b3d949c237abb5e6c60867
func Min(src1, src2 GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuMin(src1.p, src2.p, dst.p, nil))
}

// MinWithStream computes the per-element minimum of two matrices (or a matrix and a scalar).
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga74f0b05a65b3d949c237abb5e6c60867
func MinWithStream(src1, src2 GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuMin(src1.p, src2.p, dst.p, s.p))
}

// Multiply computes a matrix-matrix or matrix-scalar multiplication.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga497cc0615bf717e1e615143b56f00591
func Multiply(src1, src2 GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuMultiply(src1.p, src2.p, dst.p, nil))
}

// MultiplyWithStream computes a matrix-matrix or matrix-scalar multiplication.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga497cc0615bf717e1e615143b56f00591
func MultiplyWithStream(src1, src2 GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuMultiply(src1.p, src2.p, dst.p, s.p))
}

// Sqr computes a square value of each matrix element.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga8aae233da90ce0ffe309ab8004342acb
func Sqr(src GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuSqr(src.p, dst.p, nil))
}

// SqrWithStream computes a square value of each matrix element
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga8aae233da90ce0ffe309ab8004342acb
func SqrWithStream(src GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuSqr(src.p, dst.p, s.p))
}

// Sqrt computes a square root of each matrix element.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga09303680cb1a5521a922b6d392028d8c
func Sqrt(src GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuSqrt(src.p, dst.p, nil))
}

// SqrtWithStream computes a square root of each matrix element.
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga09303680cb1a5521a922b6d392028d8c
func SqrtWithStream(src GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuSqrt(src.p, dst.p, s.p))
}

// Subtract computes a matrix-matrix or matrix-scalar difference.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga6eab60fc250059e2fda79c5636bd067f
func Subtract(src1, src2 GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuSubtract(src1.p, src2.p, dst.p, nil))
}

// SubtractWithStream computes a matrix-matrix or matrix-scalar difference
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga6eab60fc250059e2fda79c5636bd067f
func SubtractWithStream(src1, src2 GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuSubtract(src1.p, src2.p, dst.p, s.p))
}

// Threshold applies a fixed-level threshold to each array element.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga40f1c94ae9a9456df3cad48e3cb008e1
func Threshold(src GpuMat, dst *GpuMat, thresh, maxval float64, typ gocv.ThresholdType) error {
	return OpenCVResult(C.GpuThreshold(src.p, dst.p, C.double(thresh), C.double(maxval), C.int(typ), nil))
}

// ThresholdWithStream applies a fixed-level threshold to each array element
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/d8/d34/group__cudaarithm__elem.html#ga40f1c94ae9a9456df3cad48e3cb008e1
func ThresholdWithStream(src GpuMat, dst *GpuMat, thresh, maxval float64, typ gocv.ThresholdType, s Stream) error {
	return OpenCVResult(C.GpuThreshold(src.p, dst.p, C.double(thresh), C.double(maxval), C.int(typ), s.p))
}

// Flip flips a 2D matrix around vertical, horizontal, or both axes.
//
// For further details, please see:
// https://docs.opencv.org/master/de/d09/group__cudaarithm__core.html#ga4d0a3f2b46e8f0f1ec2b5ac178dcd871
func Flip(src GpuMat, dst *GpuMat, flipCode int) error {
	return OpenCVResult(C.GpuFlip(src.p, dst.p, C.int(flipCode), nil))
}

// FlipWithStream flips a 2D matrix around vertical, horizontal, or both axes
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/de/d09/group__cudaarithm__core.html#ga4d0a3f2b46e8f0f1ec2b5ac178dcd871
func FlipWithStream(src GpuMat, dst *GpuMat, flipCode int, stream Stream) error {
	return OpenCVResult(C.GpuFlip(src.p, dst.p, C.int(flipCode), stream.p))
}

// Merge makes a multi-channel matrix out of several single-channel matrices.
//
// For further details, please see:
// https://docs.opencv.org/4.x/de/d09/group__cudaarithm__core.html#gafce19eb0fcad23f67ab45d544992436d
func Merge(mv []GpuMat, dst *GpuMat) error {
	cMatArray := make([]C.GpuMat, len(mv))
	for i, r := range mv {
		cMatArray[i] = r.p
	}
	cMats := C.GpuMats{
		mats:   (*C.GpuMat)(&cMatArray[0]),
		length: C.int(len(mv)),
	}

	return OpenCVResult(C.GpuMerge(cMats, dst.p, nil))
}

// MergeWithStream makes a multi-channel matrix out of several single-channel matrices
// using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/4.x/de/d09/group__cudaarithm__core.html#gafce19eb0fcad23f67ab45d544992436d
func MergeWithStream(mv []GpuMat, dst *GpuMat, s Stream) error {
	cMatArray := make([]C.GpuMat, len(mv))
	for i, r := range mv {
		cMatArray[i] = r.p
	}
	cMats := C.GpuMats{
		mats:   (*C.GpuMat)(&cMatArray[0]),
		length: C.int(len(mv)),
	}

	return OpenCVResult(C.GpuMerge(cMats, dst.p, s.p))
}

// Transpose transposes a matrix.
//
// For further details, please see:
// https://docs.opencv.org/4.x/de/d09/group__cudaarithm__core.html#ga327b71c3cb811a904ccf5fba37fc29f2
func Transpose(src GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.GpuTranspose(src.p, dst.p, nil))
}

// Transpose transposes a matrix using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/4.x/de/d09/group__cudaarithm__core.html#ga327b71c3cb811a904ccf5fba37fc29f2
func TransposeWithStream(src GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.GpuTranspose(src.p, dst.p, s.p))
}

// AddWeighted computes a weighted sum of two matrices.
//
// For further details, please see:
// https://docs.opencv.org/4.x/d8/d34/group__cudaarithm__elem.html#ga2cd14a684ea70c6ab2a63ee90ffe6201
func AddWeighted(src1 GpuMat, alpha float64, src2 GpuMat, beta float64, gamma float64, dst *GpuMat, dType int) error {
	return OpenCVResult(C.GpuAddWeighted(src1.p, C.double(alpha), src2.p, C.double(beta), C.double(gamma), dst.p, C.int(dType), nil))
}

// AddWeightedWithStream computes a weighted sum of two matrices using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/4.x/d8/d34/group__cudaarithm__elem.html#ga2cd14a684ea70c6ab2a63ee90ffe6201
func AddWeightedWithStream(src1 GpuMat, alpha float64, src2 GpuMat, beta float64, gamma float64, dst *GpuMat, dType int, s Stream) error {
	return OpenCVResult(C.GpuAddWeighted(src1.p, C.double(alpha), src2.p, C.double(beta), C.double(gamma), dst.p, C.int(dType), s.p))
}

// CopyMakeBorder forms a border around an image.
//
// For further details, please see:
// https://docs.opencv.org/master/de/d09/group__cudaarithm__core.html#ga5368db7656eacf846b40089c98053a49
func CopyMakeBorder(src GpuMat, dst *GpuMat, top, bottom, left, right int, borderType gocv.BorderType, value gocv.Scalar) error {
	bv := C.struct_Scalar{
		val1: C.double(value.Val1),
		val2: C.double(value.Val2),
		val3: C.double(value.Val3),
		val4: C.double(value.Val4),
	}

	return OpenCVResult(C.GpuCopyMakeBorder(src.p, dst.p, C.int(top), C.int(bottom), C.int(left), C.int(right), C.int(borderType), bv, nil))
}

// CopyMakeBorderWithStream forms a border around an image using a Stream for concurrency.
//
// For further details, please see:
// https://docs.opencv.org/master/de/d09/group__cudaarithm__core.html#ga5368db7656eacf846b40089c98053a49
func CopyMakeBorderWithStream(src GpuMat, dst *GpuMat, top, bottom, left, right int, borderType gocv.BorderType, value gocv.Scalar, s Stream) error {
	bv := C.struct_Scalar{
		val1: C.double(value.Val1),
		val2: C.double(value.Val2),
		val3: C.double(value.Val3),
		val4: C.double(value.Val4),
	}

	return OpenCVResult(C.GpuCopyMakeBorder(src.p, dst.p, C.int(top), C.int(bottom), C.int(left), C.int(right), C.int(borderType), bv, s.p))
}

type LookUpTable struct {
	p C.LookUpTable
}

// NewLookUpTable Creates implementation for cuda::LookUpTable .
//
// lut	Look-up table of 256 elements. It is a continuous CV_8U matrix.
//
// For further details, please see:
// https://docs.opencv.org/4.x/de/d09/group__cudaarithm__core.html#gaa75254a07dcf7996b4b5a68d383847f8
func NewLookUpTable(lut GpuMat) LookUpTable {
	return LookUpTable{p: C.Cuda_Create_LookUpTable(lut.p)}
}

// Close releases LookUpTable resources.
func (lt *LookUpTable) Close() {
	C.Cuda_LookUpTable_Close(lt.p)
}

// Transform Transforms the source matrix into the destination
// matrix using the given look-up table: dst(I) = lut(src(I)) .
//
// src: Source matrix. CV_8UC1 and CV_8UC3 matrices are supported for now.
//
// dst: Destination matrix.
//
// For further details, please see:
// https://docs.opencv.org/4.x/df/d29/classcv_1_1cuda_1_1LookUpTable.html#afdbcbd3047f847451892f3b18cd018de
func (lt *LookUpTable) Transform(src GpuMat, dst *GpuMat) error {
	return OpenCVResult(C.Cuda_LookUpTable_Transform(lt.p, src.p, dst.p, nil))
}

// Empty Returns true if the Algorithm is empty
// (e.g. in the very beginning or after unsuccessful read.
//
// For further details, please see:
// https://docs.opencv.org/4.x/d3/d46/classcv_1_1Algorithm.html#a827c8b2781ed17574805f373e6054ff1
func (lt *LookUpTable) Empty() bool {
	b := C.Cuda_LookUpTable_Empty(lt.p)

	return bool(b)
}

// TransformWithStream Transforms the source matrix into the destination
// matrix using the given look-up table: dst(I) = lut(src(I)) .
//
// src: Source matrix. CV_8UC1 and CV_8UC3 matrices are supported for now.
//
// dst: Destination matrix.
//
// stream: Stream for the asynchronous version.
//
// For further details, please see:
// https://docs.opencv.org/4.x/df/d29/classcv_1_1cuda_1_1LookUpTable.html#afdbcbd3047f847451892f3b18cd018de
func (lt *LookUpTable) TransformWithStream(src GpuMat, dst *GpuMat, s Stream) error {
	return OpenCVResult(C.Cuda_LookUpTable_Transform(lt.p, src.p, dst.p, s.p))
}

// Split Copies each plane of a multi-channel matrix into an array.
//
// src: Source matrix.
//
// dst: Destination array/vector of single-channel matrices.
//
// For further details, please see:
// https://docs.opencv.org/4.x/de/d09/group__cudaarithm__core.html#gaf1714e7a9ea0719c29bf378beaf5f99d
func Split(src GpuMat, dst []GpuMat) error {

	dstv := make([]C.GpuMat, len(dst))

	for i := range dst {
		dstv[i] = dst[i].p
	}

	c_dstv := C.GpuMats{
		mats:   unsafe.SliceData(dstv),
		length: C.int(len(dstv)),
	}

	return OpenCVResult(C.Cuda_Split(src.p, c_dstv, nil))
}

// SplitWithStream Copies each plane of a multi-channel matrix into an array.
//
// src: Source matrix.
//
// dst: Destination array/vector of single-channel matrices.
//
// stream: Stream for the asynchronous version.
//
// For further details, please see:
// https://docs.opencv.org/4.x/de/d09/group__cudaarithm__core.html#gaf1714e7a9ea0719c29bf378beaf5f99d
func SplitWithStream(src GpuMat, dst []GpuMat, s Stream) error {

	dstv := make([]C.GpuMat, len(dst))

	for i := range dst {
		dstv[i] = dst[i].p
	}

	c_dstv := C.GpuMats{
		mats:   unsafe.SliceData(dstv),
		length: C.int(len(dstv)),
	}

	return OpenCVResult(C.Cuda_Split(src.p, c_dstv, s.p))
}
