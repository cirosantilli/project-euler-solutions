import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk000
import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk001
import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk002
import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk003
import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk004
import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk005
import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk006
import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk007
import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk008
import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk009
import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk010
import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk011
import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk012
import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk013

namespace ProjectEulerSolutions.PrimePairVertices
open PrimePairCertificate

def small_left : List Nat := block_000 ++ block_001
theorem small_left_checked : primeSegment 2 4096 = small_left := by
  change primeSegment 2 (2048 + 2048) = _
  rw [primeSegment_append, block_000_checked, block_001_checked]
  rfl

def small_right_right : List Nat := block_003 ++ block_004
theorem small_right_right_checked : primeSegment 6146 2243 = small_right_right := by
  change primeSegment 6146 (2048 + 195) = _
  rw [primeSegment_append, block_003_checked, block_004_checked]
  rfl

def small_right : List Nat := block_002 ++ small_right_right
theorem small_right_checked : primeSegment 4098 4291 = small_right := by
  change primeSegment 4098 (2048 + 2243) = _
  rw [primeSegment_append, block_002_checked, small_right_right_checked]
  rfl

def small : List Nat := small_left ++ small_right
theorem small_checked : primeSegment 2 8387 = small := by
  change primeSegment 2 (4096 + 4291) = _
  rw [primeSegment_append, small_left_checked, small_right_checked]
  rfl

def large_left_left : List Nat := small ++ block_005
theorem large_left_left_checked : primeSegment 2 10368 = large_left_left := by
  change primeSegment 2 (8387 + 1981) = _
  rw [primeSegment_append, small_checked, block_005_checked]
  rfl

def large_left_right_right : List Nat := block_007 ++ block_008
theorem large_left_right_right_checked : primeSegment 12418 4096 = large_left_right_right := by
  change primeSegment 12418 (2048 + 2048) = _
  rw [primeSegment_append, block_007_checked, block_008_checked]
  rfl

def large_left_right : List Nat := block_006 ++ large_left_right_right
theorem large_left_right_checked : primeSegment 10370 6144 = large_left_right := by
  change primeSegment 10370 (2048 + 4096) = _
  rw [primeSegment_append, block_006_checked, large_left_right_right_checked]
  rfl

def large_left : List Nat := large_left_left ++ large_left_right
theorem large_left_checked : primeSegment 2 16512 = large_left := by
  change primeSegment 2 (10368 + 6144) = _
  rw [primeSegment_append, large_left_left_checked, large_left_right_checked]
  rfl

def large_right_left : List Nat := block_009 ++ block_010
theorem large_right_left_checked : primeSegment 16514 4096 = large_right_left := by
  change primeSegment 16514 (2048 + 2048) = _
  rw [primeSegment_append, block_009_checked, block_010_checked]
  rfl

def large_right_right_right : List Nat := block_012 ++ block_013
theorem large_right_right_right_checked : primeSegment 22658 3376 = large_right_right_right := by
  change primeSegment 22658 (2048 + 1328) = _
  rw [primeSegment_append, block_012_checked, block_013_checked]
  rfl

def large_right_right : List Nat := block_011 ++ large_right_right_right
theorem large_right_right_checked : primeSegment 20610 5424 = large_right_right := by
  change primeSegment 20610 (2048 + 3376) = _
  rw [primeSegment_append, block_011_checked, large_right_right_right_checked]
  rfl

def large_right : List Nat := large_right_left ++ large_right_right
theorem large_right_checked : primeSegment 16514 9520 = large_right := by
  change primeSegment 16514 (4096 + 5424) = _
  rw [primeSegment_append, large_right_left_checked, large_right_right_checked]
  rfl

def large : List Nat := large_left ++ large_right
theorem large_checked : primeSegment 2 26032 = large := by
  change primeSegment 2 (16512 + 9520) = _
  rw [primeSegment_append, large_left_checked, large_right_checked]
  rfl

end ProjectEulerSolutions.PrimePairVertices
