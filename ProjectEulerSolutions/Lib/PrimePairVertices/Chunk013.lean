import ProjectEulerSolutions.Lib.PrimePairCertificate

/-! Generated bounded vertex intervals; every list is kernel checked. -/
namespace ProjectEulerSolutions.PrimePairVertices
open PrimePairCertificate

def interval_24706 : List Nat := [24709, 24733, 24749, 24763, 24767, 24781, 24793, 24799, 24809, 24821]
theorem interval_24706_checked : primeSegment 24706 128 = interval_24706 := by
  rw [← computedSegment_eq_primeSegment 24706 128 (by decide)]
  decide

def interval_24834 : List Nat := [24841, 24847, 24851, 24859, 24877, 24889, 24907, 24917, 24919, 24923, 24943, 24953]
theorem interval_24834_checked : primeSegment 24834 128 = interval_24834 := by
  rw [← computedSegment_eq_primeSegment 24834 128 (by decide)]
  decide

def interval_24962 : List Nat := [24967, 24971, 24977, 24979, 24989, 25013, 25031, 25033, 25037, 25057, 25073, 25087]
theorem interval_24962_checked : primeSegment 24962 128 = interval_24962 := by
  rw [← computedSegment_eq_primeSegment 24962 128 (by decide)]
  decide

def interval_25090 : List Nat := [25097, 25111, 25117, 25121, 25127, 25147, 25153, 25163, 25169, 25171, 25183, 25189]
theorem interval_25090_checked : primeSegment 25090 128 = interval_25090 := by
  rw [← computedSegment_eq_primeSegment 25090 128 (by decide)]
  decide

def interval_25218 : List Nat := [25219, 25229, 25237, 25243, 25247, 25253, 25261, 25301, 25303, 25307, 25309, 25321, 25339, 25343]
theorem interval_25218_checked : primeSegment 25218 128 = interval_25218 := by
  rw [← computedSegment_eq_primeSegment 25218 128 (by decide)]
  decide

def interval_25346 : List Nat := [25349, 25357, 25367, 25373, 25391, 25409, 25411, 25423, 25439, 25447, 25453, 25457, 25463, 25469, 25471]
theorem interval_25346_checked : primeSegment 25346 128 = interval_25346 := by
  rw [← computedSegment_eq_primeSegment 25346 128 (by decide)]
  decide

def interval_25474 : List Nat := [25523, 25537, 25541, 25561, 25577, 25579, 25583, 25589, 25601]
theorem interval_25474_checked : primeSegment 25474 128 = interval_25474 := by
  rw [← computedSegment_eq_primeSegment 25474 128 (by decide)]
  decide

def interval_25602 : List Nat := [25603, 25609, 25621, 25633, 25639, 25643, 25657, 25667, 25673, 25679, 25693, 25703, 25717]
theorem interval_25602_checked : primeSegment 25602 128 = interval_25602 := by
  rw [← computedSegment_eq_primeSegment 25602 128 (by decide)]
  decide

def interval_25730 : List Nat := [25733, 25741, 25747, 25759, 25763, 25771, 25793, 25799, 25801, 25819, 25841, 25847, 25849]
theorem interval_25730_checked : primeSegment 25730 128 = interval_25730 := by
  rw [← computedSegment_eq_primeSegment 25730 128 (by decide)]
  decide

def interval_25858 : List Nat := [25867, 25873, 25889, 25903, 25913, 25919, 25931, 25933, 25939, 25943, 25951, 25969, 25981]
theorem interval_25858_checked : primeSegment 25858 128 = interval_25858 := by
  rw [← computedSegment_eq_primeSegment 25858 128 (by decide)]
  decide

def interval_25986 : List Nat := [25997, 25999, 26003, 26017, 26021, 26029]
theorem interval_25986_checked : primeSegment 25986 48 = interval_25986 := by
  rw [← computedSegment_eq_primeSegment 25986 48 (by decide)]
  decide

def block_013_left_left : List Nat := interval_24706 ++ interval_24834
theorem block_013_left_left_checked : primeSegment 24706 256 = block_013_left_left := by
  change primeSegment 24706 (128 + 128) = _
  rw [primeSegment_append, interval_24706_checked, interval_24834_checked]
  rfl

def block_013_left_right_right : List Nat := interval_25090 ++ interval_25218
theorem block_013_left_right_right_checked : primeSegment 25090 256 = block_013_left_right_right := by
  change primeSegment 25090 (128 + 128) = _
  rw [primeSegment_append, interval_25090_checked, interval_25218_checked]
  rfl

def block_013_left_right : List Nat := interval_24962 ++ block_013_left_right_right
theorem block_013_left_right_checked : primeSegment 24962 384 = block_013_left_right := by
  change primeSegment 24962 (128 + 256) = _
  rw [primeSegment_append, interval_24962_checked, block_013_left_right_right_checked]
  rfl

def block_013_left : List Nat := block_013_left_left ++ block_013_left_right
theorem block_013_left_checked : primeSegment 24706 640 = block_013_left := by
  change primeSegment 24706 (256 + 384) = _
  rw [primeSegment_append, block_013_left_left_checked, block_013_left_right_checked]
  rfl

def block_013_right_left_right : List Nat := interval_25474 ++ interval_25602
theorem block_013_right_left_right_checked : primeSegment 25474 256 = block_013_right_left_right := by
  change primeSegment 25474 (128 + 128) = _
  rw [primeSegment_append, interval_25474_checked, interval_25602_checked]
  rfl

def block_013_right_left : List Nat := interval_25346 ++ block_013_right_left_right
theorem block_013_right_left_checked : primeSegment 25346 384 = block_013_right_left := by
  change primeSegment 25346 (128 + 256) = _
  rw [primeSegment_append, interval_25346_checked, block_013_right_left_right_checked]
  rfl

def block_013_right_right_right : List Nat := interval_25858 ++ interval_25986
theorem block_013_right_right_right_checked : primeSegment 25858 176 = block_013_right_right_right := by
  change primeSegment 25858 (128 + 48) = _
  rw [primeSegment_append, interval_25858_checked, interval_25986_checked]
  rfl

def block_013_right_right : List Nat := interval_25730 ++ block_013_right_right_right
theorem block_013_right_right_checked : primeSegment 25730 304 = block_013_right_right := by
  change primeSegment 25730 (128 + 176) = _
  rw [primeSegment_append, interval_25730_checked, block_013_right_right_right_checked]
  rfl

def block_013_right : List Nat := block_013_right_left ++ block_013_right_right
theorem block_013_right_checked : primeSegment 25346 688 = block_013_right := by
  change primeSegment 25346 (384 + 304) = _
  rw [primeSegment_append, block_013_right_left_checked, block_013_right_right_checked]
  rfl

def block_013 : List Nat := block_013_left ++ block_013_right
theorem block_013_checked : primeSegment 24706 1328 = block_013 := by
  change primeSegment 24706 (640 + 688) = _
  rw [primeSegment_append, block_013_left_checked, block_013_right_checked]
  rfl

end ProjectEulerSolutions.PrimePairVertices
