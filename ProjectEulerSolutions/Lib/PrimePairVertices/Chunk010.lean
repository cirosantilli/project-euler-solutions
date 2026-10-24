import ProjectEulerSolutions.Lib.PrimePairCertificate

/-! Generated bounded vertex intervals; every list is kernel checked. -/
namespace ProjectEulerSolutions.PrimePairVertices
open PrimePairCertificate

def interval_18562 : List Nat := [18583, 18587, 18593, 18617, 18637, 18661, 18671, 18679]
theorem interval_18562_checked : primeSegment 18562 128 = interval_18562 := by
  rw [← computedSegment_eq_primeSegment 18562 128 (by decide)]
  decide

def interval_18690 : List Nat := [18691, 18701, 18713, 18719, 18731, 18743, 18749, 18757, 18773, 18787, 18793, 18797, 18803]
theorem interval_18690_checked : primeSegment 18690 128 = interval_18690 := by
  rw [← computedSegment_eq_primeSegment 18690 128 (by decide)]
  decide

def interval_18818 : List Nat := [18839, 18859, 18869, 18899, 18911, 18913, 18917, 18919]
theorem interval_18818_checked : primeSegment 18818 128 = interval_18818 := by
  rw [← computedSegment_eq_primeSegment 18818 128 (by decide)]
  decide

def interval_18946 : List Nat := [18947, 18959, 18973, 18979, 19001, 19009, 19013, 19031, 19037, 19051, 19069, 19073]
theorem interval_18946_checked : primeSegment 18946 128 = interval_18946 := by
  rw [← computedSegment_eq_primeSegment 18946 128 (by decide)]
  decide

def interval_19074 : List Nat := [19079, 19081, 19087, 19121, 19139, 19141, 19157, 19163, 19181, 19183]
theorem interval_19074_checked : primeSegment 19074 128 = interval_19074 := by
  rw [← computedSegment_eq_primeSegment 19074 128 (by decide)]
  decide

def interval_19202 : List Nat := [19207, 19211, 19213, 19219, 19231, 19237, 19249, 19259, 19267, 19273, 19289, 19301, 19309, 19319]
theorem interval_19202_checked : primeSegment 19202 128 = interval_19202 := by
  rw [← computedSegment_eq_primeSegment 19202 128 (by decide)]
  decide

def interval_19330 : List Nat := [19333, 19373, 19379, 19381, 19387, 19391, 19403, 19417, 19421, 19423, 19427, 19429, 19433, 19441, 19447, 19457]
theorem interval_19330_checked : primeSegment 19330 128 = interval_19330 := by
  rw [← computedSegment_eq_primeSegment 19330 128 (by decide)]
  decide

def interval_19458 : List Nat := [19463, 19469, 19471, 19477, 19483, 19489, 19501, 19507, 19531, 19541, 19543, 19553, 19559, 19571, 19577, 19583]
theorem interval_19458_checked : primeSegment 19458 128 = interval_19458 := by
  rw [← computedSegment_eq_primeSegment 19458 128 (by decide)]
  decide

def interval_19586 : List Nat := [19597, 19603, 19609, 19661, 19681, 19687, 19697, 19699, 19709]
theorem interval_19586_checked : primeSegment 19586 128 = interval_19586 := by
  rw [← computedSegment_eq_primeSegment 19586 128 (by decide)]
  decide

def interval_19714 : List Nat := [19717, 19727, 19739, 19751, 19753, 19759, 19763, 19777, 19793, 19801, 19813, 19819, 19841]
theorem interval_19714_checked : primeSegment 19714 128 = interval_19714 := by
  rw [← computedSegment_eq_primeSegment 19714 128 (by decide)]
  decide

def interval_19842 : List Nat := [19843, 19853, 19861, 19867, 19889, 19891, 19913, 19919, 19927, 19937, 19949, 19961, 19963]
theorem interval_19842_checked : primeSegment 19842 128 = interval_19842 := by
  rw [← computedSegment_eq_primeSegment 19842 128 (by decide)]
  decide

def interval_19970 : List Nat := [19973, 19979, 19991, 19993, 19997, 20011, 20021, 20023, 20029, 20047, 20051, 20063, 20071, 20089]
theorem interval_19970_checked : primeSegment 19970 128 = interval_19970 := by
  rw [← computedSegment_eq_primeSegment 19970 128 (by decide)]
  decide

def interval_20098 : List Nat := [20101, 20107, 20113, 20117, 20123, 20129, 20143, 20147, 20149, 20161, 20173, 20177, 20183, 20201, 20219]
theorem interval_20098_checked : primeSegment 20098 128 = interval_20098 := by
  rw [← computedSegment_eq_primeSegment 20098 128 (by decide)]
  decide

def interval_20226 : List Nat := [20231, 20233, 20249, 20261, 20269, 20287, 20297, 20323, 20327, 20333, 20341, 20347, 20353]
theorem interval_20226_checked : primeSegment 20226 128 = interval_20226 := by
  rw [← computedSegment_eq_primeSegment 20226 128 (by decide)]
  decide

def interval_20354 : List Nat := [20357, 20359, 20369, 20389, 20393, 20399, 20407, 20411, 20431, 20441, 20443, 20477, 20479]
theorem interval_20354_checked : primeSegment 20354 128 = interval_20354 := by
  rw [← computedSegment_eq_primeSegment 20354 128 (by decide)]
  decide

def interval_20482 : List Nat := [20483, 20507, 20509, 20521, 20533, 20543, 20549, 20551, 20563, 20593, 20599]
theorem interval_20482_checked : primeSegment 20482 128 = interval_20482 := by
  rw [← computedSegment_eq_primeSegment 20482 128 (by decide)]
  decide

def block_010_left_left_left : List Nat := interval_18562 ++ interval_18690
theorem block_010_left_left_left_checked : primeSegment 18562 256 = block_010_left_left_left := by
  change primeSegment 18562 (128 + 128) = _
  rw [primeSegment_append, interval_18562_checked, interval_18690_checked]
  rfl

def block_010_left_left_right : List Nat := interval_18818 ++ interval_18946
theorem block_010_left_left_right_checked : primeSegment 18818 256 = block_010_left_left_right := by
  change primeSegment 18818 (128 + 128) = _
  rw [primeSegment_append, interval_18818_checked, interval_18946_checked]
  rfl

def block_010_left_left : List Nat := block_010_left_left_left ++ block_010_left_left_right
theorem block_010_left_left_checked : primeSegment 18562 512 = block_010_left_left := by
  change primeSegment 18562 (256 + 256) = _
  rw [primeSegment_append, block_010_left_left_left_checked, block_010_left_left_right_checked]
  rfl

def block_010_left_right_left : List Nat := interval_19074 ++ interval_19202
theorem block_010_left_right_left_checked : primeSegment 19074 256 = block_010_left_right_left := by
  change primeSegment 19074 (128 + 128) = _
  rw [primeSegment_append, interval_19074_checked, interval_19202_checked]
  rfl

def block_010_left_right_right : List Nat := interval_19330 ++ interval_19458
theorem block_010_left_right_right_checked : primeSegment 19330 256 = block_010_left_right_right := by
  change primeSegment 19330 (128 + 128) = _
  rw [primeSegment_append, interval_19330_checked, interval_19458_checked]
  rfl

def block_010_left_right : List Nat := block_010_left_right_left ++ block_010_left_right_right
theorem block_010_left_right_checked : primeSegment 19074 512 = block_010_left_right := by
  change primeSegment 19074 (256 + 256) = _
  rw [primeSegment_append, block_010_left_right_left_checked, block_010_left_right_right_checked]
  rfl

def block_010_left : List Nat := block_010_left_left ++ block_010_left_right
theorem block_010_left_checked : primeSegment 18562 1024 = block_010_left := by
  change primeSegment 18562 (512 + 512) = _
  rw [primeSegment_append, block_010_left_left_checked, block_010_left_right_checked]
  rfl

def block_010_right_left_left : List Nat := interval_19586 ++ interval_19714
theorem block_010_right_left_left_checked : primeSegment 19586 256 = block_010_right_left_left := by
  change primeSegment 19586 (128 + 128) = _
  rw [primeSegment_append, interval_19586_checked, interval_19714_checked]
  rfl

def block_010_right_left_right : List Nat := interval_19842 ++ interval_19970
theorem block_010_right_left_right_checked : primeSegment 19842 256 = block_010_right_left_right := by
  change primeSegment 19842 (128 + 128) = _
  rw [primeSegment_append, interval_19842_checked, interval_19970_checked]
  rfl

def block_010_right_left : List Nat := block_010_right_left_left ++ block_010_right_left_right
theorem block_010_right_left_checked : primeSegment 19586 512 = block_010_right_left := by
  change primeSegment 19586 (256 + 256) = _
  rw [primeSegment_append, block_010_right_left_left_checked, block_010_right_left_right_checked]
  rfl

def block_010_right_right_left : List Nat := interval_20098 ++ interval_20226
theorem block_010_right_right_left_checked : primeSegment 20098 256 = block_010_right_right_left := by
  change primeSegment 20098 (128 + 128) = _
  rw [primeSegment_append, interval_20098_checked, interval_20226_checked]
  rfl

def block_010_right_right_right : List Nat := interval_20354 ++ interval_20482
theorem block_010_right_right_right_checked : primeSegment 20354 256 = block_010_right_right_right := by
  change primeSegment 20354 (128 + 128) = _
  rw [primeSegment_append, interval_20354_checked, interval_20482_checked]
  rfl

def block_010_right_right : List Nat := block_010_right_right_left ++ block_010_right_right_right
theorem block_010_right_right_checked : primeSegment 20098 512 = block_010_right_right := by
  change primeSegment 20098 (256 + 256) = _
  rw [primeSegment_append, block_010_right_right_left_checked, block_010_right_right_right_checked]
  rfl

def block_010_right : List Nat := block_010_right_left ++ block_010_right_right
theorem block_010_right_checked : primeSegment 19586 1024 = block_010_right := by
  change primeSegment 19586 (512 + 512) = _
  rw [primeSegment_append, block_010_right_left_checked, block_010_right_right_checked]
  rfl

def block_010 : List Nat := block_010_left ++ block_010_right
theorem block_010_checked : primeSegment 18562 2048 = block_010 := by
  change primeSegment 18562 (1024 + 1024) = _
  rw [primeSegment_append, block_010_left_checked, block_010_right_checked]
  rfl

end ProjectEulerSolutions.PrimePairVertices
