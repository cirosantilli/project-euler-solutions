import ProjectEulerSolutions.Lib.PrimePairCertificate

/-! Generated bounded vertex intervals; every list is kernel checked. -/
namespace ProjectEulerSolutions.PrimePairVertices
open PrimePairCertificate

def interval_20610 : List Nat := [20611, 20627, 20639, 20641, 20663, 20681, 20693, 20707, 20717, 20719, 20731]
theorem interval_20610_checked : primeSegment 20610 128 = interval_20610 := by
  rw [← computedSegment_eq_primeSegment 20610 128 (by decide)]
  decide

def interval_20738 : List Nat := [20743, 20747, 20749, 20753, 20759, 20771, 20773, 20789, 20807, 20809, 20849, 20857]
theorem interval_20738_checked : primeSegment 20738 128 = interval_20738 := by
  rw [← computedSegment_eq_primeSegment 20738 128 (by decide)]
  decide

def interval_20866 : List Nat := [20873, 20879, 20887, 20897, 20899, 20903, 20921, 20929, 20939, 20947, 20959, 20963, 20981, 20983]
theorem interval_20866_checked : primeSegment 20866 128 = interval_20866 := by
  rw [← computedSegment_eq_primeSegment 20866 128 (by decide)]
  decide

def interval_20994 : List Nat := [21001, 21011, 21013, 21017, 21019, 21023, 21031, 21059, 21061, 21067, 21089, 21101, 21107, 21121]
theorem interval_20994_checked : primeSegment 20994 128 = interval_20994 := by
  rw [← computedSegment_eq_primeSegment 20994 128 (by decide)]
  decide

def interval_21122 : List Nat := [21139, 21143, 21149, 21157, 21163, 21169, 21179, 21187, 21191, 21193, 21211, 21221, 21227, 21247]
theorem interval_21122_checked : primeSegment 21122 128 = interval_21122 := by
  rw [← computedSegment_eq_primeSegment 21122 128 (by decide)]
  decide

def interval_21250 : List Nat := [21269, 21277, 21283, 21313, 21317, 21319, 21323, 21341, 21347, 21377]
theorem interval_21250_checked : primeSegment 21250 128 = interval_21250 := by
  rw [← computedSegment_eq_primeSegment 21250 128 (by decide)]
  decide

def interval_21378 : List Nat := [21379, 21383, 21391, 21397, 21401, 21407, 21419, 21433, 21467, 21481, 21487, 21491, 21493, 21499, 21503]
theorem interval_21378_checked : primeSegment 21378 128 = interval_21378 := by
  rw [← computedSegment_eq_primeSegment 21378 128 (by decide)]
  decide

def interval_21506 : List Nat := [21517, 21521, 21523, 21529, 21557, 21559, 21563, 21569, 21577, 21587, 21589, 21599, 21601, 21611, 21613, 21617]
theorem interval_21506_checked : primeSegment 21506 128 = interval_21506 := by
  rw [← computedSegment_eq_primeSegment 21506 128 (by decide)]
  decide

def interval_21634 : List Nat := [21647, 21649, 21661, 21673, 21683, 21701, 21713, 21727, 21737, 21739, 21751, 21757]
theorem interval_21634_checked : primeSegment 21634 128 = interval_21634 := by
  rw [← computedSegment_eq_primeSegment 21634 128 (by decide)]
  decide

def interval_21762 : List Nat := [21767, 21773, 21787, 21799, 21803, 21817, 21821, 21839, 21841, 21851, 21859, 21863, 21871, 21881]
theorem interval_21762_checked : primeSegment 21762 128 = interval_21762 := by
  rw [← computedSegment_eq_primeSegment 21762 128 (by decide)]
  decide

def interval_21890 : List Nat := [21893, 21911, 21929, 21937, 21943, 21961, 21977, 21991, 21997, 22003, 22013]
theorem interval_21890_checked : primeSegment 21890 128 = interval_21890 := by
  rw [← computedSegment_eq_primeSegment 21890 128 (by decide)]
  decide

def interval_22018 : List Nat := [22027, 22031, 22037, 22039, 22051, 22063, 22067, 22073, 22079, 22091, 22093, 22109, 22111, 22123, 22129, 22133]
theorem interval_22018_checked : primeSegment 22018 128 = interval_22018 := by
  rw [← computedSegment_eq_primeSegment 22018 128 (by decide)]
  decide

def interval_22146 : List Nat := [22147, 22153, 22157, 22159, 22171, 22189, 22193, 22229, 22247, 22259, 22271, 22273]
theorem interval_22146_checked : primeSegment 22146 128 = interval_22146 := by
  rw [← computedSegment_eq_primeSegment 22146 128 (by decide)]
  decide

def interval_22274 : List Nat := [22277, 22279, 22283, 22291, 22303, 22307, 22343, 22349, 22367, 22369, 22381, 22391, 22397]
theorem interval_22274_checked : primeSegment 22274 128 = interval_22274 := by
  rw [← computedSegment_eq_primeSegment 22274 128 (by decide)]
  decide

def interval_22402 : List Nat := [22409, 22433, 22441, 22447, 22453, 22469, 22481, 22483, 22501, 22511]
theorem interval_22402_checked : primeSegment 22402 128 = interval_22402 := by
  rw [← computedSegment_eq_primeSegment 22402 128 (by decide)]
  decide

def interval_22530 : List Nat := [22531, 22541, 22543, 22549, 22567, 22571, 22573, 22613, 22619, 22621, 22637, 22639, 22643, 22651]
theorem interval_22530_checked : primeSegment 22530 128 = interval_22530 := by
  rw [← computedSegment_eq_primeSegment 22530 128 (by decide)]
  decide

def block_011_left_left_left : List Nat := interval_20610 ++ interval_20738
theorem block_011_left_left_left_checked : primeSegment 20610 256 = block_011_left_left_left := by
  change primeSegment 20610 (128 + 128) = _
  rw [primeSegment_append, interval_20610_checked, interval_20738_checked]
  rfl

def block_011_left_left_right : List Nat := interval_20866 ++ interval_20994
theorem block_011_left_left_right_checked : primeSegment 20866 256 = block_011_left_left_right := by
  change primeSegment 20866 (128 + 128) = _
  rw [primeSegment_append, interval_20866_checked, interval_20994_checked]
  rfl

def block_011_left_left : List Nat := block_011_left_left_left ++ block_011_left_left_right
theorem block_011_left_left_checked : primeSegment 20610 512 = block_011_left_left := by
  change primeSegment 20610 (256 + 256) = _
  rw [primeSegment_append, block_011_left_left_left_checked, block_011_left_left_right_checked]
  rfl

def block_011_left_right_left : List Nat := interval_21122 ++ interval_21250
theorem block_011_left_right_left_checked : primeSegment 21122 256 = block_011_left_right_left := by
  change primeSegment 21122 (128 + 128) = _
  rw [primeSegment_append, interval_21122_checked, interval_21250_checked]
  rfl

def block_011_left_right_right : List Nat := interval_21378 ++ interval_21506
theorem block_011_left_right_right_checked : primeSegment 21378 256 = block_011_left_right_right := by
  change primeSegment 21378 (128 + 128) = _
  rw [primeSegment_append, interval_21378_checked, interval_21506_checked]
  rfl

def block_011_left_right : List Nat := block_011_left_right_left ++ block_011_left_right_right
theorem block_011_left_right_checked : primeSegment 21122 512 = block_011_left_right := by
  change primeSegment 21122 (256 + 256) = _
  rw [primeSegment_append, block_011_left_right_left_checked, block_011_left_right_right_checked]
  rfl

def block_011_left : List Nat := block_011_left_left ++ block_011_left_right
theorem block_011_left_checked : primeSegment 20610 1024 = block_011_left := by
  change primeSegment 20610 (512 + 512) = _
  rw [primeSegment_append, block_011_left_left_checked, block_011_left_right_checked]
  rfl

def block_011_right_left_left : List Nat := interval_21634 ++ interval_21762
theorem block_011_right_left_left_checked : primeSegment 21634 256 = block_011_right_left_left := by
  change primeSegment 21634 (128 + 128) = _
  rw [primeSegment_append, interval_21634_checked, interval_21762_checked]
  rfl

def block_011_right_left_right : List Nat := interval_21890 ++ interval_22018
theorem block_011_right_left_right_checked : primeSegment 21890 256 = block_011_right_left_right := by
  change primeSegment 21890 (128 + 128) = _
  rw [primeSegment_append, interval_21890_checked, interval_22018_checked]
  rfl

def block_011_right_left : List Nat := block_011_right_left_left ++ block_011_right_left_right
theorem block_011_right_left_checked : primeSegment 21634 512 = block_011_right_left := by
  change primeSegment 21634 (256 + 256) = _
  rw [primeSegment_append, block_011_right_left_left_checked, block_011_right_left_right_checked]
  rfl

def block_011_right_right_left : List Nat := interval_22146 ++ interval_22274
theorem block_011_right_right_left_checked : primeSegment 22146 256 = block_011_right_right_left := by
  change primeSegment 22146 (128 + 128) = _
  rw [primeSegment_append, interval_22146_checked, interval_22274_checked]
  rfl

def block_011_right_right_right : List Nat := interval_22402 ++ interval_22530
theorem block_011_right_right_right_checked : primeSegment 22402 256 = block_011_right_right_right := by
  change primeSegment 22402 (128 + 128) = _
  rw [primeSegment_append, interval_22402_checked, interval_22530_checked]
  rfl

def block_011_right_right : List Nat := block_011_right_right_left ++ block_011_right_right_right
theorem block_011_right_right_checked : primeSegment 22146 512 = block_011_right_right := by
  change primeSegment 22146 (256 + 256) = _
  rw [primeSegment_append, block_011_right_right_left_checked, block_011_right_right_right_checked]
  rfl

def block_011_right : List Nat := block_011_right_left ++ block_011_right_right
theorem block_011_right_checked : primeSegment 21634 1024 = block_011_right := by
  change primeSegment 21634 (512 + 512) = _
  rw [primeSegment_append, block_011_right_left_checked, block_011_right_right_checked]
  rfl

def block_011 : List Nat := block_011_left ++ block_011_right
theorem block_011_checked : primeSegment 20610 2048 = block_011 := by
  change primeSegment 20610 (1024 + 1024) = _
  rw [primeSegment_append, block_011_left_checked, block_011_right_checked]
  rfl

end ProjectEulerSolutions.PrimePairVertices
