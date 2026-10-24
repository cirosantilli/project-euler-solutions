import ProjectEulerSolutions.Lib.PrimePairCertificate

/-! Generated bounded vertex intervals; every list is kernel checked. -/
namespace ProjectEulerSolutions.PrimePairVertices
open PrimePairCertificate

def interval_16514 : List Nat := [16519, 16529, 16547, 16553, 16561, 16567, 16573, 16603, 16607, 16619, 16631, 16633]
theorem interval_16514_checked : primeSegment 16514 128 = interval_16514 := by
  rw [← computedSegment_eq_primeSegment 16514 128 (by decide)]
  decide

def interval_16642 : List Nat := [16649, 16651, 16657, 16661, 16673, 16691, 16693, 16699, 16703, 16729, 16741, 16747, 16759, 16763]
theorem interval_16642_checked : primeSegment 16642 128 = interval_16642 := by
  rw [← computedSegment_eq_primeSegment 16642 128 (by decide)]
  decide

def interval_16770 : List Nat := [16787, 16811, 16823, 16829, 16831, 16843, 16871, 16879, 16883, 16889]
theorem interval_16770_checked : primeSegment 16770 128 = interval_16770 := by
  rw [← computedSegment_eq_primeSegment 16770 128 (by decide)]
  decide

def interval_16898 : List Nat := [16901, 16903, 16921, 16927, 16931, 16937, 16943, 16963, 16979, 16981, 16987, 16993, 17011, 17021]
theorem interval_16898_checked : primeSegment 16898 128 = interval_16898 := by
  rw [← computedSegment_eq_primeSegment 16898 128 (by decide)]
  decide

def interval_17026 : List Nat := [17027, 17029, 17033, 17041, 17047, 17053, 17077, 17093, 17099, 17107, 17117, 17123, 17137]
theorem interval_17026_checked : primeSegment 17026 128 = interval_17026 := by
  rw [← computedSegment_eq_primeSegment 17026 128 (by decide)]
  decide

def interval_17154 : List Nat := [17159, 17167, 17183, 17189, 17191, 17203, 17207, 17209, 17231, 17239, 17257]
theorem interval_17154_checked : primeSegment 17154 128 = interval_17154 := by
  rw [← computedSegment_eq_primeSegment 17154 128 (by decide)]
  decide

def interval_17282 : List Nat := [17291, 17293, 17299, 17317, 17321, 17327, 17333, 17341, 17351, 17359, 17377, 17383, 17387, 17389, 17393, 17401]
theorem interval_17282_checked : primeSegment 17282 128 = interval_17282 := by
  rw [← computedSegment_eq_primeSegment 17282 128 (by decide)]
  decide

def interval_17410 : List Nat := [17417, 17419, 17431, 17443, 17449, 17467, 17471, 17477, 17483, 17489, 17491, 17497, 17509, 17519]
theorem interval_17410_checked : primeSegment 17410 128 = interval_17410 := by
  rw [← computedSegment_eq_primeSegment 17410 128 (by decide)]
  decide

def interval_17538 : List Nat := [17539, 17551, 17569, 17573, 17579, 17581, 17597, 17599, 17609, 17623, 17627, 17657, 17659]
theorem interval_17538_checked : primeSegment 17538 128 = interval_17538 := by
  rw [← computedSegment_eq_primeSegment 17538 128 (by decide)]
  decide

def interval_17666 : List Nat := [17669, 17681, 17683, 17707, 17713, 17729, 17737, 17747, 17749, 17761, 17783, 17789, 17791]
theorem interval_17666_checked : primeSegment 17666 128 = interval_17666 := by
  rw [← computedSegment_eq_primeSegment 17666 128 (by decide)]
  decide

def interval_17794 : List Nat := [17807, 17827, 17837, 17839, 17851, 17863, 17881, 17891, 17903, 17909, 17911, 17921]
theorem interval_17794_checked : primeSegment 17794 128 = interval_17794 := by
  rw [← computedSegment_eq_primeSegment 17794 128 (by decide)]
  decide

def interval_17922 : List Nat := [17923, 17929, 17939, 17957, 17959, 17971, 17977, 17981, 17987, 17989, 18013, 18041, 18043, 18047, 18049]
theorem interval_17922_checked : primeSegment 17922 128 = interval_17922 := by
  rw [← computedSegment_eq_primeSegment 17922 128 (by decide)]
  decide

def interval_18050 : List Nat := [18059, 18061, 18077, 18089, 18097, 18119, 18121, 18127, 18131, 18133, 18143, 18149, 18169]
theorem interval_18050_checked : primeSegment 18050 128 = interval_18050 := by
  rw [← computedSegment_eq_primeSegment 18050 128 (by decide)]
  decide

def interval_18178 : List Nat := [18181, 18191, 18199, 18211, 18217, 18223, 18229, 18233, 18251, 18253, 18257, 18269, 18287, 18289, 18301]
theorem interval_18178_checked : primeSegment 18178 128 = interval_18178 := by
  rw [← computedSegment_eq_primeSegment 18178 128 (by decide)]
  decide

def interval_18306 : List Nat := [18307, 18311, 18313, 18329, 18341, 18353, 18367, 18371, 18379, 18397, 18401, 18413, 18427, 18433]
theorem interval_18306_checked : primeSegment 18306 128 = interval_18306 := by
  rw [← computedSegment_eq_primeSegment 18306 128 (by decide)]
  decide

def interval_18434 : List Nat := [18439, 18443, 18451, 18457, 18461, 18481, 18493, 18503, 18517, 18521, 18523, 18539, 18541, 18553]
theorem interval_18434_checked : primeSegment 18434 128 = interval_18434 := by
  rw [← computedSegment_eq_primeSegment 18434 128 (by decide)]
  decide

def block_009_left_left_left : List Nat := interval_16514 ++ interval_16642
theorem block_009_left_left_left_checked : primeSegment 16514 256 = block_009_left_left_left := by
  change primeSegment 16514 (128 + 128) = _
  rw [primeSegment_append, interval_16514_checked, interval_16642_checked]
  rfl

def block_009_left_left_right : List Nat := interval_16770 ++ interval_16898
theorem block_009_left_left_right_checked : primeSegment 16770 256 = block_009_left_left_right := by
  change primeSegment 16770 (128 + 128) = _
  rw [primeSegment_append, interval_16770_checked, interval_16898_checked]
  rfl

def block_009_left_left : List Nat := block_009_left_left_left ++ block_009_left_left_right
theorem block_009_left_left_checked : primeSegment 16514 512 = block_009_left_left := by
  change primeSegment 16514 (256 + 256) = _
  rw [primeSegment_append, block_009_left_left_left_checked, block_009_left_left_right_checked]
  rfl

def block_009_left_right_left : List Nat := interval_17026 ++ interval_17154
theorem block_009_left_right_left_checked : primeSegment 17026 256 = block_009_left_right_left := by
  change primeSegment 17026 (128 + 128) = _
  rw [primeSegment_append, interval_17026_checked, interval_17154_checked]
  rfl

def block_009_left_right_right : List Nat := interval_17282 ++ interval_17410
theorem block_009_left_right_right_checked : primeSegment 17282 256 = block_009_left_right_right := by
  change primeSegment 17282 (128 + 128) = _
  rw [primeSegment_append, interval_17282_checked, interval_17410_checked]
  rfl

def block_009_left_right : List Nat := block_009_left_right_left ++ block_009_left_right_right
theorem block_009_left_right_checked : primeSegment 17026 512 = block_009_left_right := by
  change primeSegment 17026 (256 + 256) = _
  rw [primeSegment_append, block_009_left_right_left_checked, block_009_left_right_right_checked]
  rfl

def block_009_left : List Nat := block_009_left_left ++ block_009_left_right
theorem block_009_left_checked : primeSegment 16514 1024 = block_009_left := by
  change primeSegment 16514 (512 + 512) = _
  rw [primeSegment_append, block_009_left_left_checked, block_009_left_right_checked]
  rfl

def block_009_right_left_left : List Nat := interval_17538 ++ interval_17666
theorem block_009_right_left_left_checked : primeSegment 17538 256 = block_009_right_left_left := by
  change primeSegment 17538 (128 + 128) = _
  rw [primeSegment_append, interval_17538_checked, interval_17666_checked]
  rfl

def block_009_right_left_right : List Nat := interval_17794 ++ interval_17922
theorem block_009_right_left_right_checked : primeSegment 17794 256 = block_009_right_left_right := by
  change primeSegment 17794 (128 + 128) = _
  rw [primeSegment_append, interval_17794_checked, interval_17922_checked]
  rfl

def block_009_right_left : List Nat := block_009_right_left_left ++ block_009_right_left_right
theorem block_009_right_left_checked : primeSegment 17538 512 = block_009_right_left := by
  change primeSegment 17538 (256 + 256) = _
  rw [primeSegment_append, block_009_right_left_left_checked, block_009_right_left_right_checked]
  rfl

def block_009_right_right_left : List Nat := interval_18050 ++ interval_18178
theorem block_009_right_right_left_checked : primeSegment 18050 256 = block_009_right_right_left := by
  change primeSegment 18050 (128 + 128) = _
  rw [primeSegment_append, interval_18050_checked, interval_18178_checked]
  rfl

def block_009_right_right_right : List Nat := interval_18306 ++ interval_18434
theorem block_009_right_right_right_checked : primeSegment 18306 256 = block_009_right_right_right := by
  change primeSegment 18306 (128 + 128) = _
  rw [primeSegment_append, interval_18306_checked, interval_18434_checked]
  rfl

def block_009_right_right : List Nat := block_009_right_right_left ++ block_009_right_right_right
theorem block_009_right_right_checked : primeSegment 18050 512 = block_009_right_right := by
  change primeSegment 18050 (256 + 256) = _
  rw [primeSegment_append, block_009_right_right_left_checked, block_009_right_right_right_checked]
  rfl

def block_009_right : List Nat := block_009_right_left ++ block_009_right_right
theorem block_009_right_checked : primeSegment 17538 1024 = block_009_right := by
  change primeSegment 17538 (512 + 512) = _
  rw [primeSegment_append, block_009_right_left_checked, block_009_right_right_checked]
  rfl

def block_009 : List Nat := block_009_left ++ block_009_right
theorem block_009_checked : primeSegment 16514 2048 = block_009 := by
  change primeSegment 16514 (1024 + 1024) = _
  rw [primeSegment_append, block_009_left_checked, block_009_right_checked]
  rfl

end ProjectEulerSolutions.PrimePairVertices
