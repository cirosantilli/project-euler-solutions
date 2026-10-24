import ProjectEulerSolutions.Lib.PrimePairCertificate

/-! Generated bounded vertex intervals; every list is kernel checked. -/
namespace ProjectEulerSolutions.PrimePairVertices
open PrimePairCertificate

def interval_22658 : List Nat := [22669, 22679, 22691, 22697, 22699, 22709, 22717, 22721, 22727, 22739, 22741, 22751, 22769, 22777, 22783]
theorem interval_22658_checked : primeSegment 22658 128 = interval_22658 := by
  rw [← computedSegment_eq_primeSegment 22658 128 (by decide)]
  decide

def interval_22786 : List Nat := [22787, 22807, 22811, 22817, 22853, 22859, 22861, 22871, 22877, 22901, 22907]
theorem interval_22786_checked : primeSegment 22786 128 = interval_22786 := by
  rw [← computedSegment_eq_primeSegment 22786 128 (by decide)]
  decide

def interval_22914 : List Nat := [22921, 22937, 22943, 22961, 22963, 22973, 22993, 23003, 23011, 23017, 23021, 23027, 23029, 23039, 23041]
theorem interval_22914_checked : primeSegment 22914 128 = interval_22914 := by
  rw [← computedSegment_eq_primeSegment 22914 128 (by decide)]
  decide

def interval_23042 : List Nat := [23053, 23057, 23059, 23063, 23071, 23081, 23087, 23099, 23117, 23131, 23143, 23159, 23167]
theorem interval_23042_checked : primeSegment 23042 128 = interval_23042 := by
  rw [← computedSegment_eq_primeSegment 23042 128 (by decide)]
  decide

def interval_23170 : List Nat := [23173, 23189, 23197, 23201, 23203, 23209, 23227, 23251, 23269, 23279, 23291, 23293, 23297]
theorem interval_23170_checked : primeSegment 23170 128 = interval_23170 := by
  rw [← computedSegment_eq_primeSegment 23170 128 (by decide)]
  decide

def interval_23298 : List Nat := [23311, 23321, 23327, 23333, 23339, 23357, 23369, 23371, 23399, 23417]
theorem interval_23298_checked : primeSegment 23298 128 = interval_23298 := by
  rw [← computedSegment_eq_primeSegment 23298 128 (by decide)]
  decide

def interval_23426 : List Nat := [23431, 23447, 23459, 23473, 23497, 23509, 23531, 23537, 23539, 23549]
theorem interval_23426_checked : primeSegment 23426 128 = interval_23426 := by
  rw [← computedSegment_eq_primeSegment 23426 128 (by decide)]
  decide

def interval_23554 : List Nat := [23557, 23561, 23563, 23567, 23581, 23593, 23599, 23603, 23609, 23623, 23627, 23629, 23633, 23663, 23669, 23671, 23677]
theorem interval_23554_checked : primeSegment 23554 128 = interval_23554 := by
  rw [← computedSegment_eq_primeSegment 23554 128 (by decide)]
  decide

def interval_23682 : List Nat := [23687, 23689, 23719, 23741, 23743, 23747, 23753, 23761, 23767, 23773, 23789, 23801]
theorem interval_23682_checked : primeSegment 23682 128 = interval_23682 := by
  rw [← computedSegment_eq_primeSegment 23682 128 (by decide)]
  decide

def interval_23810 : List Nat := [23813, 23819, 23827, 23831, 23833, 23857, 23869, 23873, 23879, 23887, 23893, 23899, 23909, 23911, 23917, 23929]
theorem interval_23810_checked : primeSegment 23810 128 = interval_23810 := by
  rw [← computedSegment_eq_primeSegment 23810 128 (by decide)]
  decide

def interval_23938 : List Nat := [23957, 23971, 23977, 23981, 23993, 24001, 24007, 24019, 24023, 24029, 24043, 24049, 24061]
theorem interval_23938_checked : primeSegment 23938 128 = interval_23938 := by
  rw [← computedSegment_eq_primeSegment 23938 128 (by decide)]
  decide

def interval_24066 : List Nat := [24071, 24077, 24083, 24091, 24097, 24103, 24107, 24109, 24113, 24121, 24133, 24137, 24151, 24169, 24179, 24181]
theorem interval_24066_checked : primeSegment 24066 128 = interval_24066 := by
  rw [← computedSegment_eq_primeSegment 24066 128 (by decide)]
  decide

def interval_24194 : List Nat := [24197, 24203, 24223, 24229, 24239, 24247, 24251, 24281, 24317]
theorem interval_24194_checked : primeSegment 24194 128 = interval_24194 := by
  rw [← computedSegment_eq_primeSegment 24194 128 (by decide)]
  decide

def interval_24322 : List Nat := [24329, 24337, 24359, 24371, 24373, 24379, 24391, 24407, 24413, 24419, 24421, 24439, 24443]
theorem interval_24322_checked : primeSegment 24322 128 = interval_24322 := by
  rw [← computedSegment_eq_primeSegment 24322 128 (by decide)]
  decide

def interval_24450 : List Nat := [24469, 24473, 24481, 24499, 24509, 24517, 24527, 24533, 24547, 24551, 24571]
theorem interval_24450_checked : primeSegment 24450 128 = interval_24450 := by
  rw [← computedSegment_eq_primeSegment 24450 128 (by decide)]
  decide

def interval_24578 : List Nat := [24593, 24611, 24623, 24631, 24659, 24671, 24677, 24683, 24691, 24697]
theorem interval_24578_checked : primeSegment 24578 128 = interval_24578 := by
  rw [← computedSegment_eq_primeSegment 24578 128 (by decide)]
  decide

def block_012_left_left_left : List Nat := interval_22658 ++ interval_22786
theorem block_012_left_left_left_checked : primeSegment 22658 256 = block_012_left_left_left := by
  change primeSegment 22658 (128 + 128) = _
  rw [primeSegment_append, interval_22658_checked, interval_22786_checked]
  rfl

def block_012_left_left_right : List Nat := interval_22914 ++ interval_23042
theorem block_012_left_left_right_checked : primeSegment 22914 256 = block_012_left_left_right := by
  change primeSegment 22914 (128 + 128) = _
  rw [primeSegment_append, interval_22914_checked, interval_23042_checked]
  rfl

def block_012_left_left : List Nat := block_012_left_left_left ++ block_012_left_left_right
theorem block_012_left_left_checked : primeSegment 22658 512 = block_012_left_left := by
  change primeSegment 22658 (256 + 256) = _
  rw [primeSegment_append, block_012_left_left_left_checked, block_012_left_left_right_checked]
  rfl

def block_012_left_right_left : List Nat := interval_23170 ++ interval_23298
theorem block_012_left_right_left_checked : primeSegment 23170 256 = block_012_left_right_left := by
  change primeSegment 23170 (128 + 128) = _
  rw [primeSegment_append, interval_23170_checked, interval_23298_checked]
  rfl

def block_012_left_right_right : List Nat := interval_23426 ++ interval_23554
theorem block_012_left_right_right_checked : primeSegment 23426 256 = block_012_left_right_right := by
  change primeSegment 23426 (128 + 128) = _
  rw [primeSegment_append, interval_23426_checked, interval_23554_checked]
  rfl

def block_012_left_right : List Nat := block_012_left_right_left ++ block_012_left_right_right
theorem block_012_left_right_checked : primeSegment 23170 512 = block_012_left_right := by
  change primeSegment 23170 (256 + 256) = _
  rw [primeSegment_append, block_012_left_right_left_checked, block_012_left_right_right_checked]
  rfl

def block_012_left : List Nat := block_012_left_left ++ block_012_left_right
theorem block_012_left_checked : primeSegment 22658 1024 = block_012_left := by
  change primeSegment 22658 (512 + 512) = _
  rw [primeSegment_append, block_012_left_left_checked, block_012_left_right_checked]
  rfl

def block_012_right_left_left : List Nat := interval_23682 ++ interval_23810
theorem block_012_right_left_left_checked : primeSegment 23682 256 = block_012_right_left_left := by
  change primeSegment 23682 (128 + 128) = _
  rw [primeSegment_append, interval_23682_checked, interval_23810_checked]
  rfl

def block_012_right_left_right : List Nat := interval_23938 ++ interval_24066
theorem block_012_right_left_right_checked : primeSegment 23938 256 = block_012_right_left_right := by
  change primeSegment 23938 (128 + 128) = _
  rw [primeSegment_append, interval_23938_checked, interval_24066_checked]
  rfl

def block_012_right_left : List Nat := block_012_right_left_left ++ block_012_right_left_right
theorem block_012_right_left_checked : primeSegment 23682 512 = block_012_right_left := by
  change primeSegment 23682 (256 + 256) = _
  rw [primeSegment_append, block_012_right_left_left_checked, block_012_right_left_right_checked]
  rfl

def block_012_right_right_left : List Nat := interval_24194 ++ interval_24322
theorem block_012_right_right_left_checked : primeSegment 24194 256 = block_012_right_right_left := by
  change primeSegment 24194 (128 + 128) = _
  rw [primeSegment_append, interval_24194_checked, interval_24322_checked]
  rfl

def block_012_right_right_right : List Nat := interval_24450 ++ interval_24578
theorem block_012_right_right_right_checked : primeSegment 24450 256 = block_012_right_right_right := by
  change primeSegment 24450 (128 + 128) = _
  rw [primeSegment_append, interval_24450_checked, interval_24578_checked]
  rfl

def block_012_right_right : List Nat := block_012_right_right_left ++ block_012_right_right_right
theorem block_012_right_right_checked : primeSegment 24194 512 = block_012_right_right := by
  change primeSegment 24194 (256 + 256) = _
  rw [primeSegment_append, block_012_right_right_left_checked, block_012_right_right_right_checked]
  rfl

def block_012_right : List Nat := block_012_right_left ++ block_012_right_right
theorem block_012_right_checked : primeSegment 23682 1024 = block_012_right := by
  change primeSegment 23682 (512 + 512) = _
  rw [primeSegment_append, block_012_right_left_checked, block_012_right_right_checked]
  rfl

def block_012 : List Nat := block_012_left ++ block_012_right
theorem block_012_checked : primeSegment 22658 2048 = block_012 := by
  change primeSegment 22658 (1024 + 1024) = _
  rw [primeSegment_append, block_012_left_checked, block_012_right_checked]
  rfl

end ProjectEulerSolutions.PrimePairVertices
