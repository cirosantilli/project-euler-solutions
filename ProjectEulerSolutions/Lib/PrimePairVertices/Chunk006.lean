import ProjectEulerSolutions.Lib.PrimePairCertificate

/-! Generated bounded vertex intervals; every list is kernel checked. -/
namespace ProjectEulerSolutions.PrimePairVertices
open PrimePairCertificate

def interval_10370 : List Nat := [10391, 10399, 10427, 10429, 10433, 10453, 10457, 10459, 10463, 10477, 10487]
theorem interval_10370_checked : primeSegment 10370 128 = interval_10370 := by
  rw [← computedSegment_eq_primeSegment 10370 128 (by decide)]
  decide

def interval_10498 : List Nat := [10499, 10501, 10513, 10529, 10531, 10559, 10567, 10589, 10597, 10601, 10607, 10613]
theorem interval_10498_checked : primeSegment 10498 128 = interval_10498 := by
  rw [← computedSegment_eq_primeSegment 10498 128 (by decide)]
  decide

def interval_10626 : List Nat := [10627, 10631, 10639, 10651, 10657, 10663, 10667, 10687, 10691, 10709, 10711, 10723, 10729, 10733, 10739, 10753]
theorem interval_10626_checked : primeSegment 10626 128 = interval_10626 := by
  rw [← computedSegment_eq_primeSegment 10626 128 (by decide)]
  decide

def interval_10754 : List Nat := [10771, 10781, 10789, 10799, 10831, 10837, 10847, 10853, 10859, 10861, 10867]
theorem interval_10754_checked : primeSegment 10754 128 = interval_10754 := by
  rw [← computedSegment_eq_primeSegment 10754 128 (by decide)]
  decide

def interval_10882 : List Nat := [10883, 10889, 10891, 10903, 10909, 10937, 10939, 10949, 10957, 10973, 10979, 10987, 10993, 11003]
theorem interval_10882_checked : primeSegment 10882 128 = interval_10882 := by
  rw [← computedSegment_eq_primeSegment 10882 128 (by decide)]
  decide

def interval_11010 : List Nat := [11027, 11047, 11057, 11059, 11069, 11071, 11083, 11087, 11093, 11113, 11117, 11119, 11131]
theorem interval_11010_checked : primeSegment 11010 128 = interval_11010 := by
  rw [← computedSegment_eq_primeSegment 11010 128 (by decide)]
  decide

def interval_11138 : List Nat := [11149, 11159, 11161, 11171, 11173, 11177, 11197, 11213, 11239, 11243, 11251, 11257, 11261]
theorem interval_11138_checked : primeSegment 11138 128 = interval_11138 := by
  rw [← computedSegment_eq_primeSegment 11138 128 (by decide)]
  decide

def interval_11266 : List Nat := [11273, 11279, 11287, 11299, 11311, 11317, 11321, 11329, 11351, 11353, 11369, 11383, 11393]
theorem interval_11266_checked : primeSegment 11266 128 = interval_11266 := by
  rw [← computedSegment_eq_primeSegment 11266 128 (by decide)]
  decide

def interval_11394 : List Nat := [11399, 11411, 11423, 11437, 11443, 11447, 11467, 11471, 11483, 11489, 11491, 11497, 11503, 11519]
theorem interval_11394_checked : primeSegment 11394 128 = interval_11394 := by
  rw [← computedSegment_eq_primeSegment 11394 128 (by decide)]
  decide

def interval_11522 : List Nat := [11527, 11549, 11551, 11579, 11587, 11593, 11597, 11617, 11621, 11633]
theorem interval_11522_checked : primeSegment 11522 128 = interval_11522 := by
  rw [← computedSegment_eq_primeSegment 11522 128 (by decide)]
  decide

def interval_11650 : List Nat := [11657, 11677, 11681, 11689, 11699, 11701, 11717, 11719, 11731, 11743, 11777]
theorem interval_11650_checked : primeSegment 11650 128 = interval_11650 := by
  rw [← computedSegment_eq_primeSegment 11650 128 (by decide)]
  decide

def interval_11778 : List Nat := [11779, 11783, 11789, 11801, 11807, 11813, 11821, 11827, 11831, 11833, 11839, 11863, 11867, 11887, 11897, 11903]
theorem interval_11778_checked : primeSegment 11778 128 = interval_11778 := by
  rw [← computedSegment_eq_primeSegment 11778 128 (by decide)]
  decide

def interval_11906 : List Nat := [11909, 11923, 11927, 11933, 11939, 11941, 11953, 11959, 11969, 11971, 11981, 11987, 12007, 12011]
theorem interval_11906_checked : primeSegment 11906 128 = interval_11906 := by
  rw [← computedSegment_eq_primeSegment 11906 128 (by decide)]
  decide

def interval_12034 : List Nat := [12037, 12041, 12043, 12049, 12071, 12073, 12097, 12101, 12107, 12109, 12113, 12119, 12143, 12149, 12157, 12161]
theorem interval_12034_checked : primeSegment 12034 128 = interval_12034 := by
  rw [← computedSegment_eq_primeSegment 12034 128 (by decide)]
  decide

def interval_12162 : List Nat := [12163, 12197, 12203, 12211, 12227, 12239, 12241, 12251, 12253, 12263, 12269, 12277, 12281, 12289]
theorem interval_12162_checked : primeSegment 12162 128 = interval_12162 := by
  rw [← computedSegment_eq_primeSegment 12162 128 (by decide)]
  decide

def interval_12290 : List Nat := [12301, 12323, 12329, 12343, 12347, 12373, 12377, 12379, 12391, 12401, 12409, 12413]
theorem interval_12290_checked : primeSegment 12290 128 = interval_12290 := by
  rw [← computedSegment_eq_primeSegment 12290 128 (by decide)]
  decide

def block_006_left_left_left : List Nat := interval_10370 ++ interval_10498
theorem block_006_left_left_left_checked : primeSegment 10370 256 = block_006_left_left_left := by
  change primeSegment 10370 (128 + 128) = _
  rw [primeSegment_append, interval_10370_checked, interval_10498_checked]
  rfl

def block_006_left_left_right : List Nat := interval_10626 ++ interval_10754
theorem block_006_left_left_right_checked : primeSegment 10626 256 = block_006_left_left_right := by
  change primeSegment 10626 (128 + 128) = _
  rw [primeSegment_append, interval_10626_checked, interval_10754_checked]
  rfl

def block_006_left_left : List Nat := block_006_left_left_left ++ block_006_left_left_right
theorem block_006_left_left_checked : primeSegment 10370 512 = block_006_left_left := by
  change primeSegment 10370 (256 + 256) = _
  rw [primeSegment_append, block_006_left_left_left_checked, block_006_left_left_right_checked]
  rfl

def block_006_left_right_left : List Nat := interval_10882 ++ interval_11010
theorem block_006_left_right_left_checked : primeSegment 10882 256 = block_006_left_right_left := by
  change primeSegment 10882 (128 + 128) = _
  rw [primeSegment_append, interval_10882_checked, interval_11010_checked]
  rfl

def block_006_left_right_right : List Nat := interval_11138 ++ interval_11266
theorem block_006_left_right_right_checked : primeSegment 11138 256 = block_006_left_right_right := by
  change primeSegment 11138 (128 + 128) = _
  rw [primeSegment_append, interval_11138_checked, interval_11266_checked]
  rfl

def block_006_left_right : List Nat := block_006_left_right_left ++ block_006_left_right_right
theorem block_006_left_right_checked : primeSegment 10882 512 = block_006_left_right := by
  change primeSegment 10882 (256 + 256) = _
  rw [primeSegment_append, block_006_left_right_left_checked, block_006_left_right_right_checked]
  rfl

def block_006_left : List Nat := block_006_left_left ++ block_006_left_right
theorem block_006_left_checked : primeSegment 10370 1024 = block_006_left := by
  change primeSegment 10370 (512 + 512) = _
  rw [primeSegment_append, block_006_left_left_checked, block_006_left_right_checked]
  rfl

def block_006_right_left_left : List Nat := interval_11394 ++ interval_11522
theorem block_006_right_left_left_checked : primeSegment 11394 256 = block_006_right_left_left := by
  change primeSegment 11394 (128 + 128) = _
  rw [primeSegment_append, interval_11394_checked, interval_11522_checked]
  rfl

def block_006_right_left_right : List Nat := interval_11650 ++ interval_11778
theorem block_006_right_left_right_checked : primeSegment 11650 256 = block_006_right_left_right := by
  change primeSegment 11650 (128 + 128) = _
  rw [primeSegment_append, interval_11650_checked, interval_11778_checked]
  rfl

def block_006_right_left : List Nat := block_006_right_left_left ++ block_006_right_left_right
theorem block_006_right_left_checked : primeSegment 11394 512 = block_006_right_left := by
  change primeSegment 11394 (256 + 256) = _
  rw [primeSegment_append, block_006_right_left_left_checked, block_006_right_left_right_checked]
  rfl

def block_006_right_right_left : List Nat := interval_11906 ++ interval_12034
theorem block_006_right_right_left_checked : primeSegment 11906 256 = block_006_right_right_left := by
  change primeSegment 11906 (128 + 128) = _
  rw [primeSegment_append, interval_11906_checked, interval_12034_checked]
  rfl

def block_006_right_right_right : List Nat := interval_12162 ++ interval_12290
theorem block_006_right_right_right_checked : primeSegment 12162 256 = block_006_right_right_right := by
  change primeSegment 12162 (128 + 128) = _
  rw [primeSegment_append, interval_12162_checked, interval_12290_checked]
  rfl

def block_006_right_right : List Nat := block_006_right_right_left ++ block_006_right_right_right
theorem block_006_right_right_checked : primeSegment 11906 512 = block_006_right_right := by
  change primeSegment 11906 (256 + 256) = _
  rw [primeSegment_append, block_006_right_right_left_checked, block_006_right_right_right_checked]
  rfl

def block_006_right : List Nat := block_006_right_left ++ block_006_right_right
theorem block_006_right_checked : primeSegment 11394 1024 = block_006_right := by
  change primeSegment 11394 (512 + 512) = _
  rw [primeSegment_append, block_006_right_left_checked, block_006_right_right_checked]
  rfl

def block_006 : List Nat := block_006_left ++ block_006_right
theorem block_006_checked : primeSegment 10370 2048 = block_006 := by
  change primeSegment 10370 (1024 + 1024) = _
  rw [primeSegment_append, block_006_left_checked, block_006_right_checked]
  rfl

end ProjectEulerSolutions.PrimePairVertices
