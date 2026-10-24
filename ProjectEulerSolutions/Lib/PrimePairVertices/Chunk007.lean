import ProjectEulerSolutions.Lib.PrimePairCertificate

/-! Generated bounded vertex intervals; every list is kernel checked. -/
namespace ProjectEulerSolutions.PrimePairVertices
open PrimePairCertificate

def interval_12418 : List Nat := [12421, 12433, 12437, 12451, 12457, 12473, 12479, 12487, 12491, 12497, 12503, 12511, 12517, 12527, 12539, 12541]
theorem interval_12418_checked : primeSegment 12418 128 = interval_12418 := by
  rw [← computedSegment_eq_primeSegment 12418 128 (by decide)]
  decide

def interval_12546 : List Nat := [12547, 12553, 12569, 12577, 12583, 12589, 12601, 12611, 12613, 12619, 12637, 12641, 12647, 12653, 12659, 12671]
theorem interval_12546_checked : primeSegment 12546 128 = interval_12546 := by
  rw [← computedSegment_eq_primeSegment 12546 128 (by decide)]
  decide

def interval_12674 : List Nat := [12689, 12697, 12703, 12713, 12721, 12739, 12743, 12757, 12763, 12781, 12791, 12799]
theorem interval_12674_checked : primeSegment 12674 128 = interval_12674 := by
  rw [← computedSegment_eq_primeSegment 12674 128 (by decide)]
  decide

def interval_12802 : List Nat := [12809, 12821, 12823, 12829, 12841, 12853, 12889, 12893, 12899, 12907, 12911, 12917, 12919, 12923]
theorem interval_12802_checked : primeSegment 12802 128 = interval_12802 := by
  rw [← computedSegment_eq_primeSegment 12802 128 (by decide)]
  decide

def interval_12930 : List Nat := [12941, 12953, 12959, 12967, 12973, 12979, 12983, 13001, 13003, 13007, 13009, 13033, 13037, 13043, 13049]
theorem interval_12930_checked : primeSegment 12930 128 = interval_12930 := by
  rw [← computedSegment_eq_primeSegment 12930 128 (by decide)]
  decide

def interval_13058 : List Nat := [13063, 13093, 13099, 13103, 13109, 13121, 13127, 13147, 13151, 13159, 13163, 13171, 13177, 13183]
theorem interval_13058_checked : primeSegment 13058 128 = interval_13058 := by
  rw [← computedSegment_eq_primeSegment 13058 128 (by decide)]
  decide

def interval_13186 : List Nat := [13187, 13217, 13219, 13229, 13241, 13249, 13259, 13267, 13291, 13297, 13309, 13313]
theorem interval_13186_checked : primeSegment 13186 128 = interval_13186 := by
  rw [← computedSegment_eq_primeSegment 13186 128 (by decide)]
  decide

def interval_13314 : List Nat := [13327, 13331, 13337, 13339, 13367, 13381, 13397, 13399, 13411, 13417, 13421, 13441]
theorem interval_13314_checked : primeSegment 13314 128 = interval_13314 := by
  rw [← computedSegment_eq_primeSegment 13314 128 (by decide)]
  decide

def interval_13442 : List Nat := [13451, 13457, 13463, 13469, 13477, 13487, 13499, 13513, 13523, 13537, 13553, 13567]
theorem interval_13442_checked : primeSegment 13442 128 = interval_13442 := by
  rw [← computedSegment_eq_primeSegment 13442 128 (by decide)]
  decide

def interval_13570 : List Nat := [13577, 13591, 13597, 13613, 13619, 13627, 13633, 13649, 13669, 13679, 13681, 13687, 13691, 13693, 13697]
theorem interval_13570_checked : primeSegment 13570 128 = interval_13570 := by
  rw [← computedSegment_eq_primeSegment 13570 128 (by decide)]
  decide

def interval_13698 : List Nat := [13709, 13711, 13721, 13723, 13729, 13751, 13757, 13759, 13763, 13781, 13789, 13799, 13807]
theorem interval_13698_checked : primeSegment 13698 128 = interval_13698 := by
  rw [← computedSegment_eq_primeSegment 13698 128 (by decide)]
  decide

def interval_13826 : List Nat := [13829, 13831, 13841, 13859, 13873, 13877, 13879, 13883, 13901, 13903, 13907, 13913, 13921, 13931, 13933]
theorem interval_13826_checked : primeSegment 13826 128 = interval_13826 := by
  rw [← computedSegment_eq_primeSegment 13826 128 (by decide)]
  decide

def interval_13954 : List Nat := [13963, 13967, 13997, 13999, 14009, 14011, 14029, 14033, 14051, 14057, 14071, 14081]
theorem interval_13954_checked : primeSegment 13954 128 = interval_13954 := by
  rw [← computedSegment_eq_primeSegment 13954 128 (by decide)]
  decide

def interval_14082 : List Nat := [14083, 14087, 14107, 14143, 14149, 14153, 14159, 14173, 14177, 14197, 14207]
theorem interval_14082_checked : primeSegment 14082 128 = interval_14082 := by
  rw [← computedSegment_eq_primeSegment 14082 128 (by decide)]
  decide

def interval_14210 : List Nat := [14221, 14243, 14249, 14251, 14281, 14293, 14303, 14321, 14323, 14327]
theorem interval_14210_checked : primeSegment 14210 128 = interval_14210 := by
  rw [← computedSegment_eq_primeSegment 14210 128 (by decide)]
  decide

def interval_14338 : List Nat := [14341, 14347, 14369, 14387, 14389, 14401, 14407, 14411, 14419, 14423, 14431, 14437, 14447, 14449, 14461]
theorem interval_14338_checked : primeSegment 14338 128 = interval_14338 := by
  rw [← computedSegment_eq_primeSegment 14338 128 (by decide)]
  decide

def block_007_left_left_left : List Nat := interval_12418 ++ interval_12546
theorem block_007_left_left_left_checked : primeSegment 12418 256 = block_007_left_left_left := by
  change primeSegment 12418 (128 + 128) = _
  rw [primeSegment_append, interval_12418_checked, interval_12546_checked]
  rfl

def block_007_left_left_right : List Nat := interval_12674 ++ interval_12802
theorem block_007_left_left_right_checked : primeSegment 12674 256 = block_007_left_left_right := by
  change primeSegment 12674 (128 + 128) = _
  rw [primeSegment_append, interval_12674_checked, interval_12802_checked]
  rfl

def block_007_left_left : List Nat := block_007_left_left_left ++ block_007_left_left_right
theorem block_007_left_left_checked : primeSegment 12418 512 = block_007_left_left := by
  change primeSegment 12418 (256 + 256) = _
  rw [primeSegment_append, block_007_left_left_left_checked, block_007_left_left_right_checked]
  rfl

def block_007_left_right_left : List Nat := interval_12930 ++ interval_13058
theorem block_007_left_right_left_checked : primeSegment 12930 256 = block_007_left_right_left := by
  change primeSegment 12930 (128 + 128) = _
  rw [primeSegment_append, interval_12930_checked, interval_13058_checked]
  rfl

def block_007_left_right_right : List Nat := interval_13186 ++ interval_13314
theorem block_007_left_right_right_checked : primeSegment 13186 256 = block_007_left_right_right := by
  change primeSegment 13186 (128 + 128) = _
  rw [primeSegment_append, interval_13186_checked, interval_13314_checked]
  rfl

def block_007_left_right : List Nat := block_007_left_right_left ++ block_007_left_right_right
theorem block_007_left_right_checked : primeSegment 12930 512 = block_007_left_right := by
  change primeSegment 12930 (256 + 256) = _
  rw [primeSegment_append, block_007_left_right_left_checked, block_007_left_right_right_checked]
  rfl

def block_007_left : List Nat := block_007_left_left ++ block_007_left_right
theorem block_007_left_checked : primeSegment 12418 1024 = block_007_left := by
  change primeSegment 12418 (512 + 512) = _
  rw [primeSegment_append, block_007_left_left_checked, block_007_left_right_checked]
  rfl

def block_007_right_left_left : List Nat := interval_13442 ++ interval_13570
theorem block_007_right_left_left_checked : primeSegment 13442 256 = block_007_right_left_left := by
  change primeSegment 13442 (128 + 128) = _
  rw [primeSegment_append, interval_13442_checked, interval_13570_checked]
  rfl

def block_007_right_left_right : List Nat := interval_13698 ++ interval_13826
theorem block_007_right_left_right_checked : primeSegment 13698 256 = block_007_right_left_right := by
  change primeSegment 13698 (128 + 128) = _
  rw [primeSegment_append, interval_13698_checked, interval_13826_checked]
  rfl

def block_007_right_left : List Nat := block_007_right_left_left ++ block_007_right_left_right
theorem block_007_right_left_checked : primeSegment 13442 512 = block_007_right_left := by
  change primeSegment 13442 (256 + 256) = _
  rw [primeSegment_append, block_007_right_left_left_checked, block_007_right_left_right_checked]
  rfl

def block_007_right_right_left : List Nat := interval_13954 ++ interval_14082
theorem block_007_right_right_left_checked : primeSegment 13954 256 = block_007_right_right_left := by
  change primeSegment 13954 (128 + 128) = _
  rw [primeSegment_append, interval_13954_checked, interval_14082_checked]
  rfl

def block_007_right_right_right : List Nat := interval_14210 ++ interval_14338
theorem block_007_right_right_right_checked : primeSegment 14210 256 = block_007_right_right_right := by
  change primeSegment 14210 (128 + 128) = _
  rw [primeSegment_append, interval_14210_checked, interval_14338_checked]
  rfl

def block_007_right_right : List Nat := block_007_right_right_left ++ block_007_right_right_right
theorem block_007_right_right_checked : primeSegment 13954 512 = block_007_right_right := by
  change primeSegment 13954 (256 + 256) = _
  rw [primeSegment_append, block_007_right_right_left_checked, block_007_right_right_right_checked]
  rfl

def block_007_right : List Nat := block_007_right_left ++ block_007_right_right
theorem block_007_right_checked : primeSegment 13442 1024 = block_007_right := by
  change primeSegment 13442 (512 + 512) = _
  rw [primeSegment_append, block_007_right_left_checked, block_007_right_right_checked]
  rfl

def block_007 : List Nat := block_007_left ++ block_007_right
theorem block_007_checked : primeSegment 12418 2048 = block_007 := by
  change primeSegment 12418 (1024 + 1024) = _
  rw [primeSegment_append, block_007_left_checked, block_007_right_checked]
  rfl

end ProjectEulerSolutions.PrimePairVertices
